import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import MILVideoDataset
from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import save_confusion_matrix
from mil.utils import train_one_epoch_mil, validate_extended_mil


class DinoMIL(nn.Module):
    def __init__(
            self,
            checkpoint_path=None,
            freeze_backbone=True,
            D=512,
            K=1
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, D),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(D, K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._init_weights()

        # Override with checkpoint weights if provided
        if checkpoint_path is not None:
            print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Load backbone weights
            backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            if backbone_state:
                result = self.backbone.load_state_dict(backbone_state, strict=False)
                print("Missing:", result.missing_keys)
                print("Backbone weights loaded.")

            # Load MIL Head weights (Only does something during evaluation)
            head_state = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
            if head_state:
                result = self.load_state_dict(head_state, strict=False)
                print("Missing:", result.missing_keys)
                print("MIL Head weights loaded.")

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self):
        for m in [self.attention_V, self.attention_U, self.attention_w, self.classifier]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        print(f"--- DINO MIL [{self.L} features] weights initialized ---")

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS (For 4-way script)
    # ---------------------------------------------------------
    def extract_features(self, x):
        # Runs ONLY the DINO backbone to cache features.
        feats = self.backbone.forward_features(x)
        return feats["x_norm_clstoken"]

    def forward_head(self, h):
        # Runs ONLY the MIL Head on pre-computed features.
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        a = self.attention_w(a_v * a_u)  # [N, K]

        weights = torch.softmax(a, dim=0)  # over frames
        bag_representation = torch.sum(weights * h, dim=0)  # [L]
        logits = self.classifier(bag_representation.unsqueeze(0))  # [1,1]

        # Returns raw_a for the 'Max Attention' evaluation strategy
        return logits, weights, a.flatten()

    def forward(self, x):
        """
        x: [num_frames, 3, H, W]
        """
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)

        return logits, weights


def train_mil(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    description = f"dino_mil_{'complex' if args.complex_augs else 'simple'}_{args.split_path.split('/')[-1]}_{'segmented' if args.random_segment_sample else 'uniform'}"
    run_name = f"{timestamp}_{description}" + (f"_{args.run_tag}" if args.run_tag else "")

    run_dir = os.path.join("mil", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # --- Model ---
    model = DinoMIL(checkpoint_path=args.classifier_checkpoint, freeze_backbone=args.freeze_backbone).to(device)

    # --- Transforms ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    # --- Datasets ---
    if args.fused_dataset:
        search_dir_paths = ["data/ensemble_results/cleaned_videos", "data/ensemble_results_paxos2020/cleaned_videos"]
    else:
        search_dir_paths = ["data/ensemble_results/cleaned_videos"]

    train_ds = MILVideoDataset(os.path.join(args.split_path, "mil_train.json"),
                               num_frames=32,
                               transform=train_trans,
                               random_segment_sample=args.random_segment_sample,
                               search_dir_paths = search_dir_paths)
    val_ds = MILVideoDataset(os.path.join(args.split_path, "mil_val.json"),
                             num_frames=32,
                             transform=val_trans,
                             search_dir_paths=search_dir_paths)

    # --- Setup Oversampling ---
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels)

    # Compute inverse class frequencies
    class_weights = 1.0 / class_counts

    # Map each video in the dataset to its class weight
    sample_weights = class_weights[train_labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Use sampler in DataLoader (shuffle MUST be set to False)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=sampler,
        shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    best_pr_auc = 0.0

    print(f"Starting MIL Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_mil(model, train_loader, criterion, optimizer, device)
        m = validate_extended_mil(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save confusion matrix
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir)

        # Save best model
        if m['pr_auc'] > best_pr_auc:
            best_pr_auc = m['pr_auc']
            torch.save(model.state_dict(), os.path.join(run_dir, "best_mil_model.pth"))

        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)

        writer.add_scalar('Metric/Accuracy', m['acc'], epoch)
        writer.add_scalar('Metric/F1', m['f1'], epoch)
        writer.add_scalar('Metric/Precision', m['precision'], epoch)
        writer.add_scalar('Metric/Recall', m['recall'], epoch)
        writer.add_scalar('Metric/PR_AUC', m['pr_auc'], epoch)
        writer.add_scalar('Metric/ROC_AUC', m['roc_auc'], epoch)

        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {t_loss:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {m['f1']:.3f} | "
            f"PR-AUC: {m['pr_auc']:.3f}"
        )

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained DINO classifier weights')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')

    args = parser.parse_args()

    train_mil(args)