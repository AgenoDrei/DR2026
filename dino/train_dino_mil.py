import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

from data.dataset import MILVideoDataset
from .utils import (make_train_transform_dino, make_val_transform_dino, load_dino_backbone,
                    load_backbone_from_classifier, freeze_module, make_mlp_head)
from misc.training import (seed_everything, make_run_dir, video_search_dirs, make_balanced_loaders,
                           make_optimizer, EpochLogger)
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class DinoMIL(nn.Module):
    """DINOv3 ViT-L backbone -> gated-attention MIL pooling over the frame bag -> binary MLP head.

    classifier_checkpoint_path loads only backbone.* weights from a DinoClassifier
    checkpoint; full_checkpoint_path loads a full DinoMIL state dict. Both strict."""

    def __init__(
            self,
            freeze_backbone=True,
            D=512,
            K=1,
            classifier_checkpoint_path=None,
            full_checkpoint_path=None,
    ):
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError("Specify either classifier_checkpoint_path or full_checkpoint_path, not both.")

        self.backbone = load_dino_backbone()
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
        self.classifier = make_mlp_head(self.L, 1)

        for layer in (self.attention_V[0], self.attention_U[0], self.attention_w):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        print(f"--- DINO MIL [{self.L} features] weights initialized ---")

        if classifier_checkpoint_path is not None:
            load_backbone_from_classifier(self.backbone, classifier_checkpoint_path)
        elif full_checkpoint_path is not None:
            print(f"Loading full DinoMIL checkpoint: {full_checkpoint_path}")
            self.load_state_dict(torch.load(full_checkpoint_path, map_location="cpu"), strict=True)

        if freeze_backbone:
            freeze_module(self.backbone)

    def extract_features(self, x):
        # [num_frames, 3, H, W] -> [num_frames, L] DINO CLS tokens.
        return self.backbone.forward_features(x)["x_norm_clstoken"]

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
    seed_everything(args.seed)

    description = (f"dino_mil_{'complex' if args.complex_augs else 'simple'}_{args.split_path.split('/')[-1]}_"
                   f"{'segmented' if args.random_segment_sample else 'uniform'}")
    run_name, run_dir = make_run_dir("mil", description, args.run_tag)

    # --- Model ---
    model = DinoMIL(classifier_checkpoint_path=args.classifier_checkpoint,
                    freeze_backbone=args.freeze_backbone).to(device)

    # --- Data ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)
    search_dir_paths = video_search_dirs(args.fused_dataset)
    train_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                               num_frames=32,
                               transform=train_trans,
                               random_segment_sample=args.random_segment_sample,
                               search_dir_paths=search_dir_paths)
    val_ds = MILVideoDataset(os.path.join(args.split_path, args.val_json),
                             num_frames=32,
                             transform=val_trans,
                             search_dir_paths=search_dir_paths)
    # Binary only: the MIL head has a single output.
    train_loader, val_loader = make_balanced_loaders(train_ds, val_ds, binary_classification=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model, args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    # Mixed precision, as for the transformer, so an unfrozen backbone fits in 24GB.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    logger = EpochLogger(run_name, run_dir, args, True, "best_mil_model.pth")
    print(f"Starting MIL Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss, _ = train_one_epoch_trans(model, train_loader, criterion, optimizer, True, device, scaler)
        m = validate_extended_trans(model, val_loader, criterion, True, device)
        scheduler.step()
        logger.log_epoch(epoch, model, optimizer, t_loss, m)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
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
