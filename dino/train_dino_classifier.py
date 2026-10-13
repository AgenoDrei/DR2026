import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
import os
import numpy as np
import argparse
from PIL import Image

from data.dataset import FrameDataset
from misc.utils import save_confusion_matrix
from classifier.utils import validate_classifier, train_one_epoch_classifier
from dino.utils import make_train_transform_dino, make_val_transform_dino


class DinoClassifier(nn.Module):
    def __init__(self, freeze_backbone=True, num_classes=5):
        super().__init__()

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        # Perform classification based on the class token
        x = feats["x_norm_clstoken"]
        return self.head(x)


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    description = (f"dino_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{"full" if args.train_json == "frame_train.json" else args.train_json.split(".")[0].split("_")[-1]}")
    run_name = (f"{timestamp}_{description}"
                f"{f'_{args.run_tag}' if args.run_tag else ''}")

    run_dir = os.path.join("classifier", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # Step 1: Initialize model with ORIGINAL training setup (5 classes)
    model = DinoClassifier(
        freeze_backbone=args.freeze_backbone,
        num_classes=5
    ).to(device)

    # Step 2: Load pretrained weights if provided
    if args.weight_path:
        print(f"Loading pretrained weights from: {args.weight_path}")
        state_dict = torch.load(args.weight_path, map_location=device)

        model.load_state_dict(state_dict)
        print("Successfully loaded pretrained DINO weights.")

    if args.binary_classification:
        # Step 3: Replace head for NEW task (binary = 1 output)
        model.head = nn.Sequential(
                nn.Linear(model.embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ).to(device)
        model._init_weights()

    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    train_dataset = FrameDataset(os.path.join(args.split_path, args.train_json),
                                 ["data/2024_Paxos_Frames/cropped_matched_frames"],
                                 transform=train_trans)
    val_dataset = FrameDataset(os.path.join(args.split_path, args.val_json),
                               ["data/2024_Paxos_Frames/cropped_matched_frames"],
                               transform=val_trans)

    if args.binary_classification:
        train_labels = np.array(train_dataset.labels)
        class_counts = np.bincount(train_labels)
        assert len(class_counts) == 2, f"Expected 2 classes, got {len(class_counts)} — some label missing from split"
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
    else:
        train_grades = np.array(train_dataset.grades)
        class_counts = np.bincount(train_grades)
        assert len(class_counts) == 5, f"Expected 5 classes, got {len(class_counts)} — some grade missing from split"
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_grades]

    # Convert sample weights to a PyTorch DoubleTensor
    sample_weights = torch.from_numpy(sample_weights).double()

    # 3. Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Pass the sampler to DataLoader (Note: shuffle MUST be False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Handles random sampling weighted by class distribution
        shuffle=False,  # Set shuffle to False when using a custom sampler
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if args.binary_classification:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    class_names = ['non-referable', 'referable'] if args.binary_classification else ['0', '1', '2', '3', '4']
    best_selection_metric  = 0.0
    print(f"Start Training: {run_name}")
    for epoch in range(args.epochs):
        t_loss = train_one_epoch_classifier(model, train_loader, criterion, optimizer, args.binary_classification,
                                            device)
        m = validate_classifier(model, val_loader, criterion, args.binary_classification, device)

        scheduler.step()

        cm_path, cm = save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir, class_names=class_names)
        cm_img = np.array(Image.open(cm_path).convert('RGB')).transpose(2, 0, 1)  # HWC -> CHW for TB
        writer.add_image('ConfusionMatrix', cm_img, epoch)

        selection_metric = m['qwk'] if not args.binary_classification else m['pr_auc']
        if selection_metric > best_selection_metric:
            best_selection_metric = selection_metric
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            writer.add_scalar('Meta/Best_Selection_Metric', best_selection_metric, epoch)

        writer.add_scalar('Meta/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)

        # Log every scalar metric generically -- covers both binary keys (f1, precision, ...)
        # and multiclass keys (f1_macro, f1_class0, roc_auc_class3, ...) without branching.
        skip_keys = {'loss', 'y_true', 'y_pred'}
        for key, value in m.items():
            if key in skip_keys:
                continue
            writer.add_scalar(f'Metric/{key}', value, epoch)

        # Console summary -- pick the headline aggregates so the print stays readable
        acc = m['acc']
        f1_headline = m['f1'] if args.binary_classification else m['f1_macro']
        headline_metric_name = 'PR-AUC' if args.binary_classification else 'QWK'
        headline_metric_value = m['pr_auc'] if args.binary_classification else m['qwk']
        print(
            f"Epoch {epoch} | "
            f"Loss: {t_loss:.3f} | "
            f"Val Loss: {m['loss']:.3f} | "
            f"Acc: {acc:.3f} | "
            f"F1: {f1_headline:.3f} | "
            f"{headline_metric_name}: {headline_metric_value:.3f}"
        )

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="frame_train.json")
    parser.add_argument('--val_json', type=str, default="frame_val.json")
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained DINO classifier weights')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')

    args = parser.parse_args()

    train_classifier(args)
