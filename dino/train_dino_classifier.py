import torch
import torch.nn as nn
import os
import argparse

from data.dataset import FrameDataset
from classifier.utils import validate_classifier, train_one_epoch_classifier
from dino.utils import make_train_transform_dino, make_val_transform_dino, load_dino_backbone, make_mlp_head
from misc.training import (seed_everything, make_run_dir, make_balanced_loaders, make_optimizer, make_cosine_scheduler,
                           EpochLogger, make_criterion, add_balancing_args, zoom_settings)


class DinoClassifier(nn.Module):
    """Unfrozen DINOv3 ViT-L/16 backbone + MLP head on the CLS token."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = load_dino_backbone()
        self.embed_dim = self.backbone.embed_dim
        self.head = make_mlp_head(self.embed_dim, num_classes)

    @classmethod
    def from_checkpoint(cls, path):
        """Rebuild from a best_model.pth, reading num_classes (1 = binary, 5) off the head."""
        state_dict = torch.load(path, map_location="cpu")
        model = cls(num_classes=state_dict["head.5.weight"].shape[0])
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(self, x):
        return self.head(self.backbone.forward_features(x)["x_norm_clstoken"])


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"dino_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{"full" if args.train_json == "frame_train.json" else args.train_json.split(".")[0].split("_")[-1]}")
    run_name, run_dir = make_run_dir("classifier", description, args.run_tag)

    model = DinoClassifier(num_classes=1 if args.binary_classification else 5).to(device)

    zoom, train_zoom = zoom_settings(args)
    train_trans = make_train_transform_dino(args.img_size, train_zoom=train_zoom)
    val_trans = make_val_transform_dino(args.img_size, complex_augs=True, zoom=zoom)
    frame_dirs = ["data/2024_Paxos_Frames/cropped_matched_frames"]
    train_dataset = FrameDataset(os.path.join(args.split_path, args.train_json), frame_dirs, transform=train_trans,
                                 crop_borders=args.crop_borders)
    val_dataset = FrameDataset(os.path.join(args.split_path, args.val_json), frame_dirs, transform=val_trans,
                               crop_borders=args.crop_borders)
    train_loader, val_loader = make_balanced_loaders(train_dataset, val_dataset, args.binary_classification,
                                                     batch_size=args.batch_size, require_all_classes=True,
                                                     sampler_power=args.sampler_power)

    # Class weights only in the training loss; validation loss stays unweighted.
    train_criterion = make_criterion(train_dataset, args.binary_classification, args.class_weight_power).to(device)
    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr, weight_decay=args.weight_decay)
    scheduler = make_cosine_scheduler(optimizer, args.epochs, args.min_lr)

    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_model.pth",
                         cm_to_tensorboard=True)
    print(f"Start Training: {run_name}")
    for epoch in range(args.epochs):
        t_loss = train_one_epoch_classifier(model, train_loader, train_criterion, optimizer, args.binary_classification,
                                            device, args.distance_scale_weight)
        m = validate_classifier(model, val_loader, criterion, args.binary_classification, device)
        scheduler.step()
        logger.log_epoch(epoch, model, optimizer, t_loss, m)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="frame_train.json")
    parser.add_argument('--val_json', type=str, default="frame_val.json")
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Final LR of the cosine schedule.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay.')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--distance_scale_weight', type=float, default=0.2,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')
    add_balancing_args(parser)

    args = parser.parse_args()

    train_classifier(args)
