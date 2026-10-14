import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

from data.dataset import FrameDataset
from classifier.utils import validate_classifier, train_one_epoch_classifier
from dino.utils import (make_train_transform_dino, make_val_transform_dino, load_dino_backbone, freeze_module,
                        make_mlp_head)
from misc.training import seed_everything, make_run_dir, make_balanced_loaders, make_optimizer, EpochLogger


class DinoClassifier(nn.Module):
    def __init__(self, freeze_backbone=True, num_classes=5):
        super().__init__()

        self.backbone = load_dino_backbone()
        self.embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            freeze_module(self.backbone)

        self.head = make_mlp_head(self.embed_dim, num_classes)

    @classmethod
    def from_checkpoint(cls, path, freeze_backbone=True):
        """Rebuild from a best_model.pth, reading num_classes (1 = binary, 5) off the head."""
        state_dict = torch.load(path, map_location="cpu")
        model = cls(freeze_backbone=freeze_backbone, num_classes=state_dict["head.5.weight"].shape[0])
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(self, x):
        # Classify from the DINO CLS token.
        return self.head(self.backbone.forward_features(x)["x_norm_clstoken"])


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"dino_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{"full" if args.train_json == "frame_train.json" else args.train_json.split(".")[0].split("_")[-1]}")
    run_name, run_dir = make_run_dir("classifier", description, args.run_tag)

    num_classes = 1 if args.binary_classification else 5
    if args.weight_path:
        # Pretrained weights are a 5-class DinoClassifier; swap in a fresh head for binary.
        print(f"Loading pretrained weights from: {args.weight_path}")
        model = DinoClassifier(freeze_backbone=args.freeze_backbone, num_classes=5)
        model.load_state_dict(torch.load(args.weight_path, map_location="cpu"))
        if num_classes != 5:
            model.head = make_mlp_head(model.embed_dim, num_classes)
    else:
        model = DinoClassifier(freeze_backbone=args.freeze_backbone, num_classes=num_classes)
    model = model.to(device)

    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)
    frame_dirs = ["data/2024_Paxos_Frames/cropped_matched_frames"]
    train_dataset = FrameDataset(os.path.join(args.split_path, args.train_json), frame_dirs, transform=train_trans)
    val_dataset = FrameDataset(os.path.join(args.split_path, args.val_json), frame_dirs, transform=val_trans)
    train_loader, val_loader = make_balanced_loaders(train_dataset, val_dataset, args.binary_classification,
                                                     batch_size=args.batch_size, require_all_classes=True)

    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_model.pth",
                         cm_to_tensorboard=True)
    print(f"Start Training: {run_name}")
    for epoch in range(args.epochs):
        t_loss = train_one_epoch_classifier(model, train_loader, criterion, optimizer, args.binary_classification,
                                            device)
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
    parser.add_argument('--weight_path', type=str, default=None,
                        help='5-class DinoClassifier checkpoint to initialize from (head is replaced for --binary_classification).')
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
