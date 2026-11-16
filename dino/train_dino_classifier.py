import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse

from data.dataset import FrameDataset
from classifier.utils import validate_classifier, train_one_epoch_classifier
from dino.utils import (make_train_transform_dino, make_val_transform_dino, load_dino_backbone, freeze_module,
                        make_mlp_head, DINO_WEIGHTS, DINO_WEIGHTS_SAT)
from transformer.adapters import build_adapter, retina_patch_mask
from misc.training import (seed_everything, make_run_dir, make_balanced_loaders, make_optimizer, EpochLogger,
                           make_criterion, add_balancing_args, zoom_settings)


class DinoClassifier(nn.Module):
    """DINOv3 backbone + MLP head on the CLS token, or with dense_adapter ('linear_mean' / 'linear_max')
    on the patch-token grid pooled by that transformer/adapters.py adapter (black border patches masked)."""

    def __init__(self, freeze_backbone=True, num_classes=5, weights=DINO_WEIGHTS, dense_adapter=None, adapter_dim=1024):
        super().__init__()

        self.backbone = load_dino_backbone(weights)
        self.embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            freeze_module(self.backbone)

        self.dense_adapter = dense_adapter
        self.adapter = build_adapter(dense_adapter, self.embed_dim, adapter_dim) if dense_adapter else None
        self.head = make_mlp_head(adapter_dim if dense_adapter else self.embed_dim, num_classes)

    @classmethod
    def from_checkpoint(cls, path, freeze_backbone=True):
        """Rebuild from a best_model.pth, reading num_classes (1 = binary, 5) off the head and the dense
        adapter from classifier_config.json next to it (CLS model if absent)."""
        state_dict = torch.load(path, map_location="cpu")
        weights = DINO_WEIGHTS_SAT if "backbone.local_cls_norm.weight" in state_dict else DINO_WEIGHTS
        cfg_path = os.path.join(os.path.dirname(path), "classifier_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        if "adapter.proj.weight" in state_dict and not cfg.get("dense_adapter"):
            raise ValueError(f"{path} has a dense adapter but no classifier_config.json naming it")
        model = cls(freeze_backbone=freeze_backbone, num_classes=state_dict["head.5.weight"].shape[0], weights=weights,
                    dense_adapter=cfg.get("dense_adapter"), adapter_dim=cfg.get("adapter_dim", 1024))
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if self.adapter is None:
            return self.head(feats["x_norm_clstoken"])
        patches = feats["x_norm_patchtokens"]                       # N×(h·w)×C
        side = int(patches.size(1) ** 0.5)
        fmap = patches.transpose(1, 2).reshape(x.size(0), -1, side, side)
        if getattr(self.adapter, "uses_mask", False):
            return self.head(self.adapter(fmap, retina_patch_mask(x, x.size(-1) // side)))
        return self.head(self.adapter(fmap))


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
        model = DinoClassifier(freeze_backbone=args.freeze_backbone, num_classes=5, weights=args.backbone_weights,
                               dense_adapter=args.dense_adapter, adapter_dim=args.adapter_dim)
        model.load_state_dict(torch.load(args.weight_path, map_location="cpu"))
        if num_classes != 5:
            model.head = make_mlp_head(model.head[0].in_features, num_classes)
    else:
        model = DinoClassifier(freeze_backbone=args.freeze_backbone, num_classes=num_classes, weights=args.backbone_weights,
                               dense_adapter=args.dense_adapter, adapter_dim=args.adapter_dim)
    with open(os.path.join(run_dir, "classifier_config.json"), "w") as f:
        json.dump({"dense_adapter": args.dense_adapter, "adapter_dim": args.adapter_dim}, f, indent=2)
    model.backbone.grad_checkpointing = args.grad_checkpointing
    model = model.to(device)

    zoom, train_zoom = zoom_settings(args)
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs, train_zoom=train_zoom)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs, zoom=zoom)
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
    optimizer = make_optimizer(model, args.lr, optimizer=args.optimizer, weight_decay=args.weight_decay)
    if args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

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
    parser.add_argument('--weight_path', type=str, default=None,
                        help='5-class DinoClassifier checkpoint to initialize from (head is replaced for --binary_classification).')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--backbone_weights', type=str, default=DINO_WEIGHTS,
                        help='Pretrained DINOv3 ViT-L/16 weights (e.g. dino.utils.DINO_WEIGHTS_SAT, satellite pretraining).')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--dense_adapter', type=str, default=None, choices=['linear_mean', 'linear_max'],
                        help='Classify from the patch-token grid pooled by this transformer/adapters.py adapter instead of CLS.')
    parser.add_argument('--adapter_dim', type=int, default=1024, help='--dense_adapter output dim (head input).')

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10,
                        help='StepLR step size in epochs (only for --lr_schedule step).')
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cosine'],
                        help="'step' = StepLR(--lr_step, gamma 0.1), 'cosine' = cosine annealing to --min_lr over --epochs.")
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Final LR of the cosine schedule (only for --lr_schedule cosine).')
    parser.add_argument('--distance_scale_weight', type=float, default=0.0,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Only used with --optimizer adamw (decoupled weight decay).')
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--grad_checkpointing', action='store_true',
                        help='Per-block gradient checkpointing in the backbone (unfrozen bs 16 at 512px: ~7GB instead of OOM on 24GB).')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')
    add_balancing_args(parser)

    args = parser.parse_args()

    train_classifier(args)
