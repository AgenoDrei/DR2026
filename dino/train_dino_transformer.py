import torch
import torch.nn as nn
import os
import json
import argparse

from data.dataset import MILVideoDataset
from .utils import make_train_transform_dino, make_val_transform_dino, load_dino_backbone, load_backbone_from_classifier, freeze_module
from misc.training import (seed_everything, make_run_dir, video_search_dirs, make_balanced_loaders, make_optimizer,
                           make_cosine_scheduler, EpochLogger, make_criterion, add_balancing_args, zoom_settings)
from transformer.model import BagTransformer, REMOVED_HEAD_OPTIONS, ANY
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class DinoSelfAttention(BagTransformer):
    """DINOv3 ViT-L backbone (CLS token per frame) -> BagTransformer head.

    classifier_checkpoint_path and full_checkpoint_path are mutually exclusive: the former loads only backbone.*
    weights (strict) from a DinoClassifier checkpoint, the latter this class's own full state dict (strict; prefer
    DinoSelfAttention.from_checkpoint(run_dir))."""

    # Options of older runs' architecture_config.json (dense adapters, RETFound, narrow head) removed 2026-11-17.
    removed_config = {**REMOVED_HEAD_OPTIONS, "dense_adapter": None, "dense_tokens": ANY, "adapter_dim": ANY,
                      "adapter_pe": ANY, "adapter_heads": ANY, "adapter_attn_temp": ANY, "grid_pool": ANY,
                      "head_dim": None, "backbone": "dinov3", "backbone_img_size": ANY}

    def __init__(
            self,
            freeze_backbone=True,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1,
            num_blocks=2,
            num_frames=32,
            num_classes=5,
            classifier_checkpoint_path=None,
            full_checkpoint_path=None,
    ):
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError("Specify either classifier_checkpoint_path or full_checkpoint_path, not both.")

        self.backbone = load_dino_backbone()
        self._build_head(self.backbone.embed_dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes)
        print(f"--- DINO Self-Attention [{self.L} features, {self.num_blocks} block(s), sparse_pool] weights initialized ---")

        if classifier_checkpoint_path is not None:
            load_backbone_from_classifier(self.backbone, classifier_checkpoint_path)
        elif full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

        if freeze_backbone:
            freeze_module(self.backbone)
        else:
            # Fine-tuning the backbone over a 32-frame bag only fits in 24GB with activation recomputation.
            self.backbone.grad_checkpointing = True

    def extract_features(self, x):
        # [num_frames, 3, H, W] -> [num_frames, L] DINO CLS tokens
        return self.backbone.forward_features(x)["x_norm_clstoken"]


def train_self_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"dino_sa_"
                   f"{'binary' if args.binary_classification else 'all'}_"
                   f"{args.split_path.split('/')[-1]}_"
                   f"{'full' if args.train_json == 'mil_train.json' else args.train_json.split('.')[0].split('_')[-1]}")
    run_name, run_dir = make_run_dir("transformer", description, args.run_tag)

    # --- Model ---
    # Saved as a sidecar next to the checkpoint so DinoSelfAttention.from_checkpoint can rebuild the model from run_dir alone.
    architecture_config = dict(
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        num_frames=args.num_frames,
        num_classes=1 if args.binary_classification else 5,
    )
    model = DinoSelfAttention(classifier_checkpoint_path=args.classifier_checkpoint,
                              freeze_backbone=args.freeze_backbone,
                              **architecture_config).to(device)
    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(architecture_config, f, indent=2)

    # --- Data ---
    zoom, train_zoom = zoom_settings(args)
    search_dir_paths = video_search_dirs(args.fused_dataset)
    train_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                               num_frames=args.num_frames,
                               transform=make_train_transform_dino(args.img_size, train_zoom=train_zoom),
                               random_segment_sample=True,
                               search_dir_paths=search_dir_paths,
                               crop_borders=args.crop_borders)
    val_ds = MILVideoDataset(os.path.join(args.split_path, args.val_json),
                             num_frames=args.num_frames,
                             transform=make_val_transform_dino(args.img_size, complex_augs=True, zoom=zoom),
                             search_dir_paths=search_dir_paths, crop_borders=args.crop_borders)
    train_loader, val_loader = make_balanced_loaders(train_ds, val_ds, args.binary_classification,
                                                     sampler_power=args.sampler_power)

    # Class weights only in the training loss; validation loss stays unweighted.
    train_criterion = make_criterion(train_ds, args.binary_classification, args.class_weight_power).to(device)
    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr, args.backbone_lr_mult, args.weight_decay)
    scheduler = make_cosine_scheduler(optimizer, args.epochs, args.min_lr)
    # Mixed precision: an unfrozen ViT-L backbone over a frame bag in fp32 exceeds 24GB.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Best checkpoint by mean(QWK, referable recall, acc) for 5-class, PR-AUC for binary.
    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_sa_model.pth",
                         selection="default" if args.binary_classification else "combined")
    print(f"Starting Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_trans(model, train_loader, train_criterion, optimizer, args.binary_classification,
                                       device, scaler, distance_scale_weight=args.distance_scale_weight)
        m = validate_extended_trans(model, val_loader, criterion, args.binary_classification, device)
        scheduler.step()
        logger.log_epoch(epoch, model, optimizer, t_loss, m)

    logger.close()
    return run_name, logger.best_selection_metric, logger.best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                        help='Trained DinoClassifier (best_model.pth) whose backbone initializes this model.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seeds weight init, WeightedRandomSampler, and the random-segment frame sampling.')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--fused_dataset', action='store_true', help='Also search the Paxos2020 clips.')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='Backbone LR = --lr * this (only without --freeze_backbone).')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Final LR of the cosine schedule.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay.')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--distance_scale_weight', type=float, default=0.3,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')
    add_balancing_args(parser)
    train_self_attention(parser.parse_args())
