import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse

from data.dataset import MILVideoDataset, MILVideoDatasetCluster
from .utils import (make_train_transform_dino, make_val_transform_dino, load_dino_backbone,
                    load_backbone_from_classifier, freeze_module)
from misc.training import (seed_everything, make_run_dir, video_search_dirs, make_balanced_loaders,
                           make_optimizer, EpochLogger)
from transformer.model import BagTransformer
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class DinoSelfAttention(BagTransformer):
    """DINOv3 ViT-L backbone (CLS token per frame) -> BagTransformer head.

    classifier_checkpoint_path and full_checkpoint_path are mutually
    exclusive: the former loads only backbone.* weights (strict) from a
    DinoClassifier checkpoint, the latter loads this class's own full
    state dict (strict - architecture must match exactly; prefer
    DinoSelfAttention.from_checkpoint(run_dir))."""

    def __init__(
            self,
            freeze_backbone=True,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1,
            num_blocks=1,
            use_pos_embedding=False,
            num_frames=32,
            num_classes=1,
            aggregation_mode="cls",
            num_query_tokens=1,
            classifier_checkpoint_path=None,
            full_checkpoint_path=None,
    ):
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError(
                "Specify either classifier_checkpoint_path or full_checkpoint_path, not both."
            )

        self.backbone = load_dino_backbone()
        self._build_head(self.backbone.embed_dim, num_heads, ffn_dim, dropout, num_blocks, num_frames,
                         num_classes, aggregation_mode, num_query_tokens, use_pos_embedding=use_pos_embedding)
        print(
            f"--- DINO Self-Attention [{self.L} features, {self.num_blocks} block(s), "
            f"pos_embedding={self.use_pos_embedding}, aggregation={self.aggregation_mode}"
            f"{f' (num_query_tokens={self.num_query_tokens})' if self.aggregation_mode == 'multi_token' else ''}"
            f"] weights initialized ---"
        )

        if classifier_checkpoint_path is not None:
            load_backbone_from_classifier(self.backbone, classifier_checkpoint_path)
        elif full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

        # Applied after loading, regardless of which checkpoint (if any) was used.
        if freeze_backbone:
            freeze_module(self.backbone)
        else:
            # Fine-tuning the full backbone over a frame bag is too
            # memory-hungry otherwise; recompute activations in backward
            # instead of storing them (only engages in train mode).
            self.backbone.grad_checkpointing = True

    def extract_features(self, x):
        # [num_frames, 3, H, W] -> [num_frames, L] DINO CLS tokens.
        return self.backbone.forward_features(x)["x_norm_clstoken"]


def train_self_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"dino_sa_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{"full" if args.train_json == "mil_train.json" else args.train_json.split(".")[0].split("_")[-1]}"
                f"{f'_{args.aggregation_mode}' if args.aggregation_mode != 'cls' else ''}"
                f"{'_cluster' if args.cluster_sample else ''}")
    run_name, run_dir = make_run_dir("transformer", description, args.run_tag)

    # --- Model ---
    # Saved as a sidecar next to the checkpoint so DinoSelfAttention.from_checkpoint
    # can rebuild the model from run_dir alone.
    architecture_config = dict(
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        use_pos_embedding=args.use_pos_embedding,
        num_frames=args.num_frames,
        aggregation_mode=args.aggregation_mode,
        num_query_tokens=args.num_query_tokens,
        num_classes=1 if args.binary_classification else 5,
    )
    model = DinoSelfAttention(classifier_checkpoint_path=args.classifier_checkpoint,
                              freeze_backbone=args.freeze_backbone,
                              **architecture_config).to(device)
    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(architecture_config, f, indent=2)

    # --- Data ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)
    search_dir_paths = video_search_dirs(args.fused_dataset)
    # Cluster sampling: one random frame per cluster for training, cluster medoids for validation.
    dataset_cls = MILVideoDatasetCluster if args.cluster_sample else MILVideoDataset
    train_ds = dataset_cls(os.path.join(args.split_path, args.train_json),
                           num_frames=args.num_frames,
                           transform=train_trans,
                           random_segment_sample=args.cluster_sample or args.random_segment_sample,
                           search_dir_paths=search_dir_paths)
    val_ds = dataset_cls(os.path.join(args.split_path, args.val_json),
                         num_frames=args.num_frames,
                         transform=val_trans,
                         search_dir_paths=search_dir_paths)
    train_loader, val_loader = make_balanced_loaders(train_ds, val_ds, args.binary_classification)

    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr, args.backbone_lr_mult, args.optimizer, args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    # Mixed precision: an unfrozen ViT-L backbone over a frame bag in fp32
    # exceeds 24GB; autocast + GradScaler cuts activation memory.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_sa_model.pth")
    print(f"Starting Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss, t_attn_entropy = train_one_epoch_trans(
            model, train_loader, criterion, optimizer, args.binary_classification, device, scaler,
            entropy_weight=args.attn_entropy_weight, distance_scale_weight=args.distance_scale_weight,
        )
        m = validate_extended_trans(model, val_loader, criterion, args.binary_classification, device)
        scheduler.step()
        logger.log_epoch(epoch, model, optimizer, t_loss, m, {'Meta/Attn_Entropy_Train': t_attn_entropy})

    logger.close()
    return run_name, logger.best_selection_metric, logger.best_metrics


def build_parser():
    # Also used by dino/hp_sweep.py, so sweep trials always get every argument's default.
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seeds weight init, WeightedRandomSampler, and random_segment_sample.')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Only used with --optimizer adamw (decoupled weight decay).')
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--use_pos_embedding', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Attention heads per transformer block.')
    parser.add_argument('--ffn_dim', type=int, default=2048,
                        help='Feed-forward hidden dim inside each transformer block.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout used in the transformer blocks (attention output + FFN).')
    parser.add_argument('--aggregation_mode', type=str, default='cls', choices=['cls', 'cls_mean', 'multi_token'],
                        help='How the classifier head reads the post-attention sequence (see DinoSelfAttention docstring).')
    parser.add_argument('--num_query_tokens', type=int, default=4,
                        help='Aggregator tokens for --aggregation_mode multi_token (ignored otherwise).')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='Backbone LR = --lr * this (only matters when --freeze_backbone is omitted).')
    parser.add_argument('--attn_entropy_weight', type=float, default=0.0,
                        help='Entropy-minimization weight on CLS->frame attention (peakier attention). 0 disables.')
    parser.add_argument('--distance_scale_weight', type=float, default=0.2,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--cluster_sample', action='store_true',
                        help='Sample one frame per k-means cluster (clustering/cluster_frames.py) instead of by time; Paxos2025 only.')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.cluster_sample and args.fused_dataset:
        parser.error('--cluster_sample only has cluster files for Paxos2025, not --fused_dataset')

    train_self_attention(args)
