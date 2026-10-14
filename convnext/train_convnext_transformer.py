import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse

from data.dataset import MILVideoDataset, MILVideoDatasetCluster
from dino.utils import make_train_transform_dino, make_val_transform_dino
from misc.training import (seed_everything, make_run_dir, video_search_dirs, make_balanced_loaders,
                           make_optimizer, EpochLogger)
from transformer.utils import train_one_epoch_trans, validate_extended_trans
from .model import ConvNextSelfAttention, ADAPTERS


def train_convnext_transformer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"convnext_sa_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{args.adapter}"
                f"{f'_{args.aggregation_mode}' if args.aggregation_mode != 'cls' else ''}"
                f"{'_cluster' if args.cluster_sample else ''}")
    run_name, run_dir = make_run_dir("convnext", description, args.run_tag)
    search_dir_paths = video_search_dirs(args.fused_dataset)

    # --- Model ---
    # Saved as a sidecar so ConvNextSelfAttention.from_checkpoint can rebuild it.
    architecture_config = dict(
        adapter=args.adapter,
        adapter_dim=args.adapter_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        num_frames=args.num_frames,
        aggregation_mode=args.aggregation_mode,
        num_query_tokens=args.num_query_tokens,
        num_classes=1 if args.binary_classification else 5,
    )
    model = ConvNextSelfAttention(freeze_backbone=args.freeze_backbone,
                                  grad_checkpointing=not args.no_grad_checkpointing,
                                  **architecture_config).to(device)

    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(architecture_config, f, indent=2)

    # Fixed per-channel standardization of the backbone feature map, fit once on
    # the pretrained backbone over the training videos (8 uniform frames each,
    # no augmentation). Stored as buffers, so it's saved with the checkpoint.
    if not args.no_feature_standardization:
        stats_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                                   num_frames=8,
                                   transform=make_val_transform_dino(args.img_size, args.complex_augs),
                                   search_dir_paths=search_dir_paths)
        print("Fitting backbone feature standardization on the training videos...")
        model.fit_feature_stats((stats_ds[i][0] for i in range(len(stats_ds))), device)

    # --- Data ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)
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
    # Head + adapter at lr (param_groups[0], used for LR logging), backbone at lr * backbone_lr_mult.
    optimizer = make_optimizer(model, args.lr, args.backbone_lr_mult)
    # Cosine decay per epoch from the starting LRs down to min_lr (applies to both param groups).
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_sa_model.pth")
    print(f"Starting ConvNeXt Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        # No attention-entropy loss term for this experiment; entropy is only logged.
        t_loss, t_attn_entropy = train_one_epoch_trans(
            model, train_loader, criterion, optimizer, args.binary_classification, device, scaler,
            entropy_weight=0.0, distance_scale_weight=args.distance_scale_weight,
        )
        m = validate_extended_trans(model, val_loader, criterion, args.binary_classification, device)
        scheduler.step()
        logger.log_epoch(epoch, model, optimizer, t_loss, m, {'Meta/Attn_Entropy_Train': t_attn_entropy})

    logger.close()
    return run_name, logger.best_selection_metric, logger.best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Suffix appended to the run name (distinguishes runs started in the same minute).')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--no_grad_checkpointing', action='store_true',
                        help='Disable per-block checkpointing of the unfrozen ConvNeXt backbone.')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Final LR of the cosine schedule (reached after --epochs).')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='Backbone LR = --lr * this (only matters when --freeze_backbone is omitted).')
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--no_feature_standardization', action='store_true',
                        help='Skip per-channel standardization of the backbone feature map (see ConvNextSelfAttention).')
    parser.add_argument('--adapter', type=str, default='linear_attnpool', choices=list(ADAPTERS),
                        help='ConvNeXt feature map -> per-frame vector adapter (see convnext/model.py).')
    parser.add_argument('--adapter_dim', type=int, default=1024,
                        help='Adapter output dim = transformer width.')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--aggregation_mode', type=str, default='cls', choices=['cls', 'cls_mean', 'multi_token'])
    parser.add_argument('--num_query_tokens', type=int, default=4,
                        help='Aggregator tokens for --aggregation_mode multi_token (ignored otherwise).')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--distance_scale_weight', type=float, default=0.2,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--cluster_sample', action='store_true',
                        help='Sample one frame per k-means cluster (clustering/cluster_frames.py) instead of by time; Paxos2025 only.')
    parser.add_argument('--binary_classification', action='store_true')

    args = parser.parse_args()
    if args.cluster_sample and args.fused_dataset:
        parser.error('--cluster_sample only has cluster files for Paxos2025, not --fused_dataset')

    train_convnext_transformer(args)
