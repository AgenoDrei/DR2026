import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse

from data.dataset import MILVideoDataset
from dino.utils import make_train_transform_dino, make_bag_train_transforms_dino, make_val_transform_dino, EXTRA_AUGS
from misc.training import (seed_everything, make_run_dir, video_search_dirs, make_balanced_loaders,
                           make_optimizer, EpochLogger, SELECTION_METRICS, make_criterion, add_balancing_args, zoom_settings)
from transformer.utils import train_one_epoch_trans, validate_extended_trans
from transformer.adapters import ADAPTERS
from transformer.model import AGGREGATION_MODES
from .model import ConvNextSelfAttention


def train_convnext_transformer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"convnext{'' if args.backbone == 'large' else '_' + args.backbone}_sa_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{args.adapter}{args.adapter_tokens or ''}"
                f"{f'_s{args.feature_stage}' if args.feature_stage != 4 or args.feature_block != -1 else ''}"
                f"{f'b{args.feature_block}' if args.feature_block != -1 else ''}"
                f"{'_nope' if args.no_adapter_pe else ''}"
                f"{f'_{args.aggregation_mode}' if args.aggregation_mode != 'cls' else ''}"
                f"{'_aug-' + '-'.join(args.extra_augs) if args.extra_augs else ''}"
                f"{f'_bag{args.min_bag_size}' if args.min_bag_size else ''}"
                f"{'_bagaug' if args.bag_consistent_augs else ''}"
                f"{'_noclahe' if not args.clahe else ''}"
                f"{f'_sel-{args.selection_metric}' if args.selection_metric != 'default' else ''}")
    run_name, run_dir = make_run_dir("convnext", description, args.run_tag)
    search_dir_paths = video_search_dirs(args.fused_dataset)

    # --- Model ---
    # Saved as a sidecar so ConvNextSelfAttention.from_checkpoint can rebuild it.
    architecture_config = dict(
        backbone_arch=args.backbone,
        adapter=args.adapter,
        adapter_dim=args.adapter_dim,
        adapter_pe=not args.no_adapter_pe,
        adapter_tokens=args.adapter_tokens,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        num_frames=args.num_frames,
        aggregation_mode=args.aggregation_mode,
        num_query_tokens=args.num_query_tokens,
        feature_stage=args.feature_stage,
        feature_block=args.feature_block,
        num_classes=1 if args.binary_classification else 5,
    )
    model = ConvNextSelfAttention(freeze_backbone=args.freeze_backbone,
                                  grad_checkpointing=not args.no_grad_checkpointing,
                                  **architecture_config).to(device)

    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(architecture_config, f, indent=2)

    # --- Data ---
    # --bag_consistent_augs: random augs drawn once per bag instead of once per frame.
    zoom, train_zoom = zoom_settings(args)
    if args.bag_consistent_augs:
        train_trans, train_bag_trans = make_bag_train_transforms_dino(args.img_size, args.complex_augs, args.extra_augs, args.clahe,
                                                                      train_zoom=train_zoom)
    else:
        train_trans, train_bag_trans = make_train_transform_dino(args.img_size, args.complex_augs, args.extra_augs, args.clahe,
                                                                 train_zoom=train_zoom), None
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs, args.clahe, zoom=zoom)
    train_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                           num_frames=args.num_frames,
                           transform=train_trans,
                           bag_transform=train_bag_trans,
                           random_segment_sample=args.random_segment_sample,
                           search_dir_paths=search_dir_paths, min_bag_size=args.min_bag_size,
                           crop_borders=args.crop_borders)
    val_ds = MILVideoDataset(os.path.join(args.split_path, args.val_json),
                         num_frames=args.num_frames,
                         transform=val_trans,
                         search_dir_paths=search_dir_paths, crop_borders=args.crop_borders)
    train_loader, val_loader = make_balanced_loaders(train_ds, val_ds, args.binary_classification,
                                                     sampler_power=args.sampler_power)

    # Class weights only in the training loss; validation loss stays unweighted.
    train_criterion = make_criterion(train_ds, args.binary_classification, args.class_weight_power).to(device)
    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    # Head + adapter at lr (param_groups[0], used for LR logging), backbone at lr * backbone_lr_mult.
    optimizer = make_optimizer(model, args.lr, args.backbone_lr_mult, args.optimizer, args.weight_decay)
    # Cosine decay per epoch from the starting LRs down to min_lr (applies to both param groups).
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_sa_model.pth",
                         selection=args.selection_metric)
    print(f"Starting ConvNeXt Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        # No attention-entropy loss term for this experiment; entropy is only logged.
        t_loss, t_attn_entropy = train_one_epoch_trans(
            model, train_loader, train_criterion, optimizer, args.binary_classification, device, scaler,
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
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Only used with --optimizer adamw (decoupled weight decay).')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='Backbone LR = --lr * this (only matters when --freeze_backbone is omitted).')
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--backbone', type=str, default='large', choices=['large', 'base'],
                        help='DINOv3 ConvNeXt size (base: stage dims 128/256/512/1024, same depths as large).')
    parser.add_argument('--adapter', type=str, default='linear_attnpool', choices=list(ADAPTERS),
                        help='ConvNeXt feature map -> per-frame token(s) adapter (see transformer/adapters.py).')
    parser.add_argument('--feature_stage', type=int, default=4, choices=[1, 2, 3, 4],
                        help='ConvNeXt stage whose feature map feeds the adapter (4 = final, 1536 ch at H/32; '
                             '3: 768 ch at H/16; 2: 384 ch at H/8; 1: 192 ch at H/4).')
    parser.add_argument('--feature_block', type=int, default=-1,
                        help='Block within --feature_stage to take the features after (-1 = last; stage 3 has 27).')
    parser.add_argument('--adapter_dim', type=int, default=1024,
                        help='Adapter output dim = transformer width.')
    parser.add_argument('--no_adapter_pe', action='store_true',
                        help='Drop the 2D sin/cos PE in the linear_attnpool adapter (content-only pooling).')
    parser.add_argument('--adapter_tokens', type=int, default=None,
                        help='Tokens per frame: queries for linear_attnpool (default 1).')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--aggregation_mode', type=str, default='cls', choices=list(AGGREGATION_MODES))
    parser.add_argument('--num_query_tokens', type=int, default=4,
                        help='Aggregator tokens for --aggregation_mode multi_token (ignored otherwise).')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--selection_metric', type=str, default='default', choices=SELECTION_METRICS,
                        help="Best-checkpoint criterion: 'default' = QWK (5-class) / PR-AUC (binary), "
                             "'combined' = mean of QWK, referable recall and accuracy (5-class only).")
    parser.add_argument('--distance_scale_weight', type=float, default=0.2,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--bag_consistent_augs', action='store_true',
                        help='Draw the random train augmentations once per bag (same for all frames) instead of per frame.')
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--extra_augs', nargs='+', default=[], choices=EXTRA_AUGS,
                        help='Extra train augs with --complex_augs (dino/utils.py): color (stronger jitter + gamma), '
                             'affine (zoom/shift), degrade (blur/JPEG/noise, p 0.2 each), p20_color / p20_quality / p20_light '
                             '(Paxos2020-style look, dino/domain_augs.py; best with --bag_consistent_augs).')
    parser.add_argument('--no_clahe', dest='clahe', action='store_false',
                        help='--complex_augs without CLAHE (train and val; eval with vis_video_results.py --no_clahe).')
    parser.add_argument('--min_bag_size', type=int, default=None,
                        help='Train bags keep a random k in [min_bag_size, num_frames] of the sampled frames (val: all).')
    parser.add_argument('--binary_classification', action='store_true')
    add_balancing_args(parser)

    args = parser.parse_args()
    if args.extra_augs and not args.complex_augs:
        parser.error('--extra_augs needs --complex_augs')
    if args.min_bag_size is not None and not 1 <= args.min_bag_size <= args.num_frames:
        parser.error('--min_bag_size must be in [1, --num_frames]')
    if args.selection_metric == 'combined' and args.binary_classification:
        parser.error('--selection_metric combined (QWK + referable recall + acc) is 5-class only')

    train_convnext_transformer(args)
