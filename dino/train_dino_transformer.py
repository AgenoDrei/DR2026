import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse

from data.dataset import MILVideoDataset
from .utils import (make_train_transform_dino, make_bag_train_transforms_dino, make_val_transform_dino, load_backbone, BACKBONES, EXTRA_AUGS,
                    load_backbone_from_classifier, freeze_module)
from misc.training import (seed_everything, make_run_dir, video_search_dirs, make_balanced_loaders,
                           make_optimizer, EpochLogger, SELECTION_METRICS, make_criterion, add_balancing_args, zoom_settings)
from transformer.adapters import ADAPTERS, build_adapter, retina_patch_mask
from transformer.model import AGGREGATION_MODES, BagTransformer
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class DinoSelfAttention(BagTransformer):
    """DINOv3 ViT-L backbone (CLS token per frame) -> BagTransformer head.
    backbone='retfound_dinov2' swaps in RETFound_dinov2_meh (ViT-L/14, inputs resized to backbone_img_size,
    dino/utils.py::RetFoundBackbone).

    dense_adapter (a transformer/adapters.py name) instead feeds the 32x32
    patch-token grid (512px input) through that adapter, giving one or
    dense_tokens (attnpool queries / grid side) tokens per frame, with a
    final LayerNorm after the blocks (as in ConvNextSelfAttention).

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
            dense_adapter=None,
            dense_tokens=None,
            adapter_dim=1024,
            adapter_pe=True,
            adapter_heads=4,
            head_dim=None,
            pool_temp=None,
            classifier_checkpoint_path=None,
            full_checkpoint_path=None,
            backbone="dinov3",
            backbone_img_size=224,
            backbone_weights=None,
    ):
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError(
                "Specify either classifier_checkpoint_path or full_checkpoint_path, not both."
            )

        self.backbone = load_backbone(backbone, backbone_img_size, backbone_weights)
        self.dense_adapter = dense_adapter
        if dense_adapter is None:
            self.adapter = None
            dim, tokens_per_frame = self.backbone.embed_dim, 1
        else:
            self.adapter = build_adapter(dense_adapter, self.backbone.embed_dim, adapter_dim, dense_tokens,
                                         adapter_pe, num_heads=adapter_heads)
            dim, tokens_per_frame = adapter_dim, self.adapter.tokens_per_frame
        self._build_head(dim, num_heads, ffn_dim, dropout, num_blocks, num_frames,
                         num_classes, aggregation_mode, num_query_tokens, use_pos_embedding=use_pos_embedding,
                         final_norm=dense_adapter is not None, tokens_per_frame=tokens_per_frame,
                         head_dim=head_dim, pool_temp=pool_temp)
        print(
            f"--- DINO Self-Attention [{self.L} features, {self.num_blocks} block(s), "
            f"{f'dense={dense_adapter} ({tokens_per_frame} token(s)/frame), ' if dense_adapter else ''}"
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

    def backbone_feature_map(self, x):
        """[N, 3, H, W] -> [N, C, h, w] grid of normed patch tokens (h = w = H/16 for DINOv3)."""
        tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]  # [N, h*w, C]
        side = int(tokens.size(1) ** 0.5)
        return tokens.transpose(1, 2).reshape(x.size(0), -1, side, side)

    def extract_features(self, x):
        # [num_frames, 3, H, W] -> [num_frames, L] DINO CLS tokens, or the
        # adapter's [num_frames, (K,) L] tokens from the patch grid.
        if self.adapter is None:
            return self.backbone.forward_features(x)["x_norm_clstoken"]
        fmap = self.backbone_feature_map(x)
        if getattr(self.adapter, "uses_mask", False):
            return self.adapter(fmap, retina_patch_mask(x, x.size(-1) // fmap.size(-1)))
        return self.adapter(fmap)


def train_self_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    description = (f"dino_sa_"
                f"{f'{args.backbone}-{args.backbone_img_size}_' if args.backbone != 'dinov3' else ''}"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{"full" if args.train_json == "mil_train.json" else args.train_json.split(".")[0].split("_")[-1]}"
                f"{f'_{args.aggregation_mode}' if args.aggregation_mode != 'cls' else ''}"
                f"{f'_dense-{args.dense_adapter}{args.dense_tokens or ""}' if args.dense_adapter else ''}"
                f"{'_nope' if args.no_adapter_pe else ''}"
                f"{f'_h{args.adapter_heads}' if args.adapter_heads != 4 else ''}"
                f"{f'_alr{args.adapter_lr_mult:g}' if args.adapter_lr_mult != 1 else ''}"
                f"{f'_hd{args.head_dim}' if args.head_dim else ''}"
                f"{f'_pt{args.pool_temp:g}' if args.pool_temp else ''}"
                f"{'_noclf' if args.classifier_checkpoint is None else ''}"
                f"{'_aug-' + '-'.join(args.extra_augs) if args.extra_augs else ''}"
                f"{f'_bag{args.min_bag_size}' if args.min_bag_size else ''}"
                f"{'_bagaug' if args.bag_consistent_augs else ''}"
                f"{f'_sel-{args.selection_metric}' if args.selection_metric != 'default' else ''}")
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
        dense_adapter=args.dense_adapter,
        dense_tokens=args.dense_tokens,
        adapter_dim=args.adapter_dim,
        adapter_pe=not args.no_adapter_pe,
        adapter_heads=args.adapter_heads,
        head_dim=args.head_dim,
        pool_temp=args.pool_temp,
        num_classes=1 if args.binary_classification else 5,
        backbone=args.backbone,
        backbone_img_size=args.backbone_img_size,
    )
    # backbone_weights stays out of architecture_config: checkpoints hold the full backbone state dict.
    model = DinoSelfAttention(classifier_checkpoint_path=args.classifier_checkpoint,
                              backbone_weights=args.backbone_weights,
                              freeze_backbone=args.freeze_backbone,
                              **architecture_config).to(device)
    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(architecture_config, f, indent=2)

    # --- Data ---
    # --bag_consistent_augs: random augs drawn once per bag instead of once per frame.
    zoom, train_zoom = zoom_settings(args)
    if args.bag_consistent_augs:
        train_trans, train_bag_trans = make_bag_train_transforms_dino(args.img_size, args.complex_augs, args.extra_augs,
                                                                      train_zoom=train_zoom)
    else:
        train_trans, train_bag_trans = make_train_transform_dino(args.img_size, args.complex_augs, args.extra_augs,
                                                                 train_zoom=train_zoom), None
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs, zoom=zoom)
    search_dir_paths = video_search_dirs(args.fused_dataset)
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
    optimizer = make_optimizer(model, args.lr, args.backbone_lr_mult, args.optimizer, args.weight_decay,
                               adapter_lr_mult=args.adapter_lr_mult, pool_lr_mult=args.pool_lr_mult)
    if args.lr_schedule == 'cosine':
        # Per-epoch cosine decay down to min_lr (same as convnext/train_convnext_transformer.py).
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.3)
    # Mixed precision: an unfrozen ViT-L backbone over a frame bag in fp32
    # exceeds 24GB; autocast + GradScaler cuts activation memory.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    logger = EpochLogger(run_name, run_dir, args, args.binary_classification, "best_sa_model.pth",
                         selection=args.selection_metric)
    print(f"Starting Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss, t_attn_entropy = train_one_epoch_trans(
            model, train_loader, train_criterion, optimizer, args.binary_classification, device, scaler,
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

    parser.add_argument('--backbone', type=str, default='dinov3', choices=BACKBONES,
                        help="Frame encoder: DINOv3 ViT-L/16 or RETFound_dinov2_meh (ViT-L/14).")
    parser.add_argument('--backbone_img_size', type=int, default=224,
                        help='RETFound input size (frames are resized from --img_size inside the backbone); multiple of 14.')
    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--backbone_weights', type=str, default=None,
                        help='Other DINOv3 ViT-L/16 weights (e.g. dino/ssl_lora/<run>/merged_backbone.pth); '
                             'overridden by --classifier_checkpoint.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seeds weight init, WeightedRandomSampler, and random_segment_sample.')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10,
                        help='StepLR step size in epochs (only for --lr_schedule step).')
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cosine'],
                        help="'step' = StepLR(--lr_step, gamma 0.3), 'cosine' = cosine annealing to --min_lr over --epochs.")
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Final LR of the cosine schedule (only for --lr_schedule cosine).')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.001,
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
    parser.add_argument('--aggregation_mode', type=str, default='cls', choices=list(AGGREGATION_MODES),
                        help='How the classifier head reads the post-attention sequence (see DinoSelfAttention docstring).')
    parser.add_argument('--num_query_tokens', type=int, default=4,
                        help='Aggregator tokens for --aggregation_mode multi_token (ignored otherwise).')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--dense_adapter', type=str, default=None, choices=list(ADAPTERS),
                        help='Use the patch-token grid through this adapter (transformer/adapters.py) instead of CLS tokens.')
    parser.add_argument('--dense_tokens', type=int, default=None,
                        help='Tokens per frame for --dense_adapter: queries for linear_attnpool (default 1).')
    parser.add_argument('--adapter_dim', type=int, default=1024,
                        help='--dense_adapter output dim = transformer width.')
    parser.add_argument('--no_adapter_pe', action='store_true',
                        help='Drop the 2D sin/cos PE in the linear_attnpool dense adapter (content-only pooling).')
    parser.add_argument('--adapter_heads', type=int, default=4,
                        help='Attention heads of the linear_attnpool dense adapter.')
    parser.add_argument('--head_dim', type=int, default=None,
                        help='Project frame tokens to this width (LayerNorm + Linear + dropout) before the transformer blocks; '
                             'set --num_heads/--ffn_dim to match (e.g. 384 / 6 / 768).')
    parser.add_argument('--pool_temp', type=float, default=None,
                        help='sparse_pool: learnable score temperature, initialized at this value (e.g. 10); off by default.')
    parser.add_argument('--pool_lr_mult', type=float, default=1.0,
                        help='sparse_pool scorer (+ temperature) lr = lr * this (own param group).')
    parser.add_argument('--adapter_lr_mult', type=float, default=1.0,
                        help='Dense adapter lr = lr * this (own param group).')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='Backbone LR = --lr * this (only matters when --freeze_backbone is omitted).')
    parser.add_argument('--attn_entropy_weight', type=float, default=0.0,
                        help='Entropy-minimization weight on CLS->frame attention (peakier attention). 0 disables.')
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
                             'affine (zoom/shift), degrade (blur/JPEG/noise, p 0.2 each).')
    parser.add_argument('--min_bag_size', type=int, default=None,
                        help='Train bags keep a random k in [min_bag_size, num_frames] of the sampled frames (val: all).')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')
    add_balancing_args(parser)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.extra_augs and not args.complex_augs:
        parser.error('--extra_augs needs --complex_augs')
    if args.min_bag_size is not None and not 1 <= args.min_bag_size <= args.num_frames:
        parser.error('--min_bag_size must be in [1, --num_frames]')
    if args.min_bag_size is not None and args.use_pos_embedding:
        parser.error('--min_bag_size varies the bag length; --use_pos_embedding needs exactly --num_frames frames')
    if args.selection_metric == 'combined' and args.binary_classification:
        parser.error('--selection_metric combined (QWK + referable recall + acc) is 5-class only')
    if args.no_adapter_pe and args.dense_adapter != 'linear_attnpool':
        parser.error('--no_adapter_pe needs --dense_adapter linear_attnpool')

    if args.backbone != 'dinov3' and args.backbone_weights:
        parser.error('--backbone_weights holds DINOv3 ViT-L/16 weights, only for --backbone dinov3')
    if args.backbone != 'dinov3' and args.classifier_checkpoint:
        parser.error('--classifier_checkpoint holds DINOv3 classifier weights, only for --backbone dinov3')

    train_self_attention(args)
