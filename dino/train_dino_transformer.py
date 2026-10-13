import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import MILVideoDataset, MILVideoDatasetCluster
from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import save_confusion_matrix
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class TransformerBlock(nn.Module):
    """A single MHA + FFN block (pre-norm, matching standard ViT). Returns
    output and the raw attention weights so the caller can inspect
    CLS->frame attention from the last block."""

    def __init__(self, dim, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.mha(normed, normed, normed)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x, attn_weights


class DinoSelfAttention(nn.Module):
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
        """
        classifier_checkpoint_path and full_checkpoint_path are mutually
        exclusive: the former loads only backbone.* weights (strict) from a
        DinoClassifier checkpoint, the latter loads this class's own full
        state dict (strict - architecture must match exactly).

        aggregation_mode controls how the classifier head reads the
        post-attention sequence:
        - "cls" (default): only the final CLS token.
        - "cls_mean": cat(CLS token, mean-pooled frame tokens).
        - "multi_token": num_query_tokens learnable tokens, concatenated.
        frame_weights (used for attention viz and the entropy/distance-scale
        loss terms) is the query token(s)' attention to the frames, averaged
        over query tokens - identical to the single-CLS computation when
        num_query_tokens=1.
        """
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError(
                "Specify either classifier_checkpoint_path or full_checkpoint_path, not both."
            )
        if aggregation_mode not in ("cls", "cls_mean", "multi_token"):
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

        self.num_blocks = num_blocks
        self.use_pos_embedding = use_pos_embedding
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.aggregation_mode = aggregation_mode
        # Only "multi_token" uses more than one; kept as "cls_token" for
        # checkpoint compatibility with earlier single-CLS-token runs.
        self.num_query_tokens = num_query_tokens if aggregation_mode == "multi_token" else 1

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim (e.g., 1024 for ViT-L)

        # Learnable query/aggregator token(s): [1, 1, L] unless multi_token.
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_query_tokens, self.L))

        # Optional learned positional embedding, per frame slot (not applied
        # to the query token(s)).
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, self.L))
        else:
            self.pos_embedding = None

        # Stack of Self-Attention Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.L, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Head input dim depends on aggregation_mode (see forward_head).
        if self.aggregation_mode == "cls_mean":
            classifier_in_dim = 2 * self.L
        elif self.aggregation_mode == "multi_token":
            classifier_in_dim = self.num_query_tokens * self.L
        else:
            classifier_in_dim = self.L

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

        # --- Checkpoint loading (mutually exclusive, both strict) ---
        if classifier_checkpoint_path is not None:
            self._load_classifier_backbone(classifier_checkpoint_path)
        elif full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

        # Freeze backbone (applied after loading, so it applies regardless
        # of which checkpoint path was used, or none at all)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # Fine-tuning the full backbone over a frame bag is too
            # memory-hungry otherwise; recompute activations in backward
            # instead of storing them (only engages in train mode).
            self.backbone.grad_checkpointing = True

    def _load_classifier_backbone(self, path):
        """Load only the backbone weights from a DinoClassifier checkpoint.
        Strict: the checkpoint's backbone.* keys must exactly match this
        model's backbone's own key set, or this raises."""
        print(f"Loading backbone from DinoClassifier checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")

        backbone_state = {
            k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")
        }
        if not backbone_state:
            raise ValueError(
                f"No keys starting with 'backbone.' found in {path}. "
                f"Is this really a DinoClassifier checkpoint?"
            )

        self.backbone.load_state_dict(backbone_state, strict=True)
        print(f"Backbone weights loaded strictly ({len(backbone_state)} tensors).")

    def _load_full_checkpoint(self, path):
        """Load a full DinoSelfAttention checkpoint (backbone + cls_token +
        blocks + classifier). Strict: the model must have been constructed
        with the same architecture config as when the checkpoint was saved."""
        print(f"Loading full DinoSelfAttention checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")

        self.load_state_dict(state_dict, strict=True)
        print(f"Full model weights loaded strictly ({len(state_dict)} tensors).")

    @classmethod
    def from_checkpoint(cls, run_dir, checkpoint_name="best_sa_model.pth",
                         config_name="architecture_config.json"):
        """Reconstruct a model from a run_dir written by train_self_attention,
        using the architecture-config sidecar saved next to the checkpoint
        instead of requiring the caller to already know/guess num_blocks,
        aggregation_mode, etc."""
        with open(os.path.join(run_dir, config_name), "r") as f:
            config = json.load(f)
        return cls(full_checkpoint_path=os.path.join(run_dir, checkpoint_name), **config)

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize each block's FFN + MHA
        for block in self.blocks:
            for layer in block.ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

            nn.init.xavier_uniform_(block.mha.in_proj_weight)
            if block.mha.in_proj_bias is not None:
                nn.init.constant_(block.mha.in_proj_bias, 0)
            nn.init.xavier_uniform_(block.mha.out_proj.weight)
            if block.mha.out_proj.bias is not None:
                nn.init.constant_(block.mha.out_proj.bias, 0)

        # Initialize CLS Token
        nn.init.normal_(self.cls_token, std=1e-6)

        # Initialize positional embedding, if used
        if self.pos_embedding is not None:
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        print(
            f"--- DINO Self-Attention [{self.L} features, {self.num_blocks} block(s), "
            f"pos_embedding={self.use_pos_embedding}, aggregation={self.aggregation_mode}"
            f"{f' (num_query_tokens={self.num_query_tokens})' if self.aggregation_mode == 'multi_token' else ''}"
            f"] weights initialized ---"
        )

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS
    # ---------------------------------------------------------
    def extract_features(self, x):
        # Runs ONLY the DINO backbone to cache features.
        feats = self.backbone.forward_features(x)
        return feats["x_norm_clstoken"]

    def forward_head(self, h):
        # h shape: [num_frames, L] -> e.g., [32, 1024]
        # Expand to pseudo-batch format [batch_size=1, num_frames, L]
        h = h.unsqueeze(0)

        if self.pos_embedding is not None:
            n = h.size(1)
            if n != self.num_frames:
                raise ValueError(
                    f"Got {n} frames but positional embedding was initialized "
                    f"for num_frames={self.num_frames}."
                )
            h = h + self.pos_embedding

        N = self.num_query_tokens

        query_tokens = self.cls_token.expand(h.size(0), -1, -1)  # [1, N, L]
        x = torch.cat((query_tokens, h), dim=1)  # [1, N + num_frames, L]

        # Attention weights kept from the last block only (reflects what
        # actually drove the final classification).
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x)

        query_out = x[:, :N, :]   # [1, N, L]
        frame_out = x[:, N:, :]   # [1, num_frames, L]

        # Build the bag representation per aggregation_mode (see __init__).
        if self.aggregation_mode == "cls_mean":
            bag_representation = torch.cat([query_out[:, 0, :], frame_out.mean(dim=1)], dim=-1)  # [1, 2L]
        elif self.aggregation_mode == "multi_token":
            bag_representation = query_out.reshape(query_out.size(0), -1)  # [1, N*L]
        else:  # "cls"
            bag_representation = query_out[:, 0, :]  # [1, L]

        logits = self.classifier(bag_representation)  # [1, num_classes]

        # Query token(s)' attention to the frames, averaged over query
        # tokens, normalized to sum to 1 over the frames.
        frame_weights = attn_weights[0, :N, N:].mean(dim=0)  # [num_frames]
        frame_weights = frame_weights / (frame_weights.sum() + 1e-12)

        return logits, frame_weights, frame_weights

    def forward(self, x):
        """
        x: [num_frames, 3, H, W]
        """
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)

        return logits, weights


def train_self_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Best-effort reproducibility (weight init, WeightedRandomSampler,
    # dropout, random_segment_sample) - not bitwise-exact since cudnn's own
    # algorithm choice is left non-deterministic for speed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    description = (f"dino_sa_"
                f"{"binary" if args.binary_classification else "all"}_"
                f"{args.split_path.split('/')[-1]}_"
                f"{"full" if args.train_json == "mil_train.json" else args.train_json.split(".")[0].split("_")[-1]}"
                f"{f'_{args.aggregation_mode}' if args.aggregation_mode != 'cls' else ''}"
                f"{'_cluster' if args.cluster_sample else ''}")
    run_name = (f"{timestamp}_{description}"
                f"{f'_{args.run_tag}' if args.run_tag else ''}")

    run_dir = os.path.join("transformer", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # --- Model ---
    # Kept as its own dict (rather than inlined kwargs) so the exact same
    # architecture config can be saved as a sidecar next to the checkpoint -
    # see DinoSelfAttention.from_checkpoint, which reloads a model from a
    # run_dir without the caller needing to know/guess these by hand.
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

    # Architecture doesn't change across epochs, so this is written once and
    # always matches whatever best_sa_model.pth ends up saved in run_dir.
    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(architecture_config, f, indent=2)

    # --- Transforms ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    # --- Datasets ---
    if args.fused_dataset:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos", "data/ensemble_results_paxos2020/cleaned_videos"]
    else:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos"]

    if args.cluster_sample:
        # One random frame per cluster for training, cluster medoids for validation
        train_ds = MILVideoDatasetCluster(os.path.join(args.split_path, args.train_json),
                                          num_frames=args.num_frames,
                                          transform=train_trans,
                                          random_segment_sample=True,
                                          search_dir_paths=search_dir_paths)
        val_ds = MILVideoDatasetCluster(os.path.join(args.split_path, args.val_json),
                                        num_frames=args.num_frames,
                                        transform=val_trans,
                                        search_dir_paths=search_dir_paths)
    else:
        train_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                                   num_frames=args.num_frames,
                                   transform=train_trans,
                                   random_segment_sample=args.random_segment_sample,
                                   search_dir_paths=search_dir_paths)
        val_ds = MILVideoDataset(os.path.join(args.split_path, args.val_json),
                                 num_frames=args.num_frames,
                                 transform=val_trans,
                                 search_dir_paths=search_dir_paths)

    # --- Setup Oversampling ---
    if args.binary_classification:
        train_targets = np.array(train_ds.labels)
    else:
        train_targets = np.array(train_ds.grades)
    class_counts = np.bincount(train_targets)

    # Compute inverse class frequencies
    class_weights = 1.0 / class_counts

    # Map each video in the dataset to its class weight
    sample_weights = class_weights[train_targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Use sampler in DataLoader (shuffle MUST be set to False)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=sampler,
        shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()

    # Backbone gets its own (usually smaller) LR when fine-tuned.
    # param_groups[0] stays the "head" group so the LR logging below
    # (which reads param_groups[0]) is unaffected.
    backbone_param_ids = {id(p) for p in model.backbone.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_mult})
    optimizer = optim.Adam(param_groups, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    # Mixed precision: an unfrozen ViT-L backbone over a frame bag in fp32
    # exceeds 24GB; autocast + GradScaler cuts activation memory.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_selection_metric = 0.0
    best_metrics = None

    print(f"Starting Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss, t_attn_entropy = train_one_epoch_trans(
            model, train_loader, criterion, optimizer, args.binary_classification, device, scaler,
            entropy_weight=args.attn_entropy_weight, distance_scale_weight=args.distance_scale_weight,
        )
        m = validate_extended_trans(model, val_loader, criterion, args.binary_classification, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save confusion matrix
        class_names = ['non-referable', 'referable'] if args.binary_classification else ['0', '1', '2', '3', '4']
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir, class_names=class_names)

        # Save best model
        selection_metric = m['qwk'] if not args.binary_classification else m['pr_auc']
        if selection_metric > best_selection_metric:
            best_selection_metric = selection_metric
            best_metrics = {k: v for k, v in m.items() if k not in ('y_true', 'y_pred')}
            torch.save(model.state_dict(), os.path.join(run_dir, "best_sa_model.pth"))
            writer.add_scalar('Meta/Best_Selection_Metric', best_selection_metric, epoch)

        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)
        if len(optimizer.param_groups) > 1:
            writer.add_scalar('Meta/Backbone_Learning_Rate', optimizer.param_groups[1]['lr'], epoch)

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)
        writer.add_scalar('Meta/Attn_Entropy_Train', t_attn_entropy, epoch)

        skip_keys = {'loss', 'y_true', 'y_pred'}
        for key, value in m.items():
            if key in skip_keys:
                continue
            writer.add_scalar(f'Metric/{key}', value, epoch)

        f1_headline = m['f1'] if args.binary_classification else m['f1_macro']
        headline_metric_name = 'PR-AUC' if args.binary_classification else 'QWK'
        headline_metric_value = m['pr_auc'] if args.binary_classification else m['qwk']
        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {t_loss:.3f} | "
            f"Val Loss: {m['loss']:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {f1_headline:.3f} | "
            f"{headline_metric_name}: {headline_metric_value:.3f}"
        )

    writer.close()
    print(f"Done. Run directory: {run_dir}")
    return run_name, best_selection_metric, best_metrics


if __name__ == "__main__":
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
    parser.add_argument('--distance_scale_weight', type=float, default=0.0,
                        help='Distance-Scaled CE weight: loss *= 1 + w*|pred_grade-true_grade| (5-class only). 0 disables.')
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--cluster_sample', action='store_true',
                        help='Sample one frame per k-means cluster (clustering/cluster_frames.py) instead of by time; Paxos2025 only.')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')

    args = parser.parse_args()
    if args.cluster_sample and args.fused_dataset:
        parser.error('--cluster_sample only has cluster files for Paxos2025, not --fused_dataset')

    train_self_attention(args)