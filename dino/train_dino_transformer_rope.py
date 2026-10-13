import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import MILVideoDataset, MILVideoDatasetRope
from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import save_confusion_matrix
from transformer.utils import train_one_epoch_trans_rope, validate_extended_trans_rope


def compute_frame_positions(original_indices, first_informative_idx, last_informative_idx):
    """
    Utility for building the `frame_positions` tensor expected by
    DinoSelfAttention.forward() when use_rope=True.

    original_indices: iterable of original-video frame indices for the 32
        sampled frames (in the same order the frames are fed to the model).
    first_informative_idx / last_informative_idx: original-video indices of
        the first/last informative frame, i.e. after stripping the
        non-informative frames from both ends of the original video.

    Returns a float tensor of shape [len(original_indices)] with values in
    [0, 1], preserving the true relative temporal spacing between the
    sampled frames (not just their rank/order among the 32).
    """
    span = last_informative_idx - first_informative_idx
    if span <= 0:
        raise ValueError("Need at least two distinct informative frames to normalize positions.")
    return torch.tensor(
        [(idx - first_informative_idx) / span for idx in original_indices],
        dtype=torch.float32,
    )


class ContinuousRotaryPositionalEmbeddings(nn.Module):
    """
    RoPE variant that rotates by a continuous (fractional) position per
    token rather than an integer sequence index.

    torchtune.modules.RotaryPositionalEmbeddings only supports integer
    positions: it precomputes a cache of size max_seq_len and indexes into
    it with an integer `input_pos` tensor, so it can't express "this frame
    sits at true position 0.34". This class uses the identical rotation
    math (same theta formula, same interleaved-pair rotation) but computes
    cos/sin on the fly from whatever position tensor you pass in, so
    positions can be any float.
    """

    def __init__(self, dim, base=10_000):
        super().__init__()
        self.dim = dim
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("theta", theta, persistent=False)  # [dim // 2]

    def forward(self, x, positions):
        """
        x: [b, s, n_h, h_d]
        positions: [b, s] or [s] float tensor of continuous positions.
                   e.g. CLS gets 0.0 (-> identity rotation), frames get
                   their normalized true-time positions (scaled as desired).
        """
        b, s, n_h, h_d = x.shape
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(b, -1)  # [b, s]
        positions = positions.to(device=x.device, dtype=torch.float32)

        # [b, s, dim // 2]
        idx_theta = torch.einsum("bs,d->bsd", positions, self.theta)
        cos = torch.cos(idx_theta).unsqueeze(2)  # [b, s, 1, dim // 2] (broadcast over heads)
        sin = torch.sin(idx_theta).unsqueeze(2)

        xshaped = x.float().reshape(b, s, n_h, h_d // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * cos - xshaped[..., 1] * sin,
                xshaped[..., 1] * cos + xshaped[..., 0] * sin,
            ],
            dim=-1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class RoPEMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention(dim, num_heads, batch_first=True)
    self-attention, but applies RoPE (with per-token continuous positions)
    to Q and K before the score matmul.

    Q/K/V projections and attention are implemented manually (rather than via
    F.scaled_dot_product_attention) so we can return the averaged attention
    weights, matching nn.MultiheadAttention's default `average_attn_weights=True`
    behavior that the rest of your code (frame_weights extraction) relies on.
    """

    def __init__(self, dim, num_heads=8, rope_base=10_000, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # RoPE operates per-head, so its `dim` arg is head_dim, not the full
        # embedding dim.
        self.rope = ContinuousRotaryPositionalEmbeddings(dim=self.head_dim, base=rope_base)

    def forward(self, x, positions):
        # x: [b, s, dim]  (s = 1 CLS + num_frames here)
        # positions: [b, s] or [s] float tensor, CLS entry should be 0.0
        b, s, _ = x.shape

        qkv = self.qkv_proj(x).reshape(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [b, s, num_heads, head_dim]

        # Apply RoPE to Q and K only (never V).
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # -> [b, num_heads, s, head_dim] for the attention matmul
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [b, num_heads, s, head_dim]
        out = out.transpose(1, 2).reshape(b, s, self.dim)
        out = self.out_proj(out)

        # Average attention weights across heads to match
        # nn.MultiheadAttention's default return shape [b, s, s]
        attn_weights = attn.mean(dim=1)

        return out, attn_weights


class TransformerBlock(nn.Module):
    """A single MHA + FFN block (post-norm, matching the original design).
    Returns output and the raw attention weights so the caller can inspect
    CLS->frame attention from the last block."""

    def __init__(self, dim, num_heads=8, ffn_dim=2048, dropout=0.1,
                 use_rope=False, rope_base=10_000):
        super().__init__()
        self.use_rope = use_rope

        if use_rope:
            self.mha = RoPEMultiheadAttention(
                dim=dim, num_heads=num_heads, rope_base=rope_base, dropout=dropout,
            )
        else:
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

    def forward(self, x, positions=None):
        if self.use_rope:
            attn_out, attn_weights = self.mha(x, positions)
        else:
            attn_out, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

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
            use_rope=False,
            rope_base=10_000,
            rope_position_scale=None,
            num_frames=32,
            num_classes=1,
            classifier_checkpoint_path=None,
            full_checkpoint_path=None,
    ):
        """
        Two independent, mutually exclusive, strict checkpoint-loading paths:

        - classifier_checkpoint_path: path to a *DinoClassifier* state dict.
          Only the backbone weights are extracted and loaded (strict=True
          against the backbone's own key set). Use this when starting
          Self-Attention training from a fine-tuned DinoClassifier backbone.

        - full_checkpoint_path: path to a *DinoSelfAttention* state dict
          (i.e. one saved from this exact class, with matching
          num_blocks/use_pos_embedding/use_rope/num_frames config). Loaded
          strictly into the whole model. Use this for inference / resuming
          after Self-Attention training has finished.

        Passing both is an error - decide which stage you're in.

        use_pos_embedding and use_rope are also mutually exclusive: they are
        two alternative ways to give the model frame-order information
        (additive learned embedding vs. rotary embedding inside attention).
        RoPE has no learnable parameters and, unlike the additive embedding,
        naturally leaves the CLS token untouched (it's assigned position 0.0,
        which RoPE rotates by angle 0).

        When use_rope=True, forward()/forward_head() require a
        `frame_positions` tensor of shape [num_frames] with values in [0, 1]
        giving each frame's true normalized position in the (informative,
        trimmed) original video - see compute_frame_positions(). These are
        internally multiplied by rope_position_scale before being used as
        rotation angles, since raw [0, 1] values would barely use RoPE's
        frequency spectrum with the default base=10_000 (which was designed
        around integer positions). rope_position_scale defaults to
        num_frames - 1, which puts continuous positions back on the same
        numeric scale as standard integer RoPE over `num_frames` steps.
        """
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError(
                "Specify either classifier_checkpoint_path or full_checkpoint_path, not both."
            )

        if use_pos_embedding and use_rope:
            raise ValueError(
                "use_pos_embedding and use_rope are mutually exclusive ways of "
                "injecting frame order - choose one."
            )

        self.num_blocks = num_blocks
        self.use_pos_embedding = use_pos_embedding
        self.use_rope = use_rope
        self.num_frames = num_frames
        self.num_classes = num_classes
        # Continuous [0, 1] frame positions are scaled up to roughly the
        # same numeric range as standard integer RoPE positions (0..num_frames-1)
        # so that the default base=10_000 hyperparameter stays meaningful.
        self.rope_position_scale = (
            rope_position_scale if rope_position_scale is not None else (num_frames - 1)
        )

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim (e.g., 1024 for ViT-L)

        if self.use_rope and self.L % num_heads != 0:
            raise ValueError(
                f"use_rope requires L ({self.L}) to be divisible by num_heads ({num_heads})."
            )

        # Learnable CLS token that will aggregate information across all frames
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.L))

        # Optional learned positional embedding, one per frame slot.
        # NOT applied to the CLS token, only to the frame tokens, since the
        # frames form the "sequence" with potential ordering/position info
        # and the CLS token is just an aggregation query.
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, self.L))
        else:
            self.pos_embedding = None

        # Stack of Self-Attention Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.L, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout,
                use_rope=use_rope, rope_base=rope_base,
            )
            for _ in range(num_blocks)
        ])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512),
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
            # Fine-tuning the full backbone over a 32-frame bag is otherwise
            # too memory-hungry for a 24GB GPU; recompute activations during
            # backward instead of storing them (only engages in train mode).
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
        blocks + classifier). Strict: requires an exact key/shape match,
        i.e. the model must have been constructed with the same
        num_blocks/use_pos_embedding/use_rope/num_frames as when the
        checkpoint was saved."""
        print(f"Loading full DinoSelfAttention checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")

        self.load_state_dict(state_dict, strict=True)
        print(f"Full model weights loaded strictly ({len(state_dict)} tensors).")

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

            if self.use_rope:
                nn.init.xavier_uniform_(block.mha.qkv_proj.weight)
                nn.init.constant_(block.mha.qkv_proj.bias, 0)
                nn.init.xavier_uniform_(block.mha.out_proj.weight)
                nn.init.constant_(block.mha.out_proj.bias, 0)
            else:
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
            f"pos_embedding={self.use_pos_embedding}, rope={self.use_rope}] weights initialized ---"
        )

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS
    # ---------------------------------------------------------
    def extract_features(self, x):
        # Runs ONLY the DINO backbone to cache features.
        feats = self.backbone.forward_features(x)
        return feats["x_norm_clstoken"]

    def forward_head(self, h, frame_positions=None):
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

        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(h.size(0), -1, -1)  # [1, 1, L]

        # Prepend CLS token to the frame features sequence
        x = torch.cat((cls_tokens, h), dim=1)  # [1, num_frames + 1, L]

        positions = None
        if self.use_rope:
            if frame_positions is None:
                raise ValueError(
                    "use_rope=True requires a `frame_positions` tensor of shape "
                    "[num_frames] with values in [0, 1] (see compute_frame_positions())."
                )
            n = frame_positions.numel()
            if n != self.num_frames:
                raise ValueError(
                    f"Got {n} frame_positions but model was initialized for "
                    f"num_frames={self.num_frames}."
                )
            frame_positions = frame_positions.to(device=x.device, dtype=torch.float32)
            frame_positions = frame_positions * self.rope_position_scale
            cls_position = torch.zeros((1, 1), device=x.device, dtype=torch.float32)
            # [num_frames + 1] -> CLS at position 0.0 (identity rotation)
            positions = torch.cat([cls_position, frame_positions], dim=1)

        # Run through the stack of self-attention blocks.
        # Keep the attention weights from the LAST block only, since that's
        # the one whose CLS-token attention best reflects what actually
        # drove the final classification.
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x, positions=positions)

        # Extract the processed CLS token embedding for classification
        bag_representation = x[:, 0, :]  # [1, L]
        logits = self.classifier(bag_representation)  # [1, 1]

        # Extract attention weights originating from the CLS token to all target video frames
        # attn_weights shape: [batch_size, queries, keys] -> [1, num_frames+1, num_frames+1]
        # Query 0 is the CLS token. Keys 1: are the video frames.
        frame_weights = attn_weights[0, 0, 1:]  # [num_frames]

        # Normalize weights over the frames to sum to 1 (preserves expected MIL downstream properties)
        frame_weights = frame_weights / (frame_weights.sum() + 1e-12)

        # Returns logits, frame weights, and raw attention vector for max-attention strategy compatibility
        return logits, frame_weights, frame_weights

    def forward(self, x, frame_positions=None):
        """
        x: [num_frames, 3, H, W]
        frame_positions: required iff use_rope=True. Float tensor of shape
            [num_frames], values in [0, 1], giving each frame's true
            normalized position in the trimmed/informative original video,
            in the same order as the frames in `x`. See
            compute_frame_positions() for how to build this from the
            original-video frame indices you're already tracking.
        """
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h, frame_positions=frame_positions)

        return logits, weights


def train_self_attention(args):
    if args.use_pos_embedding and args.use_rope:
        parser.error("--use_pos_embedding and --use_rope are mutually exclusive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    description = (f"dino_sa"
                f"_{"binary" if args.binary_classification else "all"}"
                f"_{args.split_path.split('/')[-1]}"
                f"{f"_rope_{args.rope_base}_{args.rope_position_scale}" if args.use_rope else ""}"
                f"{"_posEmb" if args.use_pos_embedding else ""}")
    run_name = f"{timestamp}_{description}" + (f"_{args.run_tag}" if args.run_tag else "")

    run_dir = os.path.join("transformer", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))


    # --- Model ---
    model = DinoSelfAttention(classifier_checkpoint_path=args.classifier_checkpoint,
                              freeze_backbone=args.freeze_backbone,
                              num_blocks=args.num_blocks,
                              use_pos_embedding=args.use_pos_embedding,
                              use_rope=args.use_rope,
                              rope_base=args.rope_base,
                              rope_position_scale=args.rope_position_scale,
                              num_classes=1 if args.binary_classification else 5).to(device)

    # --- Transforms ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    # --- Datasets ---
    if args.fused_dataset:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos", "data/ensemble_results_paxos2020/cleaned_videos"]
    else:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos"]

    # --- Datasets ---
    if args.use_rope:
        train_ds = MILVideoDatasetRope(os.path.join(args.split_path, args.train_json),
                                   num_frames=32,
                                   transform=train_trans,
                                   random_segment_sample=args.random_segment_sample,
                                   search_dir_paths=search_dir_paths)
        val_ds = MILVideoDatasetRope(os.path.join(args.split_path, args.val_json),
                                 num_frames=32,
                                 transform=val_trans,
                                 search_dir_paths=search_dir_paths)
    else:
        train_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                                   num_frames=32,
                                   transform=train_trans,
                                   random_segment_sample=args.random_segment_sample,
                                   search_dir_paths=search_dir_paths)
        val_ds = MILVideoDataset(os.path.join(args.split_path, args.val_json),
                                 num_frames=32,
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    # Mixed precision: fine-tuning the full (unfrozen) ViT-L backbone over a
    # 32-frame bag in fp32 exceeds 24GB. autocast + GradScaler cuts
    # activation memory substantially without changing the training config.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_selection_metric = 0.0

    print(f"Starting Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_trans_rope(model, train_loader, criterion, optimizer, args.binary_classification, device, scaler)
        m = validate_extended_trans_rope(model, val_loader, criterion, args.binary_classification, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save confusion matrix
        class_names = ['non-referable', 'referable'] if args.binary_classification else ['0', '1', '2', '3', '4']
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir, class_names=class_names)

        # Save best model
        selection_metric = m['qwk'] if not args.binary_classification else m['pr_auc']
        if selection_metric > best_selection_metric:
            best_selection_metric = selection_metric
            torch.save(model.state_dict(), os.path.join(run_dir, "best_sa_model.pth"))
            writer.add_scalar('Meta/Best_Selection_Metric', best_selection_metric, epoch)

        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained DINO classifier weights')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--use_pos_embedding', action='store_true')
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--rope_base', type=float, default=100)
    parser.add_argument('--rope_position_scale', type=float, default=300)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional short suffix appended to the run name (e.g. an experiment label).')

    args = parser.parse_args()

    train_self_attention(args)
