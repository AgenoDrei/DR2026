import json
import os

import torch
import torch.nn as nn

from dino.utils import make_mlp_head


def sparsemax(z, dim=-1):
    """Sparsemax (Martins & Astudillo 2016): Euclidean projection onto the simplex, exact zeros for low scores."""
    z_sorted, _ = z.sort(dim=dim, descending=True)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    k = k.view([-1 if d == (dim % z.dim()) else 1 for d in range(z.dim())])
    cssv = z_sorted.cumsum(dim) - 1
    k_z = (z_sorted - cssv / k > 0).sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, k_z - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class TransformerBlock(nn.Module):
    """A single MHA + FFN block (pre-norm, matching standard ViT). Returns
    output and the raw attention weights so the caller can inspect
    CLS->frame attention from the last block."""

    def __init__(self, dim, num_heads=8, ffn_dim=2048, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
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


ANY = object()  # removed option whose stored value doesn't matter for the remaining architecture

# architecture_config.json keys of options removed from BagTransformer, with the value that matches the current behavior.
REMOVED_HEAD_OPTIONS = {"aggregation_mode": "sparse_pool", "num_query_tokens": ANY, "use_pos_embedding": False,
                        "pool_temp": None, "frame_prior": False, "alternating_attention": False}


class BagTransformer(nn.Module):
    """Transformer aggregator over a bag of per-frame vectors, shared by DinoSelfAttention and ConvNextSelfAttention.
    Subclasses build self.backbone, call _build_head(), and implement extract_features
    ([num_frames, 3, H, W] -> [num_frames, dim]).

    A query token takes part in the blocks; the head reads a sparsemax attention pooling over the frame tokens after
    the blocks (exact zeros for ignored frames). frame_weights (attention viz, distance-scaled loss) are the pooling weights.

    head_dim: LayerNorm + Linear(dim -> head_dim) + dropout on the frame tokens first; blocks and MLP head then run at
    head_dim. attn_dropout: dropout on the blocks' attention weights. Both only used by dino/train_cached_transformer.py."""

    # Subclasses extend this with their own removed options (see from_checkpoint).
    removed_config = REMOVED_HEAD_OPTIONS

    def _build_head(self, dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes,
                    final_norm=False, head_dim=None, attn_dropout=0.0):
        self.in_proj = None
        if head_dim:
            self.in_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, head_dim), nn.Dropout(dropout))
            dim = head_dim

        self.L = dim
        self.num_blocks = num_blocks
        self.num_frames = num_frames
        self.num_classes = num_classes

        # Query token (named cls_token for checkpoint compatibility).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if final_norm else None

        # Per-token pooling score, zero-initialized so pooling starts uniform.
        self.pool_score = nn.Linear(dim, 1)
        nn.init.zeros_(self.pool_score.weight)
        nn.init.zeros_(self.pool_score.bias)

        self.classifier = make_mlp_head(dim, num_classes)

        for block in self.blocks:
            for layer in block.ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
            nn.init.xavier_uniform_(block.mha.in_proj_weight)
            nn.init.constant_(block.mha.in_proj_bias, 0)
            nn.init.xavier_uniform_(block.mha.out_proj.weight)
            nn.init.constant_(block.mha.out_proj.bias, 0)
        nn.init.normal_(self.cls_token, std=1e-6)

    def _load_full_checkpoint(self, path):
        """Strict: the model must have the same architecture config as when the checkpoint was saved."""
        print(f"Loading full {type(self).__name__} checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        print(f"Full model weights loaded strictly ({len(state_dict)} tensors).")

    @classmethod
    def from_checkpoint(cls, path, checkpoint_name="best_sa_model.pth",
                        config_name="architecture_config.json"):
        """Rebuild a model from a run dir (or a checkpoint file inside one) with the architecture_config.json sidecar
        saved at training time. Keys of removed options are dropped if they hold the current behavior's value."""
        run_dir, ckpt = (path, os.path.join(path, checkpoint_name)) if os.path.isdir(path) \
            else (os.path.dirname(path), path)
        with open(os.path.join(run_dir, config_name), "r") as f:
            config = json.load(f)
        for key, value in cls.removed_config.items():
            stored = config.pop(key, value)
            if value is not ANY and stored != value:
                raise ValueError(f"{run_dir} was trained with the removed option {key}={stored!r}.")
        return cls(full_checkpoint_path=ckpt, **config)

    def forward_head(self, h):
        """h: [num_frames, L] -> (logits [1, C], frame_weights [num_frames], region_weights [num_frames, 1])."""
        if self.in_proj is not None:
            h = self.in_proj(h)
        x = torch.cat((self.cls_token, h.unsqueeze(0)), dim=1)  # [1, 1 + num_frames, L]
        for block in self.blocks:
            x, _ = block(x)
        if self.final_norm is not None:
            x = self.final_norm(x)

        frame_out = x[:, 1:, :]  # [1, num_frames, L]
        scores = self.pool_score(frame_out).squeeze(-1).float()  # [1, num_frames]
        pool_weights = sparsemax(scores, dim=-1)  # sums to 1
        bag_representation = (pool_weights.unsqueeze(-1).to(frame_out.dtype) * frame_out).sum(dim=1)  # [1, L]
        logits = self.classifier(bag_representation)

        frame_weights = pool_weights[0]
        return logits, frame_weights, frame_weights.unsqueeze(1)

    def forward(self, x):
        """x: [num_frames, 3, H, W] -> (logits, frame_weights)"""
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)
        return logits, weights
