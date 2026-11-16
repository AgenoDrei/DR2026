import json
import os
import re

import torch
import torch.nn as nn

from dino.utils import make_mlp_head


AGGREGATION_MODES = ("cls", "cls_mean", "multi_token", "sparse_pool")


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


class BagTransformer(nn.Module):
    """Transformer aggregator over a bag of per-frame vectors, shared by
    DinoSelfAttention and ConvNextSelfAttention. Subclasses build
    self.backbone, call _build_head(), and implement extract_features
    ([num_frames, 3, H, W] -> [num_frames, dim], or [num_frames, K, dim] for
    K region tokens per frame, which all attend to each other).

    aggregation_mode controls how the classifier head reads the
    post-attention sequence:
    - "cls" (default): only the final CLS token.
    - "cls_mean": cat(CLS token, mean-pooled frame tokens).
    - "multi_token": num_query_tokens learnable tokens, concatenated.
    - "sparse_pool": sparsemax attention pooling over the frame (region)
      tokens after the blocks (exact zeros for ignored frames); the query
      token still takes part in the blocks but isn't read.
    frame_weights (used for attention viz and the entropy/distance-scale
    loss terms) is the query token(s)' attention to the frames, averaged
    over query tokens and summed over each frame's region tokens.

    head_dim: LayerNorm + Linear(dim -> head_dim) + dropout on the frame tokens first; blocks, embeddings
    and MLP head then run at head_dim (narrow head, fewer params). pool_temp: sparse_pool scores are
    multiplied by a learnable temperature (initialized at pool_temp) so the zero-init scores can leave
    near-uniform pooling (without it the trained pooling stays ~mean pooling)."""

    def _build_head(self, dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes,
                    aggregation_mode, num_query_tokens, use_pos_embedding=False, final_norm=False,
                    tokens_per_frame=1, head_dim=None, pool_temp=None):
        if aggregation_mode not in AGGREGATION_MODES:
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")
        if pool_temp and aggregation_mode != "sparse_pool":
            raise ValueError("pool_temp only applies to aggregation_mode='sparse_pool'")

        self.in_proj = None
        if head_dim:
            self.in_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, head_dim), nn.Dropout(dropout))
            dim = head_dim

        self.L = dim
        self.num_blocks = num_blocks
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.aggregation_mode = aggregation_mode
        self.use_pos_embedding = use_pos_embedding
        # Only "multi_token" uses more than one; kept as "cls_token" for
        # checkpoint compatibility with earlier single-CLS-token runs.
        self.num_query_tokens = num_query_tokens if aggregation_mode == "multi_token" else 1

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_query_tokens, dim))
        # Optional learned per-frame-slot embedding (not applied to the query token(s)).
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, dim)) if use_pos_embedding else None
        # Learned per-region embedding, shared across frames, when frames give K > 1 tokens.
        self.region_embedding = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, dim)) if tokens_per_frame > 1 else None

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if final_norm else None

        # sparse_pool: per-token score, zero-initialized so pooling starts uniform.
        self.pool_score = nn.Linear(dim, 1) if aggregation_mode == "sparse_pool" else None
        if self.pool_score is not None:
            nn.init.zeros_(self.pool_score.weight)
            nn.init.zeros_(self.pool_score.bias)
        # log temperature, learnable; None = plain scores (older runs)
        self.pool_log_temp = nn.Parameter(torch.tensor(float(pool_temp)).log()) if pool_temp else None

        classifier_in_dim = {"cls": dim, "cls_mean": 2 * dim, "multi_token": self.num_query_tokens * dim,
                             "sparse_pool": dim}
        self.classifier = make_mlp_head(classifier_in_dim[aggregation_mode], num_classes)

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
        if self.pos_embedding is not None:
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.region_embedding is not None:
            nn.init.trunc_normal_(self.region_embedding, std=0.02)

    def _load_full_checkpoint(self, path):
        """Strict: the model must have the same architecture config as when the checkpoint was saved."""
        print(f"Loading full {type(self).__name__} checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")
        # Feature standardization buffers of older ConvNeXt runs (removed 2026-10-20).
        # Identity buffers (runs without standardization) drop silently; fitted ones change the outputs.
        feat_mean, feat_std = state_dict.pop("feat_mean", None), state_dict.pop("feat_std", None)
        if feat_mean is not None and (feat_mean.any() or (feat_std != 1).any()):
            print(f"WARNING: {path} was trained on standardized backbone features, which are no longer "
                  f"applied, so its outputs will differ.")
        self.load_state_dict(state_dict, strict=True)
        print(f"Full model weights loaded strictly ({len(state_dict)} tensors).")

    @staticmethod
    def config_from_state_dict(state_dict):
        """Architecture kwargs recoverable from a checkpoint's keys/shapes, for
        old runs saved without architecture_config.json. num_heads isn't
        recoverable and keeps the default (8, which all such runs used)."""
        blocks = {int(m.group(1)) for k in state_dict if (m := re.match(r"blocks\.(\d+)\.", k))}
        num_query_tokens, dim = state_dict["cls_token"].shape[1:]
        in_dim = state_dict["classifier.0.weight"].shape[1]
        if num_query_tokens > 1:
            aggregation_mode = "multi_token"
        else:
            aggregation_mode = "cls_mean" if in_dim == 2 * dim else "cls"
        config = dict(
            num_blocks=max(blocks) + 1,
            ffn_dim=state_dict["blocks.0.ffn.0.weight"].shape[0],
            num_classes=state_dict["classifier.5.weight"].shape[0],
            aggregation_mode=aggregation_mode,
            num_query_tokens=num_query_tokens,
        )
        if "pos_embedding" in state_dict:
            config.update(use_pos_embedding=True, num_frames=state_dict["pos_embedding"].shape[1])
        return config

    @classmethod
    def from_checkpoint(cls, path, checkpoint_name="best_sa_model.pth",
                        config_name="architecture_config.json"):
        """Rebuild a model from a run dir (or a checkpoint file inside one),
        using the architecture_config.json sidecar saved at training time, or
        config_from_state_dict for older runs without one."""
        run_dir, ckpt = (path, os.path.join(path, checkpoint_name)) if os.path.isdir(path) \
            else (os.path.dirname(path), path)
        config_path = os.path.join(run_dir, config_name)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            # Removed options (2026-10-23); runs that had them enabled can't be rebuilt.
            for key in ("frame_prior", "alternating_attention"):
                if config.pop(key, False):
                    raise ValueError(f"{run_dir} was trained with the removed option {key}.")
            # grid adapter / grid_pool removed 2026-10-26.
            config.pop("grid_pool", None)
            if "grid" in (config.get("adapter"), config.get("dense_adapter")):
                raise ValueError(f"{run_dir} was trained with the removed grid adapter.")
            # cls_residual adapter (+ its adapter_attn_temp) removed 2026-11-16.
            config.pop("adapter_attn_temp", None)
            if config.get("dense_adapter") == "cls_residual":
                raise ValueError(f"{run_dir} was trained with the removed cls_residual adapter.")
        else:
            config = cls.config_from_state_dict(torch.load(ckpt, map_location="cpu"))
        return cls(full_checkpoint_path=ckpt, **config)

    def forward_head(self, h):
        # h: [num_frames, L] or [num_frames, K, L] -> pseudo-batch [1, num_frames(*K), L]
        if self.in_proj is not None:
            h = self.in_proj(h)
        h = h.unsqueeze(0)
        T = h.size(1)

        if self.pos_embedding is not None:
            if T != self.num_frames:
                raise ValueError(
                    f"Got {T} frames but positional embedding was initialized "
                    f"for num_frames={self.num_frames}."
                )
            h = h + (self.pos_embedding.unsqueeze(2) if h.dim() == 4 else self.pos_embedding)
        if h.dim() == 4:
            if self.region_embedding is not None:
                h = h + self.region_embedding
            h = h.flatten(1, 2)  # [1, num_frames*K, L], frame-major

        N = self.num_query_tokens
        query_tokens = self.cls_token.expand(h.size(0), -1, -1)  # [1, N, L]
        x = torch.cat((query_tokens, h), dim=1)  # [1, N + num_frames, L]

        # Attention weights kept from the last block only (reflects what
        # actually drove the final classification).
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x)
        if self.final_norm is not None:
            x = self.final_norm(x)

        query_out = x[:, :N, :]   # [1, N, L]
        frame_out = x[:, N:, :]   # [1, num_frames, L]

        pool_weights = None
        if self.aggregation_mode == "sparse_pool":
            scores = self.pool_score(frame_out).squeeze(-1).float()  # [1, num_frames*K]
            if self.pool_log_temp is not None:
                scores = scores * self.pool_log_temp.float().exp()
            pool_weights = sparsemax(scores, dim=-1)  # [1, num_frames*K], sums to 1
            bag_representation = (pool_weights.unsqueeze(-1).to(frame_out.dtype) * frame_out).sum(dim=1)  # [1, L]
        elif self.aggregation_mode == "cls_mean":
            bag_representation = torch.cat([query_out[:, 0, :], frame_out.mean(dim=1)], dim=-1)  # [1, 2L]
        elif self.aggregation_mode == "multi_token":
            bag_representation = query_out.reshape(query_out.size(0), -1)  # [1, N*L]
        else:  # "cls"
            bag_representation = query_out[:, 0, :]  # [1, L]

        logits = self.classifier(bag_representation)  # [1, num_classes]

        # Query token(s)' attention to the frame (region) tokens, averaged
        # over query tokens, normalized to sum to 1, then summed per frame.
        # sparse_pool: the pooling weights themselves.
        token_weights = pool_weights[0] if pool_weights is not None else attn_weights[0, :N, N:].mean(dim=0)  # [num_frames*K]
        region_weights = (token_weights / (token_weights.sum() + 1e-12)).view(T, -1)  # [num_frames, K]
        frame_weights = region_weights.sum(dim=1)  # [num_frames]

        return logits, frame_weights, region_weights

    def forward(self, x):
        """x: [num_frames, 3, H, W] -> (logits, frame_weights)"""
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)
        return logits, weights
