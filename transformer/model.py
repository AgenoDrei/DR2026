import json
import os
import re

import torch
import torch.nn as nn

from dino.utils import make_mlp_head


AGGREGATION_MODES = ("cls", "cls_mean", "multi_token")


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
    ([num_frames, 3, H, W] -> [num_frames, dim]).

    aggregation_mode controls how the classifier head reads the
    post-attention sequence:
    - "cls" (default): only the final CLS token.
    - "cls_mean": cat(CLS token, mean-pooled frame tokens).
    - "multi_token": num_query_tokens learnable tokens, concatenated.
    frame_weights (used for attention viz and the entropy/distance-scale
    loss terms) is the query token(s)' attention to the frames, averaged
    over query tokens."""

    def _build_head(self, dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes,
                    aggregation_mode, num_query_tokens, use_pos_embedding=False, final_norm=False):
        if aggregation_mode not in AGGREGATION_MODES:
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

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

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if final_norm else None

        classifier_in_dim = {"cls": dim, "cls_mean": 2 * dim, "multi_token": self.num_query_tokens * dim}
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

    def _load_full_checkpoint(self, path):
        """Strict: the model must have the same architecture config as when the checkpoint was saved."""
        print(f"Loading full {type(self).__name__} checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")
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
        else:
            config = cls.config_from_state_dict(torch.load(ckpt, map_location="cpu"))
        return cls(full_checkpoint_path=ckpt, **config)

    def forward_head(self, h):
        # h: [num_frames, L] -> pseudo-batch [1, num_frames, L]
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
        if self.final_norm is not None:
            x = self.final_norm(x)

        query_out = x[:, :N, :]   # [1, N, L]
        frame_out = x[:, N:, :]   # [1, num_frames, L]

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
        """x: [num_frames, 3, H, W] -> (logits, frame_weights)"""
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)
        return logits, weights
