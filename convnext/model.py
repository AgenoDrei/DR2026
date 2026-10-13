import json
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dino.train_dino_transformer import TransformerBlock


CONVNEXT_WEIGHTS = "dino/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"


def pos_enc_2d(H, W, D, device):
    """2D sine/cosine encoding (DETR-style). D must be divisible by 4."""
    d = D // 4
    freq = 1.0 / (10000 ** (torch.arange(d, device=device) / d))       # d
    y = torch.arange(H, device=device)[:, None] * freq                  # H×d
    x = torch.arange(W, device=device)[:, None] * freq                  # W×d
    y = torch.cat([y.sin(), y.cos()], dim=1)[:, None, :].expand(H, W, 2 * d)
    x = torch.cat([x.sin(), x.cos()], dim=1)[None, :, :].expand(H, W, 2 * d)
    return torch.cat([y, x], dim=-1).reshape(H * W, D)                  # (H·W)×D


class LinearMeanAdapter(nn.Module):
    """1x1 conv projection -> LayerNorm -> spatial mean. No positional
    encoding: a constant PE added before mean pooling would only shift every
    frame vector by the same offset."""

    def __init__(self, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f):
        t = self.proj(f).flatten(2).transpose(1, 2)  # B×(H·W)×D
        return self.norm(t).mean(dim=1)              # B×D


class LinearAttnPoolAdapter(nn.Module):
    """1x1 conv projection -> LayerNorm -> + 2D sin/cos PE -> attention
    pooling with a single learnable query over the H·W spatial tokens."""

    def __init__(self, in_ch, dim, num_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, f):
        f = self.proj(f)                             # B×D×H×W
        B, D, H, W = f.shape
        t = f.flatten(2).transpose(1, 2)             # B×(H·W)×D
        t = self.norm(t) + pos_enc_2d(H, W, D, t.device).to(t.dtype)
        pooled, _ = self.attn(self.query.expand(B, -1, -1), t, t, need_weights=False)
        return self.out_norm(pooled.squeeze(1))      # B×D


ADAPTERS = {
    "linear_mean": LinearMeanAdapter,
    "linear_attnpool": LinearAttnPoolAdapter,
}


def build_adapter(name, in_ch, dim):
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{name}', choose from {list(ADAPTERS)}")
    return ADAPTERS[name](in_ch, dim)


class ConvNextSelfAttention(nn.Module):
    """DINOv3 ConvNeXt-L backbone -> adapter (feature map -> one vector per
    frame) -> the same frame-level transformer head as
    dino.train_dino_transformer.DinoSelfAttention. No temporal encoding.

    backbone_weights=None builds the backbone with random init (smoke tests);
    full_checkpoint_path loads a full ConvNextSelfAttention state dict
    strictly (architecture must match)."""

    def __init__(
            self,
            freeze_backbone=False,
            grad_checkpointing=True,
            adapter="linear_attnpool",
            adapter_dim=1024,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1,
            num_blocks=1,
            num_frames=32,
            num_classes=1,
            aggregation_mode="cls",
            num_query_tokens=1,
            backbone_weights=CONVNEXT_WEIGHTS,
            full_checkpoint_path=None,
    ):
        super().__init__()

        if aggregation_mode not in ("cls", "cls_mean", "multi_token"):
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

        self.num_blocks = num_blocks
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.adapter_name = adapter
        self.aggregation_mode = aggregation_mode
        self.num_query_tokens = num_query_tokens if aggregation_mode == "multi_token" else 1
        self.grad_checkpointing = grad_checkpointing and not freeze_backbone

        if backbone_weights is not None and full_checkpoint_path is None:
            self.backbone = torch.hub.load("dino/dinov3", "dinov3_convnext_large",
                                           source="local", weights=backbone_weights)
        else:
            # Random init; full_checkpoint_path overwrites it below anyway.
            self.backbone = torch.hub.load("dino/dinov3", "dinov3_convnext_large",
                                           source="local", pretrained=False)

        # Per-channel standardization of the backbone feature map. Generic
        # DINOv3 ConvNeXt features share one dominant direction across all
        # videos (cos sim ~0.94), which left the randomly initialised head
        # stuck predicting a single grade; subtracting the train-set mean and
        # dividing by the std (as a linear probe does) fixes that. Identity
        # until fit_feature_stats() is called; saved with the state dict.
        C = self.backbone.embed_dim
        self.register_buffer("feat_mean", torch.zeros(C))
        self.register_buffer("feat_std", torch.ones(C))

        self.adapter = build_adapter(adapter, C, adapter_dim)
        self.L = adapter_dim

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_query_tokens, self.L))

        self.blocks = nn.ModuleList([
            TransformerBlock(self.L, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        # Final norm of the pre-norm stack (as in ViT), before the classifier reads the tokens.
        self.final_norm = nn.LayerNorm(self.L)

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

        if full_checkpoint_path is not None:
            print(f"Loading full ConvNextSelfAttention checkpoint: {full_checkpoint_path}")
            state_dict = torch.load(full_checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @classmethod
    def from_checkpoint(cls, run_dir, checkpoint_name="best_sa_model.pth",
                        config_name="architecture_config.json"):
        """Rebuild a model from a run_dir written by train_convnext_transformer."""
        with open(os.path.join(run_dir, config_name), "r") as f:
            config = json.load(f)
        return cls(full_checkpoint_path=os.path.join(run_dir, checkpoint_name), **config)

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

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

        nn.init.normal_(self.cls_token, std=1e-6)

        print(
            f"--- ConvNeXt Self-Attention [adapter={self.adapter_name}, {self.L} features, "
            f"{self.num_blocks} block(s), aggregation={self.aggregation_mode}] weights initialized ---"
        )

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS
    # ---------------------------------------------------------
    def backbone_feature_map(self, x):
        """[N, 3, H, W] -> [N, C, H/32, W/32] final-stage feature map (with the
        pretrained final norm), checkpointing each ConvNeXt block when
        fine-tuning so a 32-frame bag fits in 24GB."""
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for downsample, stage in zip(self.backbone.downsample_layers, self.backbone.stages):
            x = downsample(x)
            for block in stage:
                x = checkpoint(block, x, use_reentrant=False) if use_ckpt else block(x)
        x = self.backbone.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (x - self.feat_mean[:, None, None]) / self.feat_std[:, None, None]

    @torch.no_grad()
    def fit_feature_stats(self, bags, device):
        """Set feat_mean/feat_std to the per-channel mean/std of the (pretrained,
        unstandardized) backbone feature map over all frames and spatial
        positions in `bags` (an iterable of [num_frames, 3, H, W] tensors)."""
        was_training = self.training
        self.eval()
        self.feat_mean.zero_()
        self.feat_std.fill_(1.0)
        s = s2 = None
        n = 0
        for bag in bags:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                f = self.backbone_feature_map(bag.to(device))
            f = f.float().permute(1, 0, 2, 3).flatten(1)  # [C, frames*H*W]
            s = f.sum(1) if s is None else s + f.sum(1)
            s2 = (f ** 2).sum(1) if s2 is None else s2 + (f ** 2).sum(1)
            n += f.shape[1]
        mean = s / n
        self.feat_mean.copy_(mean)
        self.feat_std.copy_((s2 / n - mean ** 2).clamp_min(1e-6).sqrt())
        self.train(was_training)

    def extract_features(self, x):
        # [N, 3, H, W] -> [N, L]: one vector per frame for the transformer head.
        return self.adapter(self.backbone_feature_map(x))

    def forward_head(self, h):
        # h: [num_frames, L] -> pseudo-batch [1, num_frames, L]
        h = h.unsqueeze(0)
        N = self.num_query_tokens

        query_tokens = self.cls_token.expand(h.size(0), -1, -1)  # [1, N, L]
        x = torch.cat((query_tokens, h), dim=1)                   # [1, N + num_frames, L]

        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x)
        x = self.final_norm(x)

        query_out = x[:, :N, :]
        frame_out = x[:, N:, :]

        if self.aggregation_mode == "cls_mean":
            bag_representation = torch.cat([query_out[:, 0, :], frame_out.mean(dim=1)], dim=-1)
        elif self.aggregation_mode == "multi_token":
            bag_representation = query_out.reshape(query_out.size(0), -1)
        else:  # "cls"
            bag_representation = query_out[:, 0, :]

        logits = self.classifier(bag_representation)

        frame_weights = attn_weights[0, :N, N:].mean(dim=0)  # [num_frames]
        frame_weights = frame_weights / (frame_weights.sum() + 1e-12)

        return logits, frame_weights, frame_weights

    def forward(self, x):
        """x: [num_frames, 3, H, W]"""
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)
        return logits, weights
