import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dino.utils import freeze_module
from transformer.model import BagTransformer


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


class ConvNextSelfAttention(BagTransformer):
    """DINOv3 ConvNeXt-L backbone -> adapter (feature map -> one vector per
    frame) -> the same BagTransformer head as DinoSelfAttention, plus a final
    LayerNorm. No temporal encoding.

    backbone_weights=None builds the backbone with random init (smoke tests);
    full_checkpoint_path loads a full ConvNextSelfAttention state dict
    strictly (architecture must match; prefer from_checkpoint(run_dir))."""

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

        self.adapter_name = adapter
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
        # Final norm of the pre-norm stack (as in ViT), before the classifier reads the tokens.
        self._build_head(adapter_dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes,
                         aggregation_mode, num_query_tokens, final_norm=True)
        print(
            f"--- ConvNeXt Self-Attention [adapter={self.adapter_name}, {self.L} features, "
            f"{self.num_blocks} block(s), aggregation={self.aggregation_mode}] weights initialized ---"
        )

        if full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

        if freeze_backbone:
            freeze_module(self.backbone)

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
