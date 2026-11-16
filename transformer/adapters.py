import torch
import torch.nn as nn


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

    tokens_per_frame = 1

    def __init__(self, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f):
        t = self.proj(f).flatten(2).transpose(1, 2)  # B×(H·W)×D
        return self.norm(t).mean(dim=1)              # B×D


class LinearMaxAdapter(LinearMeanAdapter):
    """Like linear_mean but spatial max pooling (keeps the strongest local response per channel, e.g. a single lesion).
    mask (B×(H·W) bool, True = ignore; the DINO path passes retina_patch_mask) keeps black border patches out of the max."""

    uses_mask = True

    def forward(self, f, mask=None):
        t = self.norm(self.proj(f).flatten(2).transpose(1, 2))  # B×(H·W)×D
        if mask is not None:
            t = t.masked_fill(mask.unsqueeze(-1), float("-inf"))
        return t.amax(dim=1)                                    # B×D


class LinearAttnPoolAdapter(nn.Module):
    """1x1 conv projection -> LayerNorm -> + 2D sin/cos PE -> attention
    pooling with num_queries learnable queries over the H·W spatial tokens.
    Returns B×D for one query, B×K×D (K region tokens per frame) otherwise.
    use_pe=False drops the PE (purely content-based pooling)."""

    def __init__(self, in_ch, dim, num_heads=4, num_queries=1, use_pe=True):
        super().__init__()
        self.tokens_per_frame = num_queries
        self.use_pe = use_pe
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, f):
        f = self.proj(f)                             # B×D×H×W
        B, D, H, W = f.shape
        t = f.flatten(2).transpose(1, 2)             # B×(H·W)×D
        t = self.norm(t)
        if self.use_pe:
            t = t + pos_enc_2d(H, W, D, t.device).to(t.dtype)
        pooled, _ = self.attn(self.query.expand(B, -1, -1), t, t, need_weights=False)
        pooled = self.out_norm(pooled)               # B×K×D
        return pooled.squeeze(1) if self.tokens_per_frame == 1 else pooled


def retina_patch_mask(x, patch=16, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      dark_thr=0.06, max_dark=0.5):
    """[N, 3, H, W] normalized frames -> [N, (H/patch)·(W/patch)] bool, True = patch to ignore
    (more than max_dark of its pixels have every channel < dark_thr, i.e. black border/fill).
    Frames that would lose every patch keep all of them."""
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    dark = ((x * s + m).amax(dim=1, keepdim=True) < dark_thr).float()
    masked = (nn.functional.avg_pool2d(dark, patch) > max_dark).flatten(1)
    return masked & ~masked.all(dim=1, keepdim=True)


ADAPTERS = {
    "linear_mean": LinearMeanAdapter,
    "linear_max": LinearMaxAdapter,
    "linear_attnpool": LinearAttnPoolAdapter,
}


def build_adapter(name, in_ch, dim, dense_tokens=None, adapter_pe=True, num_heads=4):
    """dense_tokens: number of queries for linear_attnpool (None = 1);
    linear_mean / linear_max always give one token.
    adapter_pe: 2D sin/cos PE in linear_attnpool (linear_mean / linear_max have none)."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{name}', choose from {list(ADAPTERS)}")
    if not adapter_pe and name != "linear_attnpool":
        raise ValueError(f"adapter_pe=False only applies to linear_attnpool, not '{name}'")
    kw = {} if adapter_pe or name != "linear_attnpool" else {"use_pe": False}
    if dense_tokens is not None and name not in ("linear_mean", "linear_max"):
        kw["num_queries"] = dense_tokens
    if num_heads != 4 and name == "linear_attnpool":
        kw["num_heads"] = num_heads
    return ADAPTERS[name](in_ch, dim, **kw)
