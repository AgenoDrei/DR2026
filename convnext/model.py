import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dino.utils import freeze_module
from transformer.adapters import LinearAttnPoolAdapter
from transformer.model import BagTransformer, REMOVED_HEAD_OPTIONS, ANY


CONVNEXT_WEIGHTS = {
    "large": "dino/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    "base": "dino/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",  # only used by adafocus/
}
FEATURE_STAGE = 3  # features after the last block of stage 3 (768 ch at H/16); stage 4 and the final norm are unused


class ConvNextSelfAttention(BagTransformer):
    """DINOv3 ConvNeXt-L backbone (stage-3 feature map, fine-tuned) -> linear_attnpool adapter (one token per frame)
    -> the same BagTransformer head as DinoSelfAttention, plus a final LayerNorm. No temporal encoding.

    pretrained=False builds the backbone with random init (smoke tests); full_checkpoint_path loads a full
    ConvNextSelfAttention state dict strictly (prefer from_checkpoint(run_dir))."""

    # Options of older runs' architecture_config.json removed 2026-11-17 (now fixed to these values).
    removed_config = {**REMOVED_HEAD_OPTIONS, "backbone_arch": "large", "adapter": "linear_attnpool",
                      "adapter_tokens": None, "adapter_pe": True, "feature_stage": FEATURE_STAGE, "feature_block": -1}

    def __init__(
            self,
            adapter_dim=1024,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1,
            num_blocks=1,
            num_frames=32,
            num_classes=5,
            pretrained=True,
            full_checkpoint_path=None,
    ):
        super().__init__()

        if pretrained and full_checkpoint_path is None:
            self.backbone = torch.hub.load("dino/dinov3", "dinov3_convnext_large", source="local",
                                           weights=CONVNEXT_WEIGHTS["large"])
        else:
            # Random init; full_checkpoint_path overwrites it below anyway.
            self.backbone = torch.hub.load("dino/dinov3", "dinov3_convnext_large", source="local", pretrained=False)

        # Stage 4, its downsampling layer and the final norm don't contribute to the stage-3 features.
        for module in (self.backbone.stages[3], self.backbone.downsample_layers[3], self.backbone.norm):
            freeze_module(module)

        C = self.backbone.embed_dims[FEATURE_STAGE - 1]
        self.adapter = LinearAttnPoolAdapter(C, adapter_dim)
        self._build_head(adapter_dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, num_classes, final_norm=True)
        print(f"--- ConvNeXt-L Self-Attention [stage {FEATURE_STAGE} ({C} ch) -> linear_attnpool, {self.L} features, "
              f"{self.num_blocks} block(s), sparse_pool] weights initialized ---")

        if full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

    def backbone_feature_map(self, x):
        """[N, 3, H, W] -> [N, 768, H/16, W/16], checkpointing each ConvNeXt block in training so a 32-frame bag fits in 24GB."""
        use_ckpt = self.training and torch.is_grad_enabled()
        for downsample, stage in zip(self.backbone.downsample_layers[:FEATURE_STAGE], self.backbone.stages[:FEATURE_STAGE]):
            x = downsample(x)
            for block in stage:
                x = checkpoint(block, x, use_reentrant=False) if use_ckpt else block(x)
        return x

    def extract_features(self, x):
        # [N, 3, H, W] -> [N, L]: one vector per frame for the transformer head.
        return self.adapter(self.backbone_feature_map(x))
