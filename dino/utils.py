import re

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import v2

from dino.domain_augs import Paxos2020Aug


DINO_REPO = "dino/dinov3"
DINO_WEIGHTS = "dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINO_WEIGHTS_SAT = "dino/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"  # same architecture, satellite pretraining
RETFOUND_WEIGHTS = "dino/RETFound_dinov2_meh.pth"  # gated HF repo YukunZhou/RETFound_dinov2_meh
BACKBONES = ("dinov3", "retfound_dinov2")


def load_dino_backbone(weights=DINO_WEIGHTS):
    """Pretrained DINOv3 ViT-L/16 from the local hub clone. Files without the hub's '-<hash8>.pth' suffix (e.g.
    dino/ssl_lora.py's merged_backbone.pth, LVD architecture) are loaded strictly into an untrained model."""
    if re.search(r"-.{8}\.pth$", weights):
        return torch.hub.load(DINO_REPO, "dinov3_vitl16", source="local", weights=weights)
    model = torch.hub.load(DINO_REPO, "dinov3_vitl16", source="local", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    print(f"DINOv3 ViT-L/16 weights loaded strictly from {weights}")
    return model


class RetFoundBackbone(nn.Module):
    """RETFound_dinov2_meh (timm DINOv2 ViT-L/14, fundus-pretrained at 224px) with the DINOv3 interface used by the
    aggregators: embed_dim, patch_size, forward_features -> normed CLS/patch tokens, grad_checkpointing flag.
    Inputs are resized to img_size inside forward_features, so the 512px data pipeline stays unchanged; other img_size
    than the pretraining 224 resamples the position embedding."""

    def __init__(self, weights=RETFOUND_WEIGHTS, img_size=224):
        super().__init__()
        import timm
        from timm.layers import resample_abs_pos_embed
        self.model = timm.create_model("vit_large_patch14_dinov2", pretrained=False, img_size=img_size, num_classes=0)
        self.img_size = img_size
        self.embed_dim, self.patch_size = self.model.embed_dim, self.model.patch_embed.patch_size[0]
        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=False)
            state = state.get("teacher", state.get("model", state))
            # DINO/iBOT heads and the iBOT mask token only exist for pretraining; keys may carry a "backbone." prefix.
            state = {k.removeprefix("backbone."): v for k, v in state.items()
                     if not k.startswith(("dino_head.", "ibot_head.")) and k.removeprefix("backbone.") != "mask_token"}
            grid = self.model.patch_embed.grid_size
            if state["pos_embed"].shape != self.model.pos_embed.shape:
                state["pos_embed"] = resample_abs_pos_embed(state["pos_embed"], grid,
                                                            num_prefix_tokens=self.model.num_prefix_tokens)
            self.model.load_state_dict(state, strict=True)
            print(f"RETFound weights loaded strictly ({len(state)} tensors) from {weights}")

    @property
    def grad_checkpointing(self):
        return self.model.grad_checkpointing

    @grad_checkpointing.setter
    def grad_checkpointing(self, enable):
        self.model.set_grad_checkpointing(enable)

    def forward_features(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", antialias=True)
        tokens = self.model.forward_features(x)  # [N, prefix + h*w, C], final norm applied
        p = self.model.num_prefix_tokens
        return {"x_norm_clstoken": tokens[:, 0], "x_norm_patchtokens": tokens[:, p:]}


def load_backbone(name="dinov3", img_size=224, weights=None):
    """Backbone by name (BACKBONES): pretrained DINOv3 ViT-L/16 (optionally other ViT-L weights, e.g. an SSL-adapted
    merged_backbone.pth) or RETFound_dinov2_meh at img_size (RETFound only)."""
    if name == "dinov3":
        return load_dino_backbone(weights or DINO_WEIGHTS)
    if name == "retfound_dinov2":
        return RetFoundBackbone(img_size=img_size)
    raise ValueError(f"Unknown backbone: {name}")


def load_backbone_from_classifier(backbone, path):
    """Strictly load the backbone.* weights of a DinoClassifier checkpoint into `backbone`."""
    print(f"Loading backbone from DinoClassifier checkpoint: {path}")
    state_dict = torch.load(path, map_location="cpu")
    backbone_state = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    if not backbone_state:
        raise ValueError(f"No keys starting with 'backbone.' found in {path}. Is this really a DinoClassifier checkpoint?")
    if getattr(backbone, "local_cls_norm", None) is None:
        # SAT-493M backbones carry a local-crop CLS norm that is only used in multi-crop training, never in our forward.
        backbone_state = {k: v for k, v in backbone_state.items() if not k.startswith("local_cls_norm.")}
    backbone.load_state_dict(backbone_state, strict=True)
    print(f"Backbone weights loaded strictly ({len(backbone_state)} tensors).")


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def make_mlp_head(in_dim, num_classes):
    """The MLP head shared by all DINO/ConvNeXt models (xavier-initialized).
    Layer indices 0/3/5 are part of every checkpoint's key names - don't reorder."""
    head = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    for layer in head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    return head


class DinoBackbone(nn.Module):
    """Frozen-by-default DINOv3 CLS-token feature extractor, optionally with a classifier's backbone weights."""

    def __init__(self, checkpoint_path=None, freeze=True):
        super().__init__()
        self.model = load_dino_backbone()
        self.embed_dim = self.model.embed_dim
        if checkpoint_path is not None:
            load_backbone_from_classifier(self.model, checkpoint_path)
        if freeze:
            freeze_module(self.model)

    def forward(self, x):
        feats = self.model.forward_features(x)
        return feats["x_norm_clstoken"]


def clahe_rgb(img):
    """CLAHE on the L channel of a uint8 RGB HxWx3 array; returns uint8 RGB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def apply_clahe_cv2_dino(img):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()

    elif isinstance(img, Image.Image):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    img = clahe_rgb(img)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


EXTRA_AUGS = ("color", "affine", "degrade", "p20_color", "p20_quality", "p20_light")


class RandomGamma(nn.Module):
    """Gamma correction with a random gamma from `gamma_range`, applied with probability p (float images in [0, 1])."""

    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        super().__init__()
        self.gamma_range, self.p = gamma_range, p

    def forward(self, img):
        if torch.rand(1).item() >= self.p:
            return img
        return v2.functional.adjust_gamma(img, torch.empty(1).uniform_(*self.gamma_range).item())


def _random_augs(extra_augs):
    """Geometric + photometric train augs (after CLAHE). extra_augs: 'affine' adds zoom/shift to the rotation,
    'color' widens the jitter (+ saturation/hue) and adds random gamma; contrast stays at 0.2, since stronger
    contrast jitter lifts the black crop border to visible gray."""
    rotation = (v2.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.9, 1.1)) if "affine" in extra_augs
                else v2.RandomRotation(180))
    if "color" in extra_augs:
        color = [v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.02), RandomGamma((0.7, 1.5), p=0.5)]
    else:
        color = [v2.ColorJitter(brightness=0.1, contrast=0.1)]
    return [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), rotation, *color]


def _degrade_augs(extra_augs):
    """'degrade': per-frame JPEG artefacts (uint8, before CLAHE), blur and noise (float), each with p 0.2."""
    if "degrade" not in extra_augs:
        return [], [], []
    return ([v2.RandomApply([v2.JPEG((40, 90))], p=0.2)],
            [v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 1.5))], p=0.2)],
            [v2.RandomApply([v2.GaussianNoise(sigma=0.03)], p=0.2)])


def _p20_augs(extra_augs):
    """'p20_color' / 'p20_quality' / 'p20_light': Paxos2020-style look (dino/domain_augs.py), on float images."""
    groups = [a[4:] for a in extra_augs if a.startswith("p20_")]
    return [Paxos2020Aug(groups)] if groups else []


def _train_resize(img_size, train_zoom):
    """Resize to img_size, or with train_zoom (lo, hi) a random square crop of 1/hi..1/lo of the side (area scale uniform in
    [1/hi^2, 1/lo^2]) at a random position, resized to img_size: zoom-in plus spatial shift."""
    if not train_zoom:
        return v2.Resize((img_size, img_size), antialias=True)
    lo, hi = train_zoom
    return v2.RandomResizedCrop(img_size, scale=(1 / hi ** 2, 1 / lo ** 2), ratio=(1.0, 1.0), antialias=True)


def _val_resize(img_size, zoom):
    """Resize to img_size, center-zoomed by `zoom` (> 1: resize to img_size * zoom, then center crop img_size)."""
    if zoom <= 1:
        return [v2.Resize((img_size, img_size), antialias=True)]
    side = round(img_size * zoom)
    return [v2.Resize((side, side), antialias=True), v2.CenterCrop(img_size)]


def make_train_transform_dino(img_size, complex_augs, extra_augs=(), clahe=True, train_zoom=None):
    """extra_augs (subset of EXTRA_AUGS) only applies with complex_augs; clahe=False drops CLAHE from complex_augs.
    train_zoom (lo, hi): random zoom-in + shift instead of the plain resize (_train_resize)."""
    if complex_augs:
        jpeg, blur, noise = _degrade_augs(extra_augs)
        return v2.Compose([
            v2.ToImage(),
            _train_resize(img_size, train_zoom),
            *jpeg,

            *([v2.Lambda(apply_clahe_cv2_dino)] if clahe else []),

            *_random_augs(extra_augs),
            *blur,

            v2.ToDtype(torch.float32, scale=True),
            *_p20_augs(extra_augs),
            *noise,

            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    else:
        return v2.Compose([v2.ToImage(), _train_resize(img_size, train_zoom),
                           v2.ToDtype(torch.float32, scale=True),
                           v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
                           ])


def make_bag_train_transforms_dino(img_size, complex_augs, extra_augs=(), clahe=True, train_zoom=None):
    """Train augs split for frame bags: (per-frame part, part applied once to the stacked [T, C, H, W] bag),
    so every frame of a bag gets the same flip/rotation/jitter/zoom; 'degrade' stays per frame."""
    frame = [v2.ToImage(), v2.Resize((img_size, img_size), antialias=True)]
    bag = [_train_resize(img_size, train_zoom)] if train_zoom else []
    noise = []
    if complex_augs:
        jpeg, blur, noise = _degrade_augs(extra_augs)
        frame += [*jpeg, *([v2.Lambda(apply_clahe_cv2_dino)] if clahe else []), *blur]
        bag += _random_augs(extra_augs) + _p20_augs(extra_augs)
    frame += [v2.ToDtype(torch.float32, scale=True), *noise]
    bag.append(v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return v2.Compose(frame), v2.Compose(bag)


def make_val_transform_dino(img_size, complex_augs, clahe=True, zoom=1.0):
    """zoom > 1: center zoom-in (_val_resize); the default 1.0 keeps the plain resize (SSL probe, clustering)."""
    if complex_augs:
        return v2.Compose([
            v2.ToImage(),
            *_val_resize(img_size, zoom),

            *([v2.Lambda(apply_clahe_cv2_dino)] if clahe else []),

            v2.ToDtype(torch.float32, scale=True),

            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            *_val_resize(img_size, zoom),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
        ])