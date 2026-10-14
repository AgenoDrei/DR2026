import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import v2


DINO_REPO = "dino/dinov3"
DINO_WEIGHTS = "dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def load_dino_backbone():
    """Pretrained DINOv3 ViT-L/16 from the local hub clone."""
    return torch.hub.load(DINO_REPO, "dinov3_vitl16", source="local", weights=DINO_WEIGHTS)


def load_backbone_from_classifier(backbone, path):
    """Strictly load the backbone.* weights of a DinoClassifier checkpoint into `backbone`."""
    print(f"Loading backbone from DinoClassifier checkpoint: {path}")
    state_dict = torch.load(path, map_location="cpu")
    backbone_state = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    if not backbone_state:
        raise ValueError(f"No keys starting with 'backbone.' found in {path}. Is this really a DinoClassifier checkpoint?")
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


def make_train_transform_dino(img_size, complex_augs):
    if complex_augs:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),

            v2.Lambda(apply_clahe_cv2_dino),

            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.1, contrast=0.1),

            v2.ToDtype(torch.float32, scale=True),

            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    else:
        return v2.Compose([v2.ToImage(), v2.Resize((img_size, img_size), antialias=True),
                           v2.ToDtype(torch.float32, scale=True),
                           v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
                           ])


def make_val_transform_dino(img_size, complex_augs):
    if complex_augs:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),

            v2.Lambda(apply_clahe_cv2_dino),

            v2.ToDtype(torch.float32, scale=True),

            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
        ])