import os
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn

from .train_dino_classifier import DinoClassifier
from dino.utils import make_val_transform_dino, clahe_rgb

IMG_SIZE = 512
TOKEN_SIZE = 16
OCCLUSION_SIZE = 48
BATCH_SIZE = 16


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class ReferableLogit(nn.Module):
    """Wraps a 5-class classifier to output the logit of P(grade >= 2), so the
    occlusion code below can treat binary and 5-class checkpoints alike."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        p = torch.softmax(self.model(x).float(), dim=1)[:, 2:].sum(dim=1, keepdim=True).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))


def load_model(weights, device):
    model = DinoClassifier.from_checkpoint(weights)
    if model.head[-1].out_features != 1:
        model = ReferableLogit(model)
    return model.eval().to(device)


@torch.no_grad()
def forward_logit(model, x):
    return model(x).squeeze().item()


# ------------------------------------------------------------
# Correct normalized "black"
# ------------------------------------------------------------
def get_black(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return (0.0 - mean) / std


# ------------------------------------------------------------
# Occlusion heatmap
# ------------------------------------------------------------
@torch.no_grad()
def compute_heatmap(model, x, device):
    x = x.to(device)

    original = forward_logit(model, x.unsqueeze(0))

    grid = IMG_SIZE // TOKEN_SIZE
    heatmap = np.zeros((grid, grid), dtype=np.float32)

    radius = OCCLUSION_SIZE // 2
    black = get_black(device)

    batch, coords = [], []

    for r in range(grid):
        for c in range(grid):

            cy = r * TOKEN_SIZE + TOKEN_SIZE // 2
            cx = c * TOKEN_SIZE + TOKEN_SIZE // 2

            y0, y1 = max(0, cy - radius), min(IMG_SIZE, cy + radius)
            x0, x1 = max(0, cx - radius), min(IMG_SIZE, cx + radius)

            masked = x.clone()

            masked[:, y0:y1, x0:x1] = black

            batch.append(masked)
            coords.append((r, c))

            if len(batch) == BATCH_SIZE:

                logits = model(torch.stack(batch)).squeeze(1).cpu().numpy()

                for i, logit in enumerate(logits):
                    rr, cc = coords[i]
                    heatmap[rr, cc] = original - float(logit)

                batch, coords = [], []

    if batch:
        logits = model(torch.stack(batch)).squeeze(1).cpu().numpy()

        for i, logit in enumerate(logits):
            rr, cc = coords[i]
            heatmap[rr, cc] = original - float(logit)

    return heatmap, original


# ------------------------------------------------------------
# Visualization (Continuous Alpha Blend)
# ------------------------------------------------------------
def make_overlay(image_rgb, heatmap, max_alpha=0.7):
    import matplotlib

    heatmap = heatmap.astype(np.float32)
    max_abs = np.max(np.abs(heatmap))

    if max_abs < 1e-8:
        max_abs = 1.0

    heatmap_norm = heatmap / max_abs

    # [-1,1] -> [0,1] for colormap sampling
    heatmap_vis = (heatmap_norm + 1.0) / 2.0

    cmap = matplotlib.colormaps["bwr"]
    heatmap_color = cmap(heatmap_vis)[..., :3]
    heatmap_color = (255 * heatmap_color).astype(np.float32)

    # Pure standalone heatmap output (full color BWR)
    heatmap_u8 = heatmap_color.astype(np.uint8)

    # Continuous alpha: Higher absolute value = higher opacity.
    # Neutral values (near 0) become completely transparent, preventing a white wash.
    alpha = np.abs(heatmap_norm) * max_alpha
    alpha = np.expand_dims(alpha, axis=-1)  # Broadcasts across RGB channels

    # Complete overlay blending equation without harsh masking cutoffs
    overlay = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return heatmap_u8, overlay


# ------------------------------------------------------------
# Process single image
# ------------------------------------------------------------
def process_image(model, image_path, transform, device, out_dir):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    x = transform(Image.fromarray(img))

    heatmap, logit = compute_heatmap(model, x, device)

    img_vis = clahe_rgb(img)

    heatmap_up = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), cv2.INTER_CUBIC)

    # max_alpha=0.7 makes the peak red/blue regions highly prominent
    heat_u8, overlay = make_overlay(img_vis, heatmap_up, max_alpha=0.7)

    # Convert all panels to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    heat_bgr = cv2.cvtColor(heat_u8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Merge the 3 images side-by-side into a single canvas
    combined_view = np.hstack([img_bgr, heat_bgr, overlay_bgr])

    # Save directly to output folder without subdirectories
    name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{name}_combined.png")

    cv2.imwrite(save_path, combined_view)

    prob = torch.sigmoid(torch.tensor(logit)).item()
    print(f"[{name}] logit={logit:.4f} prob={prob:.4f} -> Saved to {save_path}")


# ------------------------------------------------------------
# Main folder runner
# ------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.weights, device)

    transform = make_val_transform_dino(
        img_size=512,
        complex_augs=args.complex_augs
    )

    image_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".png")
    ]

    image_files.sort()

    for img_path in image_files:
        process_image(
            model,
            img_path,
            transform,
            device,
            args.output_dir
        )


# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_dir", default="occlusion_results")
    parser.add_argument("--complex_augs", action="store_true")

    args = parser.parse_args()

    main(args)