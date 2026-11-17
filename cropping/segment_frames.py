import cv2
import torch
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from informative_frames.crop_frames import get_largest_component_mask, pad_to_square_info, crop_and_pad

IMG_DIR = Path('data/2024_Paxos_Frames/matched_frames')
CROPPED_DIR = Path('data/2024_Paxos_Frames/cropped_matched_frames')
SEG_MODEL_DIR = Path('cropping/models')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 768

CROPPED_DIR.mkdir(parents=True, exist_ok=True)


def get_transforms():
    return albu.Compose([
        albu.Resize(IMG_SIZE, IMG_SIZE),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def load_segmentation_ensemble():
    print(f"Loading 5-Fold Unet++ (B4) ensemble on {DEVICE}...")
    seg_ensemble = []
    for fold in range(5):
        sm = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1).to(DEVICE).eval()
        model_paths = list(SEG_MODEL_DIR.glob(f"*fold_{fold}*"))
        if not model_paths:
            raise FileNotFoundError(f"Could not find model for fold {fold} in {SEG_MODEL_DIR}")
        sm.load_state_dict(torch.load(model_paths[0], map_location=DEVICE))
        seg_ensemble.append(sm)
    return seg_ensemble


def save_cropped_images(seg_ensemble):
    all_files = sorted([f for f in IMG_DIR.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    print(f"Generating cropped frames for {len(all_files)} images...")

    transforms = get_transforms()

    with torch.no_grad():
        for i, img_path in enumerate(all_files):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue

            h, w = img_bgr.shape[:2]
            sq, y_off, x_off, side = pad_to_square_info(img_bgr)
            input_tensor = transforms(image=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0).to(DEVICE)

            mean_mask = torch.stack([torch.sigmoid(m(input_tensor)) for m in seg_ensemble]).mean(0).squeeze().cpu().numpy()
            full_mask = cv2.resize(mean_mask, (side, side))[y_off:y_off + h, x_off:x_off + w]

            clean_mask, box = get_largest_component_mask(full_mask > 0.5)
            if clean_mask is None:
                print(f"Warning: No valid retina component found for {img_path.name}. Skipping...")
                continue

            x1, y1, x2, y2 = box
            masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
            cropped_img = crop_and_pad(masked_img, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1))

            cv2.imwrite(str(CROPPED_DIR / img_path.name), cropped_img)

            if (i + 1) % 50 == 0:
                print(f"Cropped {i + 1}/{len(all_files)} images...")

    print(f"\nSuccessfully saved cropped images to: {CROPPED_DIR}")


if __name__ == "__main__":
    ensemble = load_segmentation_ensemble()
    save_cropped_images(ensemble)