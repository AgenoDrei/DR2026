"""Dataset and transforms shared by the Paxos2025 cropping stage.

FundusDataset builds the fundus mask from the ellipse annotation columns
(center_x, center_y, radius_x, radius_y) of cropping/annotations_cropping_2025.csv.
"""

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 768


class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not find {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate Mask from CSV coordinates on the original image size
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        center = (int(row['center_x']), int(row['center_y']))
        axes = (int(row['radius_x']), int(row['radius_y']))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # [1, H, W]

        return image, mask


def get_transforms(train=True, img_size=IMG_SIZE):
    if train:
        return albu.Compose([
            # 1. Minor Crop & Rescale
            # Takes a random 80% to 100% chunk of the original image, keeps it square,
            # and resizes it to img_size.
            albu.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(1.0, 1.0),
                p=1.0
            ),

            # 2. Flips and Full Rotations
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Affine(
                rotate=(-180, 180),  # Full 360-degree rotation freedom
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # Minor shifts
                scale=(0.95, 1.05),  # Minor zoom in/out
                p=0.7
            ),

            # 3. Simple Blurring & Lighting
            albu.GaussianBlur(blur_limit=(3, 5), p=0.3),
            albu.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),

            # 4. Normalization
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        # Validation: Only Resize and Normalize
        return albu.Compose([
            albu.Resize(height=img_size, width=img_size),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
