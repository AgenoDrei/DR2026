import json
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
import matplotlib

matplotlib.use('Agg')

IMG_DIR = Path('cropping/frames_to_annotate_larger')
CSV_PATH = Path('cropping/annotations_video_frames_larger.csv')
OUTPUT_DIR = Path('cropping/models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = 768
N_SPLITS = 5
SPLITS_JSON_PATH = 'cropping/splits.json'


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


def get_transforms(train=True):
    if train:
        return albu.Compose([
            # 1. Minor Crop & Rescale
            # Takes a random 80% to 100% chunk of the original image, keeps it square,
            # and resizes it to IMG_SIZE.
            albu.RandomResizedCrop(
                size=(IMG_SIZE, IMG_SIZE),
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
            albu.Resize(height=IMG_SIZE, width=IMG_SIZE),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


def run_training():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Total images found in CSV: {len(df)}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits_dict = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        splits_dict[f"fold_{fold}"] = {
            "train": df.iloc[train_idx]['filename'].tolist(),
            "val": df.iloc[val_idx]['filename'].tolist()
        }

    with open(SPLITS_JSON_PATH, 'w') as f:
        json.dump(splits_dict, f, indent=4)
    print(f"Saved cross-validation splits to {SPLITS_JSON_PATH}")

    for fold in range(N_SPLITS):
        print(f"\n{'=' * 30}")
        print(f"Starting Training for FOLD {fold + 1}/{N_SPLITS}")
        print(f"{'=' * 30}")

        # Extract data for current fold
        train_filenames = splits_dict[f"fold_{fold}"]["train"]
        val_filenames = splits_dict[f"fold_{fold}"]["val"]

        train_df = df[df['filename'].isin(train_filenames)]
        valid_df = df[df['filename'].isin(val_filenames)]

        train_ds = FundusDataset(train_df, IMG_DIR, get_transforms(train=True))
        valid_ds = FundusDataset(valid_df, IMG_DIR, get_transforms(train=False))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # Initialize Modernized Model (U-Net++ with EfficientNet-b4 encoder)
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        best_iou = 0.0
        model_save_path = f'cropping/models/best_model_fold_{fold}.pth'

        for epoch in range(EPOCHS):
            model.train()
            train_loss = []

            for imgs, masks in train_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                optimizer.zero_grad()
                output = model(imgs)
                loss = loss_fn(output, masks)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            # Validation
            model.eval()
            tp, fp, fn, tn = 0, 0, 0, 0

            with torch.no_grad():
                for imgs, masks in valid_loader:
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    output = model(imgs)
                    pred = (torch.sigmoid(output) > 0.5).long()

                    s = smp.metrics.get_stats(pred, masks.long(), mode='binary')
                    tp += s[0].sum()
                    fp += s[1].sum()
                    fn += s[2].sum()
                    tn += s[3].sum()

            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()

            print(
                f"Fold {fold} | Epoch {epoch + 1}/{EPOCHS} | Train Loss: {np.mean(train_loss):.4f} | Val IoU: {iou:.4f}")

            if iou > best_iou:
                best_iou = iou
                torch.save(model.state_dict(), model_save_path)
                print(f"   -> New Best IoU! Saved {model_save_path}")

        print(f"Finished Fold {fold}. Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    run_training()
