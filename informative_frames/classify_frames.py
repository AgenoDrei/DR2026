"""Train the informative-frame selection classifier for Paxos2025.

One run per fold: fold i trains on the rows with fold==i & split=="train" of the split
CSV built by informative_frames/make_splits_from_cropping.py
(split_info_5fold_paxos2025_from_q5v3.csv, built by cropping/make_splits_from_q5v3.py) and
validates on fold==i & split=="val", so the val frames of fold i were never seen during
training. The training images are the frames cropped by informative_frames/crop_frames.py
(<img_base>/frames_cropped/<label>/<filename>, each cropped by the fold model of its own
val fold).

The best-val-PR-AUC model is saved to <models_dir>/cls_fold_<fold>.pth (consumed by
informative_frames/evaluate_classification.py), confusion matrices to
<models_dir>/Selection_B4_fold<fold>_<timestamp>_cm_epoch_<e>.png, a per-epoch
log to <models_dir>/cls_fold_<fold>_log.json, and TensorBoard logs to runs/.

Run one fold from the repo root:
    python -u -m informative_frames.classify_frames --fold 0
All 5 folds: sbatch scripts/train_classify.sh
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix,
                             precision_score, recall_score)

from misc.utils import apply_clahe_cv2


class FrameDataset(Dataset):
    """samples: list of (image_path, label); loads with PIL like datasets.ImageFolder."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path, y = self.samples[index]
        x = Image.open(path).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.samples)


def split_csv(split_dir):
    return Path(split_dir) / "split_info_5fold_paxos2025_from_q5v3.csv"


def cropped_dir(img_base):
    return Path(img_base) / "frames_cropped"


def build_samples(fold, split, img_base, split_dir):
    """(path, label) pairs for one fold+split."""
    df = pd.read_csv(split_csv(split_dir))
    df = df[(df["fold"] == fold) & (df["split"] == split)]
    folder = cropped_dir(img_base)
    samples = []
    n_missing = 0
    for row in df.itertuples():
        path = folder / str(int(row.label)) / row.filename
        if path.exists():
            samples.append((path, int(row.label)))
        else:
            n_missing += 1
    print(f"  fold={fold} {split}: {len(df)} rows in {split_csv(split_dir).name}, "
          f"{len(df) - n_missing} found in {folder}"
          + (f", {n_missing} missing" if n_missing else ""))
    return samples


def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def save_train_batch_plots(train_loader, out_dir, n_batches=10, max_images=8):
    """Plot the first n_batches training batches exactly as the model sees them: after
    the augmentations, CLAHE and normalization (denormalized for display), with the
    frame label as title. One PNG per batch: <out_dir>/batch_<nn>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for b, (imgs, labels) in enumerate(train_loader):
        if b >= n_batches:
            break
        imgs = imgs[:max_images]
        labels = labels[:max_images]
        imgs = imgs * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).numpy()
        fig, axes = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3.4))
        if len(imgs) == 1:
            axes = [axes]
        for ax, img, label in zip(axes, imgs, labels):
            ax.imshow(img)
            ax.set_title(f"label {int(label)}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Training batch {b} of fold 0 (as seen by the model)")
        fig.tight_layout()
        fig.savefig(out_dir / f"batch_{b:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1
    print(f"Saved {saved} training-batch plots to {out_dir}")


def save_confusion_matrix(y_true, y_pred, epoch, run_name, models_dir):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted (1=Informative, 0=Non-Informative)')
    plt.ylabel('Actual (1=Informative, 0=Non-Informative)')
    plt.title(f'CM - {run_name} - Epoch {epoch}')
    plt.savefig(os.path.join(models_dir, f"{run_name}_cm_epoch_{epoch:03d}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, help="Fold index 0..4 (its val part is held out).")
    parser.add_argument("--img_base", type=str, default="data/informative_frames",
                        help="Base of the cropped frame folder frames_cropped.")
    parser.add_argument("--split_dir", type=str, default="informative_frames",
                        help="Dir with split_info_5fold_paxos2025_from_q5v3.csv.")
    parser.add_argument("--models_dir", type=str, default="informative_frames/models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"Selection_B4_fold{args.fold}_{timestamp}"
    os.makedirs(args.models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"--- {run_name} on {device} ---")

    train_samples = build_samples(args.fold, "train", args.img_base, args.split_dir)
    val_samples = build_samples(args.fold, "val", args.img_base, args.split_dir)
    assert train_samples and val_samples, "empty train or val split -- check --img_base/--split_dir"

    train_labels = np.array([y for _, y in train_samples])
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(FrameDataset(train_samples, get_train_transforms(args.img_size)),
                              batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(FrameDataset(val_samples, get_val_transforms(args.img_size)),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.fold == 0:
        save_train_batch_plots(train_loader, Path("informative_frames") / "train_vis",
                               n_batches=10, max_images=min(args.batch_size, 8))

    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_pr_auc = 0.0
    best_epoch = -1
    log = []
    model_save_path = os.path.join(args.models_dir, f"cls_fold_{args.fold}.pth")
    log_path = os.path.join(args.models_dir, f"cls_fold_{args.fold}_log.json")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"F{args.fold} Ep {epoch} Train"):
            imgs, lbls = imgs.to(device), lbls.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        t_loss = running_loss / len(train_loader)

        model.eval()
        y_true, y_probs, v_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device).float().view(-1, 1)
                outputs = model(imgs)
                v_loss += criterion(outputs, lbls).item()
                y_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                y_true.extend(lbls.cpu().numpy())
        v_loss /= len(val_loader)
        y_true, y_probs = np.array(y_true), np.array(y_probs)
        y_pred = (y_probs > 0.5).astype(float)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(r_curve, p_curve)

        scheduler.step()
        print(f"Fold {args.fold} Ep {epoch} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Loss: {v_loss:.4f}")
        log.append({"epoch": epoch, "train_loss": float(t_loss), "val_loss": float(v_loss),
                    "acc": float(acc), "f1": float(f1), "precision": float(prec), "recall": float(rec),
                    "roc_auc": float(roc_auc), "pr_auc": float(pr_auc)})

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/F1', f1, epoch)
        writer.add_scalar('Metrics/Precision', prec, epoch)
        writer.add_scalar('Metrics/Recall', rec, epoch)
        writer.add_scalar('Metrics/ROC_AUC', roc_auc, epoch)
        writer.add_scalar('Metrics/PR_AUC', pr_auc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            save_confusion_matrix(y_true, y_pred, epoch, run_name, args.models_dir)

    writer.close()
    with open(log_path, "w") as f:
        json.dump({"fold": args.fold, "epochs": args.epochs,
                   "epochs_run": len(log),
                   "best_epoch": best_epoch, "best_val_pr_auc": best_pr_auc, "history": log}, f, indent=2)
    print(f"Finished fold={args.fold} after {len(log)}/{args.epochs} epochs. "
          f"Best val PR-AUC: {best_pr_auc:.4f} (epoch {best_epoch + 1}) -> {model_save_path}")


if __name__ == "__main__":
    main()
