"""Train the fundus-cropping segmentation model (Unet++ / EfficientNet-B4) for Paxos2025.

One run per fold: fold i trains on the rows fold==i & split=="train" of the split CSV
(cropping/split_cropping_5fold_paxos2025_from_q5v3.csv) and validates on fold==i &
split=="val", so the val frames of fold i were never seen during training.
Saves the best-val-IoU model to <out_dir>/seg_fold_<fold>.pth and a per-epoch log to
<out_dir>/seg_fold_<fold>_log.json.

Run from the repo root:
    python -u -m cropping.train_segmentation --fold 0 --epochs 40
All 5 folds: sbatch scripts/train_cropping.sh
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from cropping.segmentation_pipeline import FundusDataset, get_transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def save_train_batch_plots(train_loader, out_dir, n_batches=10, max_images=8):
    """Plot the first n_batches training batches exactly as the model sees them: after
    the augmentations and normalization (denormalized for display), with the fundus
    mask contour overlaid. One PNG per batch: <out_dir>/batch_<nn>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for b, (imgs, masks) in enumerate(train_loader):
        if b >= n_batches:
            break
        imgs = imgs[:max_images]
        masks = masks[:max_images]
        imgs = imgs * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        imgs = imgs.permute(0, 2, 3, 1).clamp(0, 1).numpy()
        masks = masks[:, 0].numpy()
        fig, axes = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3.2))
        if len(imgs) == 1:
            axes = [axes]
        for ax, img, mask in zip(axes, imgs, masks):
            ax.imshow(img)
            ax.contour(mask, levels=[0.5], colors="red", linewidths=1.2)
            ax.axis("off")
        fig.suptitle(f"Training batch {b} of fold 0 (as seen by the model, mask in red)")
        fig.tight_layout()
        fig.savefig(out_dir / f"batch_{b:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1
    print(f"Saved {saved} training-batch plots to {out_dir}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fold(ann_csv, splits_csv, fold):
    """Return (train_df, val_df) for the given fold of the Paxos2025 cropping split CSV."""
    df = pd.read_csv(ann_csv)
    sp = pd.read_csv(splits_csv)
    sp = sp[sp["fold"] == fold]
    train_files = sp[sp["split"] == "train"]["filename"]
    val_files = sp[sp["split"] == "val"]["filename"]
    return df[df["filename"].isin(train_files)], df[df["filename"].isin(val_files)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, default=0, help="Fold index 0..4 (its val part is held out).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision (autocast + GradScaler); big speedup on A100 tensor cores.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--out_dir", type=str, default="cropping/models")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="'imagenet' or 'none' for random init.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_fold(args.ann_csv, args.splits_csv, args.fold)
    print(f"fold={args.fold}: {len(train_df)} train / {len(val_df)} val frames")

    train_ds = FundusDataset(train_df, args.img_dir, get_transforms(train=True, img_size=args.img_size))
    val_ds = FundusDataset(val_df, args.img_dir, get_transforms(train=False, img_size=args.img_size))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin)

    if args.fold == 0:
        save_train_batch_plots(train_loader, Path(args.out_dir).parent / "train_vis",
                               n_batches=10, max_images=min(args.batch_size, 8))

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = smp.UnetPlusPlus(
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model_save_path = out_dir / f"seg_fold_{args.fold}.pth"
    log_path = out_dir / f"seg_fold_{args.fold}_log.json"

    best_iou = -1.0  # guarantees a checkpoint is saved even if val IoU stays 0
    best_epoch = -1
    log = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = []

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                output = model(imgs)
                loss = loss_fn(output, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(loss.item())

        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                output = model(imgs)
                pred = (torch.sigmoid(output) > 0.5).long()
                s = smp.metrics.get_stats(pred, masks.long(), mode="binary")
                tp += s[0].sum()
                fp += s[1].sum()
                fn += s[2].sum()
                tn += s[3].sum()

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        epoch_log = {"epoch": epoch, "train_loss": float(np.mean(train_loss)), "val_iou": iou}
        log.append(epoch_log)
        print(f"Fold {args.fold} | Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {epoch_log['train_loss']:.4f} | Val IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"   -> New Best IoU! Saved {model_save_path}")

    with open(log_path, "w") as f:
        json.dump({"fold": args.fold, "epochs": args.epochs,
                   "epochs_run": len(log),
                   "best_epoch": best_epoch, "best_val_iou": best_iou, "history": log}, f, indent=2)
    print(f"Finished fold={args.fold} after {len(log)}/{args.epochs} epochs. "
          f"Best Val IoU: {best_iou:.4f} (epoch {best_epoch + 1})")


if __name__ == "__main__":
    main()
