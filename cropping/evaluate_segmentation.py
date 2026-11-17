"""Evaluate the Paxos2025 cropping (segmentation) experiment.

Every trained model (<models_dir>/seg_fold_<i>.pth) is evaluated on the fold-i *val*
rows of cropping/split_cropping_5fold_paxos2025_from_q5v3.csv -- the part it never saw
during training. Together the
5 folds cover every annotated frame exactly once, so the per-image metrics are a
clean out-of-sample estimate over the whole dataset.

Metrics per image (binary fundus mask, at --img_size): IoU, Dice, pixel accuracy;
micro over all evaluated images per fold, plus per-image means (macro).

Outputs in --out_dir:
  per_image_metrics.csv  one row per (fold, image)
  summary.csv            one row per fold: micro + macro
  bar_summary.csv        one row per metric: mean over the 5 folds
  segmentation_eval.png  3 subplots (IoU, Dice, pixel acc), ONE bar each
                         (mean over the 5 folds, points = the 5 fold values)

Run from the repo root:
    python -u -m cropping.evaluate_segmentation
"""

import argparse
import json
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

METRICS = ["iou", "dice", "acc"]
METRIC_TITLES = {"iou": "IoU (Jaccard)", "dice": "Dice", "acc": "Pixel accuracy"}


def load_val_df(ann_csv, splits_csv, fold):
    df = pd.read_csv(ann_csv)
    sp = pd.read_csv(splits_csv)
    val_files = sp[(sp["fold"] == fold) & (sp["split"] == "val")]["filename"]
    return df[df["filename"].isin(val_files)]


def evaluate_model(model, loader, device):
    """Per-image (tp, fp, fn, tn) rows aligned with loader order (single model)."""
    model.eval()
    per_image = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = (torch.sigmoid(model(imgs)) > 0.5).long()
            s = smp.metrics.get_stats(pred, masks.long(), mode="binary")
            for i in range(imgs.size(0)):
                per_image.append((s[0][i].item(), s[1][i].item(), s[2][i].item(), s[3][i].item()))
    return per_image


def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "acc": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def summarize(filenames, per_image, fold):
    """per_image: list of (tp, fp, fn, tn) per image, aligned with filenames."""
    tp = sum(p[0] for p in per_image)
    fp = sum(p[1] for p in per_image)
    fn = sum(p[2] for p in per_image)
    tn = sum(p[3] for p in per_image)
    micro = metrics_from_counts(tp, fp, fn, tn)
    per_img_metrics = [metrics_from_counts(*p) for p in per_image]
    macro = {m: float(np.mean([pm[m] for pm in per_img_metrics])) for m in METRICS}
    row = {"fold": fold, "n_images": len(per_image), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           **{f"micro_{m}": v for m, v in micro.items()},
           **{f"macro_{m}": v for m, v in macro.items()}}
    img_rows = [{"fold": fold, "filename": fn_name, "tp": p[0], "fp": p[1], "fn": p[2], "tn": p[3],
                 **metrics_from_counts(*p)}
                for fn_name, p in zip(filenames, per_image)]
    return row, img_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="cropping/models")
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--out_dir", type=str, default="cropping/eval")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows, summary_rows = [], []

    for fold in range(args.n_splits):
        model_path = Path(args.models_dir) / f"seg_fold_{fold}.pth"
        if not model_path.exists():
            print(f"WARNING: missing model {model_path} -- fold {fold} skipped")
            continue
        df = load_val_df(args.ann_csv, args.splits_csv, fold)
        ds = FundusDataset(df, args.img_dir, get_transforms(train=False, img_size=args.img_size))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"fold {fold} val: {len(df)} images")

        model = smp.UnetPlusPlus(encoder_name=args.encoder_name, encoder_weights=None,
                                 in_channels=3, classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        per_image = evaluate_model(model, loader, device)
        filenames = loader.dataset.df["filename"].tolist()
        assert len(filenames) == len(per_image)
        row, img_rows = summarize(filenames, per_image, fold)
        summary_rows.append(row)
        per_image_rows.extend(img_rows)
        print(f"fold={fold} (val fold): " + " ".join(f"{m}={row[f'micro_{m}']:.4f}" for m in METRICS))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not summary_rows:
        raise RuntimeError(f"No models found in {args.models_dir} -- train with cropping/train_segmentation.py first.")

    pd.DataFrame(per_image_rows).to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"Wrote {out_dir / 'per_image_metrics.csv'} ({len(per_image_rows)} rows)")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir / 'summary.csv'} ({len(summary)} rows)")

    # Bar summary: one row per metric (mean over the 5 folds)
    bar_rows = []
    for m in METRICS:
        v = summary[f"micro_{m}"].to_numpy()
        bar_rows.append({"metric": m, "mean": float(np.mean(v)),
                         "std": float(np.std(v)), "n_points": len(v)})
    bar_summary = pd.DataFrame(bar_rows)
    bar_summary.to_csv(out_dir / "bar_summary.csv", index=False)
    print(f"Wrote {out_dir / 'bar_summary.csv'}")

    # Plot: 3 subplots (one per metric), ONE bar each; points = the 5 fold values
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7 * len(METRICS), 7), sharey=False)
    color = "#DD8452"
    for ax, m in zip(axes, METRICS):
        vals = summary[f"micro_{m}"].to_numpy()
        mean, std = float(np.mean(vals)), float(np.std(vals))
        ax.bar(0, mean, yerr=std, capsize=6, color=color, alpha=0.85, edgecolor="black")
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(jitter, vals, color="black", zorder=5, s=30)
        for xi, v in zip(jitter, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=9)
        ax.annotate(f"{mean:.3f}", (0, mean), textcoords="offset points", xytext=(28, 0),
                    ha="left", fontsize=11, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels(["Paxos2025"], fontsize=11)
        ax.set_title(METRIC_TITLES[m])
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("micro " + METRIC_TITLES[m])
    fig.suptitle("Cropping (segmentation) evaluation on Paxos2025 -- one bar = mean over 5 folds, "
                 "points = fold-i model on fold-i val (never trained on it)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = out_dir / "segmentation_eval.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    print("\nBar summary (micro):")
    print(bar_summary.to_string(index=False))


if __name__ == "__main__":
    main()
