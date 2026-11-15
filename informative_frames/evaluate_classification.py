"""Evaluate the Paxos2025 informative-frame classifier experiment.

Mirrors cropping/evaluate_segmentation.py. Every trained classifier
(<models_dir>/cls_fold_<i>.pth) is evaluated on the fold-i *val* rows of
<split_dir>/split_info_5fold_paxos2025_from_q5v3.csv, using the frames cropped by
informative_frames/crop_frames.py (<img_base>/frames_cropped/<label>/<filename>).
Together the 5 val folds cover every labeled frame exactly once, so the per-image
metrics are a clean out-of-sample estimate over the whole dataset.

Metrics per fold: micro accuracy / F1 / precision / recall at threshold 0.5 over all
evaluated images, plus ROC-AUC and PR-AUC over the probabilities.

Outputs in --out_dir:
  per_image_metrics.csv  one row per (fold, image): label, prob, pred
  summary.csv            one row per fold
  bar_summary.csv        one row per metric: mean over the 5 folds
  classification_eval.png  3 subplots (ROC-AUC, PR-AUC, F1), ONE bar each
                         (mean over the 5 folds, points = the 5 fold values)

Run from the repo root:
    python -u -m informative_frames.evaluate_classification
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from informative_frames.classify_frames import FrameDataset, cropped_dir, split_csv, get_val_transforms

METRICS = ["roc_auc", "pr_auc", "f1"]
METRIC_TITLES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 @ 0.5"}


def build_loader(fold, img_base, split_dir, transform, batch_size, num_workers):
    """The fold-i val rows of the split CSV, as a loader over the cropped frames."""
    df = pd.read_csv(split_csv(split_dir))
    df = df[(df["fold"] == fold) & (df["split"] == "val")]
    folder = cropped_dir(img_base)
    samples = []
    for row in df.itertuples():
        path = folder / str(int(row.label)) / row.filename
        if path.exists():
            samples.append((path, int(row.label)))
    filenames = [p.name for p, _ in samples]
    ds = FrameDataset(samples, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, filenames


def load_model(path, device):
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def predict(model, loader, device):
    """Sigmoid prob per image, aligned with loader order."""
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs.extend(torch.sigmoid(model(imgs)).view(-1).cpu().numpy().tolist())
    return probs


def group_metrics(y_true, probs, eps=1e-7):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = probs > 0.5
    tp = int(((pred) & (y_true == 1)).sum())
    fp = int(((pred) & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum())
    tn = int(((~pred) & (y_true == 0)).sum())
    row = {
        "n_images": len(y_true), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": (tp + tn) / (tp + fp + fn + tn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan,
    }
    p_curve, r_curve, _ = precision_recall_curve(y_true, probs)
    row["pr_auc"] = auc(r_curve, p_curve) if len(np.unique(y_true)) > 1 else np.nan
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="informative_frames/models")
    parser.add_argument("--img_base", type=str, default="data/informative_frames")
    parser.add_argument("--split_dir", type=str, default="informative_frames")
    parser.add_argument("--out_dir", type=str, default="informative_frames/eval")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transforms(args.img_size)

    per_image_rows, summary_rows = [], []

    for fold in range(args.n_splits):
        model_path = Path(args.models_dir) / f"cls_fold_{fold}.pth"
        if not model_path.exists():
            print(f"WARNING: missing model {model_path} -- fold {fold} skipped")
            continue
        model = load_model(model_path, device)
        loader, filenames = build_loader(fold, args.img_base, args.split_dir,
                                         transform, args.batch_size, args.num_workers)
        probs = predict(model, loader, device)
        y_true = [lbl for _, lbl in loader.dataset.samples]
        assert len(filenames) == len(probs) == len(y_true)
        m = group_metrics(y_true, probs)
        m.update({"fold": fold})
        summary_rows.append(m)
        for fn, y, p in zip(filenames, y_true, probs):
            per_image_rows.append({"fold": fold, "filename": fn, "label": y,
                                   "prob": p, "pred": int(p > 0.5)})
        print(f"fold={fold} (val fold): " + " ".join(f"{k}={m[k]:.4f}" for k in METRICS))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not summary_rows:
        raise RuntimeError(f"No models found in {args.models_dir} -- train with informative_frames/classify_frames.py first.")

    pd.DataFrame(per_image_rows).to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"Wrote {out_dir / 'per_image_metrics.csv'} ({len(per_image_rows)} rows)")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir / 'summary.csv'} ({len(summary)} rows)")

    bar_rows = []
    for m in METRICS:
        v = summary[m].to_numpy(dtype=float)
        bar_rows.append({"metric": m, "mean": float(np.nanmean(v)),
                         "std": float(np.nanstd(v)), "n_points": len(v)})
    bar_summary = pd.DataFrame(bar_rows)
    bar_summary.to_csv(out_dir / "bar_summary.csv", index=False)
    print(f"Wrote {out_dir / 'bar_summary.csv'}")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(7 * len(METRICS), 7), sharey=False)
    color = "#DD8452"
    for ax, m in zip(axes, METRICS):
        vals = summary[m].to_numpy(dtype=float)
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
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
        ax.set_ylabel(METRIC_TITLES[m])
    fig.suptitle("Informative-frame selection (classification) evaluation on Paxos2025 -- one bar = mean over 5 folds, "
                 "points = fold-i model on fold-i val (never trained on it)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = out_dir / "classification_eval.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    print("\nBar summary:")
    print(bar_summary.to_string(index=False))


if __name__ == "__main__":
    main()
