import os
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    precision_recall_curve, roc_curve, auc, confusion_matrix
)

BINARY_HEADLINE = ['F1', 'ROC_AUC_Direct', 'PR_AUC_Direct', 'Best_Thresh']
MULTI_HEADLINE = [
    'acc', 'qwk', 'referable_recall', 'f1_macro', 'f1_weighted',
    'precision_macro', 'recall_macro', 'roc_auc_macro', 'pr_auc_macro'
]


# ----------------------------------------------------------------------------
# Model detection
# ----------------------------------------------------------------------------
def detect_models(df):
    """Finds every model in a prediction CSV, based on the column naming of the eval script.

    Multi-class model NAME:  NAME_pred_grade_split_N  +  NAME_prob_0 ... NAME_prob_{K-1}
    Binary model NAME:       NAME_split_N             (one probability column)

    Returns a list of dicts with keys: name, kind ('binary' | 'multiclass'), and either
    score_col (binary) or pred_col / prob_cols (multiclass).
    """
    models = []
    claimed = set()

    for col in df.columns:
        m = re.match(r"^(?P<name>.+)_pred_grade_split_\d+$", col)
        if not m:
            continue
        name = m.group("name")
        prob_re = re.compile(rf"^{re.escape(name)}_prob_(\d+)$")
        prob_cols = sorted(
            (c for c in df.columns if prob_re.match(c)),
            key=lambda c: int(prob_re.match(c).group(1))
        )
        if not prob_cols:
            raise ValueError(f"Model '{name}' has '{col}' but no '{name}_prob_<c>' columns.")
        expected = [f"{name}_prob_{c}" for c in range(len(prob_cols))]
        if prob_cols != expected:
            raise ValueError(f"Model '{name}' has non-consecutive probability columns: {prob_cols}")

        models.append({"name": name, "kind": "multiclass", "pred_col": col, "prob_cols": prob_cols})
        claimed.update(prob_cols + [col])

    for col in df.columns:
        if col in claimed:
            continue
        m = re.match(r"^(?P<name>.+)_split_\d+$", col)
        if m:
            models.append({"name": m.group("name"), "kind": "binary", "score_col": col})

    return models


# ----------------------------------------------------------------------------
# Binary analysis
# ----------------------------------------------------------------------------
def find_best_threshold(y_true, y_probs):
    prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.divide(
        2 * (prec * rec), (prec + rec),
        out=np.zeros_like(prec), where=(prec + rec) != 0
    )
    best_idx = np.argmax(f1_scores)
    thresh_idx = min(best_idx, len(thresholds) - 1)
    return thresholds[thresh_idx], f1_scores[best_idx]


def get_video_frame_stats(video_name, txt_dir):
    """Parses the frame-level txt file of a video: (#informative frames, mean prob among them)."""
    txt_path = os.path.join(txt_dir, os.path.splitext(str(video_name))[0] + ".txt")
    if not os.path.exists(txt_path):
        return 0, 0.0

    try:
        txt_df = pd.read_csv(txt_path, sep=r"\s+")
        inf_df = txt_df[txt_df["informative"] == True]
        num_inf = len(inf_df)
        return num_inf, (inf_df["prob"].mean() if num_inf > 0 else 0.0)
    except Exception:
        return 0, 0.0


def analyse_binary(df, split_name, score_col, model_name, output_dir, txt_dir):
    y_true = df["label"].values.astype(int)
    y_probs = df[score_col].values.astype(float)

    roc_auc_direct = roc_auc_score(y_true, y_probs)
    pr_auc_direct = average_precision_score(y_true, y_probs)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    roc_auc_integral = auc(fpr, tpr)
    pr_auc_integral = auc(rec, prec)

    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

    sort_idx = np.argsort(-y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)
    is_correct = (y_true == y_pred)

    video_col = "video" if "video" in df.columns else df.columns[0]
    sorted_videos = df[video_col].values[sort_idx]
    sorted_correct = is_correct[sort_idx]

    grade_colors = {0: "green", 1: "yellow", 2: "orange", 3: "red", 4: "black"}
    grade_col = "grade" if "grade" in df.columns else "label"
    colors = [grade_colors.get(g, "gray") for g in df[grade_col].values[sort_idx]]

    show_inf = txt_dir is not None and os.path.isdir(txt_dir)
    n_panels = 6 if show_inf else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes = iter(axes)

    # Ranking
    ax = next(axes)
    ax.scatter(range(len(y_probs)), y_probs[sort_idx], c=colors, s=15, alpha=0.8,
               edgecolors="black", linewidths=0.5)
    ax.axhline(best_thresh, color="black", linestyle="--", label=f"Thresh: {best_thresh:.2f}")
    ax.set_title(f"Ranking\nF1: {best_f1:.3f}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score")
    ax.legend()

    # Informative-frame panels (only if the txt directory exists)
    if show_inf:
        inf_counts, mean_scores, bar_colors = [], [], []
        for vid, correct in zip(sorted_videos, sorted_correct):
            cnt, mean_s = get_video_frame_stats(vid, txt_dir)
            inf_counts.append(cnt)
            mean_scores.append(mean_s)
            bar_colors.append("#2ecc71" if correct else "#e74c3c")
        x_indices = np.arange(len(sorted_videos))

        ax = next(axes)
        ax.bar(x_indices, inf_counts, color=bar_colors, width=0.8, edgecolor="none")
        ax.set_title("Informative Frame Count\n(Green = Correct, Red = Wrong)")
        ax.set_xlabel("Video Index (Ranked)")
        ax.set_ylabel("Frame Count")

        ax = next(axes)
        ax.bar(x_indices, mean_scores, color=bar_colors, width=0.8, edgecolor="none")
        ax.set_title("Mean Informative Frame Score\n(Green = Correct, Red = Wrong)")
        ax.set_xlabel("Video Index (Ranked)")
        ax.set_ylabel("Mean Prob Score")
        ax.set_ylim(-0.05, 1.05)

    # Confusion matrix
    ax = next(axes)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("CM @ Opt. Thresh")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # ROC
    ax = next(axes)
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"Direct: {roc_auc_direct:.3f}\nIntegral: {roc_auc_integral:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize="small")

    # PR
    ax = next(axes)
    ax.plot(rec, prec, color="green", lw=2,
            label=f"Direct: {pr_auc_direct:.3f}\nIntegral: {pr_auc_integral:.3f}")
    ax.set_title("PR Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize="small")

    plt.suptitle(f"Split Analysis: {split_name} | Model: {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"full_diag_{split_name}_{model_name.replace(' ', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "Split": split_name,
        "Model": model_name,
        "Best_Thresh": best_thresh,
        "F1": best_f1,
        "ROC_AUC_Direct": roc_auc_direct,
        "ROC_AUC_Integral": roc_auc_integral,
        "PR_AUC_Direct": pr_auc_direct,
        "PR_AUC_Integral": pr_auc_integral,
    }
    wrong_videos = list(df[video_col].values[~is_correct])
    return metrics, wrong_videos


# ----------------------------------------------------------------------------
# Multi-class analysis
# ----------------------------------------------------------------------------
def multiclass_metrics(y_true, y_pred, y_probs, num_classes):
    labels = list(range(num_classes))
    y_true_onehot = np.eye(num_classes)[y_true]

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    referable_recall = recall_score(
        (y_true >= 2).astype(int), (y_pred >= 2).astype(int), zero_division=0
    )

    present = np.unique(y_true)

    if len(present) > 1:
        roc_auc_macro = roc_auc_score(y_true, y_probs, labels=labels, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_true, y_probs, labels=labels, multi_class='ovr', average='weighted')
    else:
        roc_auc_macro = 0.5
        roc_auc_weighted = 0.5

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'qwk': qwk,
        'referable_recall': referable_recall,
        'f1_macro': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'pr_auc_macro': average_precision_score(y_true_onehot, y_probs, average='macro'),
        'pr_auc_weighted': average_precision_score(y_true_onehot, y_probs, average='weighted'),
    }

    f1_pc = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec_pc = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_pc = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    ap_pc = average_precision_score(y_true_onehot, y_probs, average=None)

    for c in labels:
        metrics[f'f1_class{c}'] = f1_pc[c]
        metrics[f'precision_class{c}'] = prec_pc[c]
        metrics[f'recall_class{c}'] = rec_pc[c]
        metrics[f'pr_auc_class{c}'] = ap_pc[c]

        if len(present) > 1 and c in present:
            metrics[f'roc_auc_class{c}'] = roc_auc_score((y_true == c).astype(int), y_probs[:, c])
        else:
            metrics[f'roc_auc_class{c}'] = 0.5

    metrics['pr_auc'] = metrics['pr_auc_macro']
    return metrics


def save_confusion_matrix(y_true, y_pred, tag, run_dir, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Counts - {tag}')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Row-normalized (Recall) - {tag}')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"cm_{tag}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def analyse_multiclass(df, split_name, model_info, output_dir):
    y_true = df['grade'].values.astype(int)
    y_pred = df[model_info["pred_col"]].values.astype(int)
    y_probs = df[model_info["prob_cols"]].values.astype(float)
    num_classes = len(model_info["prob_cols"])
    class_names = [str(c) for c in range(num_classes)]

    metrics = multiclass_metrics(y_true, y_pred, y_probs, num_classes)

    model_name = model_info["name"]
    save_confusion_matrix(y_true, y_pred, f"{split_name}_{model_name.replace(' ', '_')}",
                          output_dir, class_names)

    row = {"Split": split_name, "Model": model_name}
    row.update(metrics)

    video_col = "video" if "video" in df.columns else df.columns[0]
    wrong_videos = list(df[video_col].values[y_true != y_pred])
    return row, wrong_videos


# ----------------------------------------------------------------------------
# Shared reporting
# ----------------------------------------------------------------------------
def print_cross_model_error_analysis(tracker, mode):
    """Lists misclassified videos across all models, per split."""
    print("\n" + "=" * 50)
    print(f"   CROSS-MODEL MISCLASSIFICATION ANALYSIS ({mode})")
    print("=" * 50)

    for split_name, model_errors in tracker.items():
        n_models = len(model_errors)
        print(f"\n--- {split_name.upper()} ---")

        video_error_counts = defaultdict(list)
        for model_name, wrong_vids in model_errors.items():
            for v in wrong_vids:
                video_error_counts[v].append(model_name)

        if not video_error_counts:
            print("No errors found across any models.")
            continue

        multi = {v: m for v, m in video_error_counts.items() if len(m) > 1}
        single = {v: m for v, m in video_error_counts.items() if len(m) == 1}

        print(f"Total Unique Wrong Videos: {len(video_error_counts)}")
        print(f"Videos Wrong by MULTIPLE Models ({len(multi)}):")
        if multi:
            for vid, models in sorted(multi.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"  • {str(vid):<15} -> Failed in {len(models)}/{n_models} models: {', '.join(models)}")
        else:
            print("  None! No video was misclassified by more than one model.")

        print(f"\nVideos Wrong by ONLY ONE Model ({len(single)}):")
        if single:
            for vid, models in sorted(single.items(), key=lambda x: str(x[0])):
                print(f"  • {str(vid):<15} -> Failed in: {models[0]}")
        else:
            print("  None!")


def plot_metric_summary(summary_df, metrics_to_plot, model_order, shape, figsize, suptitle, output_dir):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(*shape, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, (metric, title) in enumerate(metrics_to_plot):
        if metric not in summary_df.columns:
            continue
        ax = axes[idx]

        sns.barplot(
            data=summary_df, x="Model", y=metric, hue="Model",
            order=model_order, hue_order=model_order, ax=ax, palette="Set2",
            width=0.5, alpha=0.7, legend=False, errorbar="sd", capsize=0.15,
            err_kws={"linewidth": 1.5, "color": "#333333"}
        )
        sns.stripplot(
            data=summary_df, x="Model", y=metric, order=model_order, ax=ax,
            color="#111111", jitter=0.08, size=6, alpha=0.9, linewidth=0
        )

        vals = summary_df[metric]
        lo, hi = vals.min(), vals.max()
        val_range = hi - lo
        margin = val_range * 0.2 if val_range > 0 else max(abs(hi) * 0.1, 0.02)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")

    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_comparison_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved to: {save_path}")


def print_comparison_overview(summary_df, metrics, model_order, output_dir):
    """Prints and saves mean ± SD per model and metric."""
    cols = [c for c in metrics if c in summary_df.columns]
    grouped = summary_df.groupby('Model')[cols]
    means, stds = grouped.mean(), grouped.std()

    rows = []
    for metric in cols:
        row = {'Metric': metric}
        for model in model_order:
            row[model] = (f"{means.loc[model, metric]:.4f} ± {stds.loc[model, metric]:.4f}"
                          if model in means.index else "N/A")
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    print("\n" + "=" * 20 + " MODEL COMPARISON OVERVIEW " + "=" * 20)
    print(comp_df.to_string(index=False))

    out_file = os.path.join(output_dir, "model_comparison_summary.csv")
    comp_df.to_csv(out_file, index=False)
    print(f"Comparison table saved to: {out_file}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    csv_files = [f for f in csv_files if "summary" not in os.path.basename(f)]
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}.")
        return

    if args.txt_dir is not None and not os.path.isdir(args.txt_dir):
        print(f"Note: --txt_dir '{args.txt_dir}' not found; skipping the informative-frame panels.")

    # Results per mode: summary rows, misclassification tracker, model order, combined predictions
    results = {
        mode: {"rows": [], "tracker": defaultdict(dict), "models": [], "combined": {}}
        for mode in ("binary", "multiclass")
    }

    for f in csv_files:
        df = pd.read_csv(f)
        split_name = os.path.splitext(os.path.basename(f))[0]

        models = detect_models(df)
        if not models:
            print(f"Skipping {split_name}: no model columns found.")
            continue

        processed = []
        for info in models:
            mode, name = info["kind"], info["name"]
            res = results[mode]

            if mode == "binary":
                if "label" not in df.columns:
                    print(f"Skipping binary model '{name}' in {split_name}: no 'label' column.")
                    continue
                out_dir = os.path.join(args.output_dir, "binary", name.replace(' ', '_'))
                os.makedirs(out_dir, exist_ok=True)
                row, wrong = analyse_binary(df, split_name, info["score_col"], name, out_dir, args.txt_dir)
            else:
                if "grade" not in df.columns:
                    print(f"Skipping multi-class model '{name}' in {split_name}: no 'grade' column.")
                    continue
                out_dir = os.path.join(args.output_dir, "multiclass", "confusion_matrices")
                os.makedirs(out_dir, exist_ok=True)
                row, wrong = analyse_multiclass(df, split_name, info, out_dir)

                comb = res["combined"].setdefault(name, {"y_true": [], "y_pred": [],
                                                         "classes": len(info["prob_cols"])})
                comb["y_true"].extend(df['grade'].values.astype(int))
                comb["y_pred"].extend(df[info["pred_col"]].values.astype(int))

            res["rows"].append(row)
            res["tracker"][split_name][name] = wrong
            if name not in res["models"]:
                res["models"].append(name)
            processed.append(f"{name} ({mode})")

        if processed:
            print(f"Processed {split_name} for models: {', '.join(processed)}")

    if not any(r["rows"] for r in results.values()):
        print("No valid metrics were computed.")
        return

    # ---- Binary report ----
    res = results["binary"]
    if res["rows"]:
        out = os.path.join(args.output_dir, "binary")
        summary_df = pd.DataFrame(res["rows"]).sort_values(by=["Split", "Model"])
        summary_df.to_csv(os.path.join(out, "analysis_summary.csv"), index=False)

        plot_metric_summary(
            summary_df,
            [("F1", "Optimal F1 Score"), ("ROC_AUC_Direct", "ROC-AUC"),
             ("PR_AUC_Direct", "PR-AUC"), ("Best_Thresh", "Optimal Threshold")],
            res["models"], (2, 2), (12, 10),
            f"Binary Performance Across {summary_df['Split'].nunique()} Splits (Mean ± SD)", out
        )
        print_cross_model_error_analysis(res["tracker"], "binary")
        print_comparison_overview(summary_df, BINARY_HEADLINE, res["models"], out)

    # ---- Multi-class report ----
    res = results["multiclass"]
    if res["rows"]:
        out = os.path.join(args.output_dir, "multiclass")
        summary_df = pd.DataFrame(res["rows"]).sort_values(by=["Split", "Model"])
        summary_df.to_csv(os.path.join(out, "analysis_summary_full.csv"), index=False)

        for name, data in res["combined"].items():
            save_confusion_matrix(
                np.array(data["y_true"]), np.array(data["y_pred"]),
                f"all_splits_combined_{name.replace(' ', '_')}",
                os.path.join(out, "confusion_matrices"),
                [str(c) for c in range(data["classes"])]
            )

        plot_metric_summary(
            summary_df,
            [('acc', 'Accuracy'), ('qwk', 'Quadratic Weighted Kappa (QWK)'),
             ('referable_recall', 'Referable Recall (Grade >= 2)'), ('f1_macro', 'F1 Macro'),
             ('roc_auc_macro', 'ROC-AUC Macro'), ('pr_auc_macro', 'PR-AUC Macro')],
            res["models"], (2, 3), (16, 10),
            f"Model Performance Across {summary_df['Split'].nunique()} Splits (Mean ± SD)", out
        )
        print_cross_model_error_analysis(res["tracker"], "multiclass")
        print_comparison_overview(summary_df, MULTI_HEADLINE, res["models"], out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=".", help="Folder with the prediction CSVs.")
    parser.add_argument('--output_dir', type=str, default="split_analysis_results")
    parser.add_argument('--txt_dir', type=str, default="../ensemble_results_paxos2025/txt_files",
                        help="Frame-level txt files for the binary informativeness panels. "
                             "If the folder does not exist, those panels are skipped.")
    main(parser.parse_args())