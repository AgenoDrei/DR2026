import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    auc, cohen_kappa_score, confusion_matrix
)


def binary_metrics(y_true, y_pred, y_probs):
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5,
        'pr_auc': auc(rec_curve, prec_curve),
    }


def multiclass_metrics(y_true, y_pred, y_probs, num_classes):
    labels = list(range(num_classes))
    y_true_onehot = np.eye(num_classes)[y_true]

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    y_true_referable = (y_true >= 2).astype(int)
    y_pred_referable = (y_pred >= 2).astype(int)
    referable_recall = recall_score(y_true_referable, y_pred_referable, zero_division=0)

    present = np.unique(y_true)
    roc_auc_macro = (
        roc_auc_score(y_true, y_probs, labels=labels, multi_class='ovr', average='macro')
        if len(present) > 1 else 0.5
    )
    roc_auc_weighted = (
        roc_auc_score(y_true, y_probs, labels=labels, multi_class='ovr', average='weighted')
        if len(present) > 1 else 0.5
    )

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

    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    ap_per_class = average_precision_score(y_true_onehot, y_probs, average=None)

    for c in labels:
        metrics[f'f1_class{c}'] = f1_per_class[c]
        metrics[f'precision_class{c}'] = prec_per_class[c]
        metrics[f'recall_class{c}'] = rec_per_class[c]
        metrics[f'pr_auc_class{c}'] = ap_per_class[c]
        if len(present) > 1 and c in present:
            metrics[f'roc_auc_class{c}'] = roc_auc_score((y_true == c).astype(int), y_probs[:, c])

    metrics['pr_auc'] = metrics['pr_auc_macro']
    return metrics


def save_confusion_matrix(y_true, y_pred, epoch, run_dir, class_names=None):
    num_classes = len(class_names) if class_names is not None else (int(max(y_true.max(), y_pred.max())) + 1)
    labels = list(range(num_classes))
    if class_names is None:
        class_names = [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)  # row-normalized, avoid div/0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Counts - Epoch {epoch}')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Row-normalized (Recall) - Epoch {epoch}')

    plt.tight_layout()
    save_path = os.path.join(run_dir, f"cm_epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close(fig)
    return save_path, cm


def apply_clahe_cv2(img_pil):
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))