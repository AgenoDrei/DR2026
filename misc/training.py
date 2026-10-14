"""Boilerplate shared by the training scripts (dino/train_dino_*.py, convnext/train_convnext_transformer.py)."""
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from misc.utils import save_confusion_matrix


def seed_everything(seed):
    """Best-effort reproducibility (weight init, sampler, dropout, frame sampling) -
    not bitwise-exact since cudnn's algorithm choice stays non-deterministic for speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(model_dir, description, run_tag=None):
    """-> (run_name, run_dir): run_name = '<YYYY-MM-DD_HH-MM>_<description>[_<run_tag>]', run_dir = <model_dir>/models/<run_name>."""
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{description}{f'_{run_tag}' if run_tag else ''}"
    run_dir = os.path.join(model_dir, "models", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_name, run_dir


def video_search_dirs(fused):
    """cleaned_videos dirs of Paxos2025 (+ Paxos2020 if fused)."""
    dirs = ["data/ensemble_results_paxos2025/cleaned_videos"]
    if fused:
        dirs.append("data/ensemble_results_paxos2020/cleaned_videos")
    return dirs


def make_balanced_loaders(train_ds, val_ds, binary_classification, batch_size=1, require_all_classes=False):
    """Train loader oversampling by inverse class frequency (binary labels or 0-4 grades), plain val loader."""
    targets = np.array(train_ds.labels if binary_classification else train_ds.grades).astype(int)
    class_counts = np.bincount(targets)
    if require_all_classes:
        expected = 2 if binary_classification else 5
        assert len(class_counts) == expected and class_counts.all(), \
            f"Expected {expected} classes, got counts {class_counts.tolist()} - some class missing from split"
    sample_weights = torch.from_numpy((1.0 / class_counts)[targets]).double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader


def make_optimizer(model, lr, backbone_lr_mult=None, optimizer="adam", weight_decay=0.01):
    """Adam/AdamW over the trainable params. With backbone_lr_mult, the backbone gets
    its own group at lr * backbone_lr_mult; param_groups[0] is always the head group."""
    if backbone_lr_mult is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    else:
        backbone_param_ids = {id(p) for p in model.backbone.parameters()}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [{"params": head_params, "lr": lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    if optimizer == "adamw":
        return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    return optim.Adam(param_groups, lr=lr)


class EpochLogger:
    """Per-epoch TensorBoard logging, confusion matrix, console line and best-checkpoint
    saving (selection metric: QWK for 5-class, PR-AUC for binary)."""

    def __init__(self, run_name, run_dir, args, binary_classification, checkpoint_name, cm_to_tensorboard=False):
        self.run_dir = run_dir
        self.binary = binary_classification
        self.checkpoint_path = os.path.join(run_dir, checkpoint_name)
        self.cm_to_tensorboard = cm_to_tensorboard
        self.class_names = ['non-referable', 'referable'] if binary_classification else ['0', '1', '2', '3', '4']
        self.writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
        self.writer.add_text("args", str(args))
        # -inf (not 0.0) so a checkpoint is always written, even if the metric stays <= 0.
        self.best_selection_metric = float("-inf")
        self.best_metrics = None

    def log_epoch(self, epoch, model, optimizer, train_loss, m, extra_scalars=None):
        """m: metrics dict from a validate_* loop (incl. 'loss', 'y_true', 'y_pred')."""
        w = self.writer
        cm_path, _ = save_confusion_matrix(m['y_true'], m['y_pred'], epoch, self.run_dir, class_names=self.class_names)
        if self.cm_to_tensorboard:
            w.add_image('ConfusionMatrix', np.array(Image.open(cm_path).convert('RGB')).transpose(2, 0, 1), epoch)

        selection_metric = m['pr_auc'] if self.binary else m['qwk']
        if selection_metric > self.best_selection_metric:
            self.best_selection_metric = selection_metric
            self.best_metrics = {k: v for k, v in m.items() if k not in ('y_true', 'y_pred')}
            torch.save(model.state_dict(), self.checkpoint_path)
            w.add_scalar('Meta/Best_Selection_Metric', selection_metric, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        w.add_scalar('Meta/Learning_Rate', current_lr, epoch)
        if len(optimizer.param_groups) > 1:
            w.add_scalar('Meta/Backbone_Learning_Rate', optimizer.param_groups[1]['lr'], epoch)
        w.add_scalar('Loss/train', train_loss, epoch)
        w.add_scalar('Loss/val', m['loss'], epoch)
        for tag, value in (extra_scalars or {}).items():
            w.add_scalar(tag, value, epoch)
        # Every scalar metric, generically - covers binary and multiclass keys alike.
        for key, value in m.items():
            if key not in ('loss', 'y_true', 'y_pred'):
                w.add_scalar(f'Metric/{key}', value, epoch)

        headline_name, headline_value = ('PR-AUC', m['pr_auc']) if self.binary else ('QWK', m['qwk'])
        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {train_loss:.3f} | "
            f"Val Loss: {m['loss']:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {m['f1'] if self.binary else m['f1_macro']:.3f} | "
            f"{headline_name}: {headline_value:.3f}"
        )

    def close(self):
        self.writer.close()
        print(f"Done. Run directory: {self.run_dir}")
