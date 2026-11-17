"""Boilerplate shared by the training scripts (dino/train_dino_*.py, convnext/train_convnext_transformer.py)."""
import csv
import fcntl
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dino.utils import TRAIN_ZOOM
from misc.utils import save_confusion_matrix


def seed_everything(seed):
    """Best-effort reproducibility (weight init, sampler, dropout, frame sampling) -
    not bitwise-exact since cudnn's algorithm choice stays non-deterministic for speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch's default of one thread per core (80) oversubscribes the CPU when several runs share the
    # node (~3.5x slower). OMP_NUM_THREADS overrides the cap.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 16)))
    cv2.setNumThreads(4)


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


def loader_kwargs(num_workers):
    """DataLoader worker settings shared by the training and eval loaders (decoding + augs overlap the GPU step).
    Workers run torch single-threaded by default; re-setting thread counts in a forked worker can deadlock."""
    if num_workers == 0:
        return {}
    return dict(num_workers=num_workers, persistent_workers=True)


def make_balanced_loaders(train_ds, val_ds, binary_classification, batch_size=1, require_all_classes=False,
                          num_workers=4, sampler_power=1.0):
    """Train loader oversampling by class frequency ** -sampler_power (binary labels or 0-4 grades; 1 = fully balanced,
    0.5 = sqrt-balanced, 0 = plain shuffle), plain val loader."""
    targets = np.array(train_ds.labels if binary_classification else train_ds.grades).astype(int)
    class_counts = np.bincount(targets)
    if require_all_classes:
        expected = 2 if binary_classification else 5
        assert len(class_counts) == expected and class_counts.all(), \
            f"Expected {expected} classes, got counts {class_counts.tolist()} - some class missing from split"
    class_weights = np.where(class_counts > 0, np.maximum(class_counts, 1) ** -float(sampler_power), 0.0)
    share = class_counts * class_weights / (class_counts * class_weights).sum()
    print(f"Sampler (power {sampler_power}): class counts {class_counts.tolist()} -> "
          f"expected share {np.round(share, 3).tolist()}")
    sample_weights = torch.from_numpy(class_weights[targets]).double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, **loader_kwargs(num_workers))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs(num_workers))
    return train_loader, val_loader


class WeightedCrossEntropy(nn.Module):
    """Per-sample class-weighted CE averaged over the batch. Unlike nn.CrossEntropyLoss(weight=...), which divides by the
    summed target weights (so the weights cancel for a batch of one video), the weights scale the loss itself."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        return (F.cross_entropy(logits, target, reduction="none") * self.weight[target]).mean()


def add_balancing_args(parser):
    """--sampler_power, --class_weight_power, --crop_borders (default on) and zoom (add_zoom_args)."""
    parser.add_argument('--sampler_power', type=float, default=0.5,
                        help='Oversampling by class count ** -p: 1 = fully balanced, 0.5 = sqrt-balanced, 0 = none.')
    parser.add_argument('--class_weight_power', type=float, default=0.5,
                        help='Loss class weights count ** -p (5-class: mean 1 over train samples; binary: pos_weight). 0 = unweighted.')
    parser.add_argument('--crop_borders', action='store_true', default=True,
                        help='Trim black frame borders (data.dataset.crop_black_borders) before resizing (default).')
    parser.add_argument('--no_crop_borders', dest='crop_borders', action='store_false',
                        help='Keep the black borders.')
    add_zoom_args(parser)


def add_zoom_args(parser):
    """--zoom (eval center zoom, default 1.1) / --train_zoom LO HI (random zoom + shift, default 1.05 1.3);
    --no_zoom = plain resize in train and val (evaluate such runs with --no_zoom too)."""
    parser.add_argument('--zoom', type=float, default=1.1, help='Val/eval center zoom-in factor (1 = none).')
    parser.add_argument('--train_zoom', type=float, nargs=2, default=list(TRAIN_ZOOM), metavar=('LO', 'HI'),
                        help='Train zoom-in range; the crop is placed at a random position (spatial shift).')
    parser.add_argument('--no_zoom', action='store_true', help='No zoom in train or val.')


def zoom_settings(args):
    """(val zoom, train zoom range or None) from add_zoom_args' flags."""
    return (1.0, None) if args.no_zoom else (args.zoom, tuple(args.train_zoom))


def make_criterion(train_ds, binary_classification, class_weight_power=0.0):
    """Training loss with class weights count ** -class_weight_power (0 = unweighted). 5-class: weights normalized to mean 1
    over the training samples; binary: BCE pos_weight (neg / pos) ** class_weight_power."""
    targets = np.array(train_ds.labels if binary_classification else train_ds.grades).astype(int)
    class_counts = np.bincount(targets, minlength=2 if binary_classification else 5)
    if binary_classification:
        if class_weight_power == 0:
            return nn.BCEWithLogitsLoss()
        pos_weight = (class_counts[0] / class_counts[1]) ** class_weight_power
        print(f"Class weights (power {class_weight_power}): pos_weight {pos_weight:.3f}")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    if class_weight_power == 0:
        return nn.CrossEntropyLoss()
    weight = np.maximum(class_counts, 1) ** -float(class_weight_power)
    weight = weight / weight[targets].mean()
    print(f"Class weights (power {class_weight_power}): {np.round(weight, 3).tolist()}")
    return WeightedCrossEntropy(torch.tensor(weight, dtype=torch.float32))


def make_optimizer(model, lr, backbone_lr_mult=None, weight_decay=0.0):
    """AdamW over the trainable params. With backbone_lr_mult, the backbone gets its own group at lr * backbone_lr_mult.
    param_groups[0] is always the head group."""
    if backbone_lr_mult is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    else:
        backbone_param_ids = {id(p) for p in model.backbone.parameters()}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [{"params": head_params, "lr": lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def make_cosine_scheduler(optimizer, epochs, min_lr=1e-7):
    """Per-epoch cosine decay of every param group down to min_lr."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


SELECTION_METRICS = ("default", "combined")

# One row per finished run (any script using EpochLogger): dataset/split, every CLI arg, best + last epoch metrics.
RESULTS_CSV = "results/run_results.csv"


def append_run_result(row, path=RESULTS_CSV):
    """Append row to the shared results CSV. Columns are the union over all rows (new args/metrics add
    columns, older rows get blanks); the file is locked so runs finishing at the same time don't collide."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, fieldnames = [], []
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        fieldnames += [k for k in row if k not in fieldnames]
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows + [row])
        os.replace(tmp, path)


def selection_score(m, binary_classification, selection="default"):
    """Checkpoint-selection score from a validate_* metrics dict. "default": PR-AUC (binary) / QWK (5-class);
    "combined" (5-class only): mean of QWK, referable recall and accuracy."""
    if selection == "combined":
        if binary_classification:
            raise ValueError("selection 'combined' is only defined for 5-class (QWK + referable recall + acc)")
        return (m['qwk'] + m['referable_recall'] + m['acc']) / 3
    return m['pr_auc'] if binary_classification else m['qwk']


class EpochLogger:
    """Per-epoch TensorBoard logging, confusion matrix, console line, best-checkpoint
    saving (selection metric: see selection_score) and the last epoch's weights (best_x.pth -> last_x.pth)."""

    def __init__(self, run_name, run_dir, args, binary_classification, checkpoint_name, cm_to_tensorboard=False,
                 selection="default"):
        self.run_dir = run_dir
        self.binary = binary_classification
        self.checkpoint_path = os.path.join(run_dir, checkpoint_name)
        self.last_checkpoint_path = os.path.join(run_dir, f"last_{checkpoint_name.removeprefix('best_')}")
        self.cm_to_tensorboard = cm_to_tensorboard
        self.selection = selection
        self.class_names = ['non-referable', 'referable'] if binary_classification else ['0', '1', '2', '3', '4']
        self.writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
        self.writer.add_text("args", str(args))
        # -inf (not 0.0) so a checkpoint is always written, even if the metric stays <= 0.
        self.best_selection_metric = float("-inf")
        self.best_metrics = None
        self.best_epoch = None
        self.last_metrics = None
        self.last_train_loss = None
        self.epochs_run = 0
        self.run_name = run_name
        self.args = args

    def log_epoch(self, epoch, model, optimizer, train_loss, m, extra_scalars=None):
        """m: metrics dict from a validate_* loop (incl. 'loss', 'y_true', 'y_pred')."""
        w = self.writer
        cm_path, _ = save_confusion_matrix(m['y_true'], m['y_pred'], epoch, self.run_dir, class_names=self.class_names)
        if self.cm_to_tensorboard:
            w.add_image('ConfusionMatrix', np.array(Image.open(cm_path).convert('RGB')).transpose(2, 0, 1), epoch)

        selection_metric = selection_score(m, self.binary, self.selection)
        self.last_metrics = {k: v for k, v in m.items() if k not in ('y_true', 'y_pred')}
        self.last_train_loss = train_loss
        self.epochs_run = epoch + 1
        w.add_scalar('Meta/Selection_Metric', selection_metric, epoch)
        if selection_metric > self.best_selection_metric:
            self.best_selection_metric = selection_metric
            self.best_metrics = {k: v for k, v in m.items() if k not in ('y_true', 'y_pred')}
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            w.add_scalar('Meta/Best_Selection_Metric', selection_metric, epoch)
        torch.save(model.state_dict(), self.last_checkpoint_path)

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

    def results_row(self):
        """Shared-CSV row: run identity, dataset/split, all args (hp_*), best/last aggregate metrics (no per-class)."""
        args = vars(self.args)
        main_spec = getattr(sys.modules["__main__"], "__spec__", None)
        row = {
            "finished": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "run_name": self.run_name,
            "script": main_spec.name if main_spec else os.path.basename(sys.argv[0]),
            "run_dir": self.run_dir,
            "dataset": "paxos2025+paxos2020" if args.get("fused_dataset") else "paxos2025",
            "split": os.path.basename(os.path.normpath(args.get("split_path", ""))),
            "split_path": args.get("split_path"),
            "train_json": args.get("train_json"),
            "val_json": args.get("val_json"),
            "task": "binary" if self.binary else "5-class",
            "selection": self.selection,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_selection_metric": round(self.best_selection_metric, 4),
        }
        row.update({f"hp_{k}": v for k, v in args.items() if k not in ("split_path", "train_json", "val_json")})
        for prefix, metrics in (("best", self.best_metrics), ("last", self.last_metrics)):
            for k, v in (metrics or {}).items():
                if "_class" not in k:
                    row[f"{prefix}_{k}"] = round(float(v), 4)
        if self.last_train_loss is not None:
            row["last_train_loss"] = round(float(self.last_train_loss), 4)
        return row

    def close(self):
        self.writer.close()
        if self.epochs_run:
            append_run_result(self.results_row())
            print(f"Appended results to {RESULTS_CSV}")
        print(f"Done. Run directory: {self.run_dir}")
