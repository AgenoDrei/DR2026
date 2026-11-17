"""Bag-transformer head (transformer/model.py::BagTransformer) on cached frozen DINOv3 CLS features
(dino/cache_cls_features.py: <cache_dir>/<stem>.npy = [1+K, F, 1024], view 0 clean). Trains in minutes, so head-side
questions (batching, EMA, regularization, narrow head, feature augs) can run over many splits x seeds.
Defaults = the uncached frozen `dfz` config (batch 1 bag, no EMA, per-frame random augmented view).
--ema_epochs: validate/checkpoint EMA weights (best/last_sa_model.pth = EMA; raw last = raw_last_sa_model.pth).
Head weights transfer to DinoSelfAttention (same non-backbone keys, CLS path, no PCA).
--ordinal coral: one bag score s + 4 ordered thresholds, P(grade > k) = sigmoid(s + b_k); the model returns the implied 5-class
log-probabilities (val loss = their NLL, comparable to CE); trained with the CORAL loss (sum of 4 BCEs) or --ordinal_loss nll."""
import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Dataset

from misc.training import (seed_everything, make_run_dir, make_balanced_loaders, make_optimizer, EpochLogger,
                           SELECTION_METRICS, make_criterion, add_balancing_args)
from transformer.model import BagTransformer
from transformer.utils import validate_extended_trans
from .probe_cached_features import still_clip_indices


class CachedBagDataset(Dataset):
    """Bags of cached CLS tokens. Train: random-segment frames over the cached (strided) frames, each frame from a random
    augmented view 1..views (one per bag with bag_consistent_views; view_mix interpolates two views per frame), optional
    min_bag_size / frame_dropout. Val: uniform bag of the clean view (eval_random: random-segment, clean view)."""

    def __init__(self, json_path, cache_dir, num_frames=32, train=False, views=4, random_segment_sample=True,
                 bag_consistent_views=False, view_mix=False, frame_dropout=0.0, min_frames=8, min_bag_size=None):
        with open(json_path) as f:
            label_map = json.load(f)
        self.items, missing = [], []
        for vid, grade in label_map.items():
            path = os.path.join(cache_dir, os.path.splitext(vid)[0] + ".npy")
            (self.items.append(dict(vid_name=vid, path=path, grade=int(grade))) if os.path.exists(path)
             else missing.append(vid))
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips of {json_path} missing in {cache_dir}, e.g. {missing[:3]}")
        self.num_frames, self.train, self.views = num_frames, train, views
        self.random_segment_sample, self.bag_consistent_views, self.view_mix = random_segment_sample, bag_consistent_views, view_mix
        self.frame_dropout, self.min_frames, self.min_bag_size = frame_dropout, min_frames, min_bag_size
        self.eval_random = False
        self._arrays = {}
        self.allowed = {}  # item index -> cached frame positions train bags may use (mask_stills)
        if train and views:
            available = self._array(0).shape[0] - 1
            if views > available:
                raise ValueError(f"--views {views} but the cache has {available} augmented views")

    @property
    def grades(self):
        return [d["grade"] for d in self.items]

    @property
    def labels(self):
        return [int(g >= 2) for g in self.grades]

    def __len__(self):
        return len(self.items)

    def _array(self, i):
        if i not in self._arrays:  # opened lazily, per worker
            self._arrays[i] = np.load(self.items[i]["path"], mmap_mode="r")
        return self._arrays[i]

    def _indices(self, n, random):
        """Same rule as MILVideoDataset._sample_indices (uniform linspace / one random frame per segment)."""
        if not random:
            return np.linspace(0, n - 1, self.num_frames).astype(int)
        b = np.linspace(0, n, self.num_frames + 1).astype(int)
        lo, width = b[:-1], np.maximum(b[1:] - b[:-1], 1)
        return np.minimum(lo + (torch.rand(self.num_frames).numpy() * width).astype(int), n - 1)

    def mask_stills(self, split_path, radius):
        """Train bags skip cached frames within `radius` clip frames of this split's training stills (frame_train.json), which
        the classifier backbone memorized (dino/probe_cached_features.py). -> fraction of frames masked."""
        stills, _, _ = still_clip_indices(split_path)
        total = masked = 0
        for i, d in enumerate(self.items):
            st = np.array(stills.get(os.path.splitext(d["vid_name"])[0], {}).get("train", []))
            if not len(st):
                continue
            with open(d["path"][:-4] + ".json") as f:
                fidx = np.asarray(json.load(f)["frame_indices"])
            keep = np.abs(fidx[:, None] - st[None]).min(1) > radius
            total, masked = total + len(fidx), masked + int((~keep).sum())
            if keep.sum() >= self.num_frames:
                self.allowed[i] = np.flatnonzero(keep)
        return masked / max(total, 1)

    def __getitem__(self, i):
        a = self._array(i)
        if self.train and i in self.allowed:  # random segments over the unmasked frames
            pos = self.allowed[i]
            idx = pos[self._indices(len(pos), self.random_segment_sample)]
        else:
            idx = self._indices(a.shape[1], (self.train and self.random_segment_sample) or self.eval_random)
        if self.train:
            if self.min_bag_size:
                k = int(torch.randint(self.min_bag_size, self.num_frames + 1, ()))
                idx = idx[np.sort(torch.randperm(len(idx))[:k].numpy())]
            if self.frame_dropout > 0:
                keep = torch.rand(len(idx)).numpy() >= self.frame_dropout
                if keep.sum() < min(self.min_frames, len(idx)):
                    keep[torch.randperm(len(idx))[:min(self.min_frames, len(idx))].numpy()] = True
                idx = idx[keep]
        if self.train and self.views:
            if self.bag_consistent_views:
                v = np.full(len(idx), int(torch.randint(1, self.views + 1, ())))
            else:
                v = torch.randint(1, self.views + 1, (len(idx),)).numpy()
            x = np.asarray(a[v, idx], dtype=np.float32)
            if self.view_mix:
                lam = torch.rand(len(idx), 1).numpy()
                x = lam * x + (1 - lam) * np.asarray(a[torch.randint(1, self.views + 1, (len(idx),)).numpy(), idx], dtype=np.float32)
        else:
            x = np.asarray(a[0, idx], dtype=np.float32)
        g = self.items[i]["grade"]
        return torch.from_numpy(x), torch.tensor(float(g >= 2)), torch.tensor(g)


class CachedBagTransformer(BagTransformer):
    """BagTransformer on precomputed frame tokens [T, in_dim]; optional fixed PCA projection (buffers, no trainable
    params) to pca_dim before the head. Rebuild with CachedBagTransformer.from_checkpoint(run_dir)."""

    def __init__(self, in_dim=1024, num_heads=8, ffn_dim=2048, dropout=0.1, attn_dropout=0.0, num_blocks=2,
                 num_frames=32, num_classes=5, head_dim=None, pca_dim=None, pca_whiten=False, ordinal=None, full_checkpoint_path=None):
        super().__init__()
        self.pca_dim, self.pca_whiten, self.ordinal = pca_dim, pca_whiten, ordinal
        self.cum_logits = None  # [1, num_classes - 1] of the last forward (ordinal)
        if ordinal:  # thresholds b_k = b_0 - cumsum(softplus(gap)): decreasing, so the class probabilities stay >= 0
            self.coral_b0 = nn.Parameter(torch.zeros(1))
            self.coral_gap = nn.Parameter(torch.full((num_classes - 2,), math.log(math.e - 1)))  # softplus = 1
        if pca_dim:
            self.register_buffer("pca_mean", torch.zeros(in_dim))
            self.register_buffer("pca_proj", torch.zeros(pca_dim, in_dim))
        self._build_head(pca_dim or in_dim, num_heads, ffn_dim, dropout, num_blocks, num_frames, 1 if ordinal else num_classes,
                         head_dim=head_dim, attn_dropout=attn_dropout)
        if full_checkpoint_path:
            self._load_full_checkpoint(full_checkpoint_path)

    def extract_features(self, x):
        return (x - self.pca_mean) @ self.pca_proj.T if self.pca_dim else x

    def forward_head(self, h):
        out, frame_w, region_w = super().forward_head(h)
        if not self.ordinal:
            return out, frame_w, region_w
        b = self.coral_b0 - torch.cat([self.coral_gap.new_zeros(1), torch.cumsum(F.softplus(self.coral_gap), 0)])
        self.cum_logits = out.float() + b  # [1, C-1]: logit P(grade > k)
        c = torch.sigmoid(self.cum_logits)
        one, zero = c.new_ones(c.size(0), 1), c.new_zeros(c.size(0), 1)
        p = torch.cat([one, c], 1) - torch.cat([c, zero], 1)  # P(y = k) = P(y > k-1) - P(y > k)
        return torch.log(p.clamp_min(1e-7)), frame_w, region_w


def coral_loss(cum_logits, grade, sample_weight):
    """CORAL: sum over k of BCE(sigmoid(s + b_k), grade > k), per-sample class weight."""
    levels = (grade.unsqueeze(1) > torch.arange(cum_logits.size(1), device=grade.device)).float()
    return (F.binary_cross_entropy_with_logits(cum_logits, levels, reduction="none").sum(1) * sample_weight).mean()


def median_qwk(probs, y):
    """QWK of the median prediction (CORAL's rank rule: number of P(grade > k) > 0.5)."""
    pred = (np.cumsum(probs[:, ::-1], 1)[:, ::-1][:, 1:] > 0.5).sum(1)
    return cohen_kappa_score(y, pred, weights="quadratic")


def feature_stats(ds, frame_step=4):
    """Clean-view frames (every frame_step-th cached frame) of the training clips -> [N, D] float64."""
    return np.concatenate([np.asarray(ds._array(i)[0, ::frame_step], dtype=np.float64) for i in range(len(ds))])


def fit_pca(x, dim, whiten):
    mean = x.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    ratio = np.cumsum(evals) / evals.sum()
    n = {q: int(np.searchsorted(ratio, q) + 1) for q in (0.9, 0.95, 0.99)}
    print(f"PCA on {len(x)} train frames: components for 90/95/99 % variance {n[0.9]}/{n[0.95]}/{n[0.99]}"
          + (f"; {dim} keep {ratio[dim - 1]:.3f}" if dim else ""))
    proj = None
    if dim:
        proj = evecs[:, :dim].T
        if whiten:
            proj = proj / np.sqrt(evals[:dim, None])
    return mean, proj, n, (float(ratio[dim - 1]) if dim else None)


def train_epoch(model, ema, loader, criterion, optimizer, args, device, feat_std):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    total, pending = 0.0, 0
    optimizer.zero_grad()
    for k, (x, label, grade) in enumerate(loader):
        x = x.squeeze(0).to(device)
        target = label.to(device).float().unsqueeze(1) if args.binary_classification else grade.to(device).long()
        if args.feat_noise > 0:
            x = x + torch.randn_like(x) * (args.feat_noise * feat_std)
        if args.feat_dropout > 0:
            x = F.dropout(x, args.feat_dropout)
        logits, _ = model(x)
        if args.ordinal and args.ordinal_loss == "coral":
            w = criterion.weight[target] if hasattr(criterion, "weight") and criterion.weight is not None else torch.ones_like(target, dtype=torch.float)
            loss = coral_loss(model.cum_logits, target, w)
        else:  # CE, or the ordinal model's NLL (its outputs are log-probabilities)
            loss = criterion(logits, target)
        if args.distance_scale_weight > 0 and not args.binary_classification:
            with torch.no_grad():
                scale = 1.0 + args.distance_scale_weight * (logits.argmax(-1) - target).abs().float()
            loss = loss * scale.squeeze()
        (loss / args.batch_size).backward()
        total += loss.item()
        pending += 1
        if pending == args.batch_size or k == len(loader) - 1:
            if pending < args.batch_size:  # partial last batch: mean over the bags it has
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(args.batch_size / pending)
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(params, args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            pending = 0
            if ema is not None:
                ema.update_parameters(model)
    return total / len(loader)


@torch.no_grad()
def predict(model, ds, device):
    model.eval()
    return np.stack([torch.softmax(model(ds[i][0].to(device))[0].float(), -1)[0].cpu().numpy() for i in range(len(ds))])


def bag_stability(model, val_ds, n_bags, device):
    """QWK over n_bags random-segment clean-view val bags and the share of videos whose grade / referral flips."""
    val_ds.eval_random = True
    preds = []
    for b in range(n_bags):
        torch.manual_seed(10_000 + b)
        preds.append(predict(model, val_ds, device).argmax(1))
    val_ds.eval_random = False
    preds, y = np.stack(preds), np.array(val_ds.grades)
    qwk = [cohen_kappa_score(y, p, weights="quadratic") for p in preds]
    return dict(bag_qwk_mean=float(np.mean(qwk)), bag_qwk_sd=float(np.std(qwk)),
                bag_grade_flip=float((preds != preds[:1]).any(0).mean()),
                bag_referral_flip=float(((preds >= 2) != (preds[:1] >= 2)).any(0).mean()))


class CachedLogger(EpochLogger):
    """EpochLogger + extra columns (raw-weight metrics, bag stability, PCA) in the results CSV row."""
    extra_row = {}

    def results_row(self):
        row = super().results_row()
        row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.extra_row.items()})
        return row


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    binary = args.binary_classification
    split = os.path.basename(os.path.normpath(args.split_path))
    description = f"cached_sa_{'binary' if binary else 'all'}_{split}_{os.path.basename(os.path.normpath(args.cache_dir))}"
    # seed in the name: parallel seeds of one split start in the same minute and would share the run dir
    run_name, run_dir = make_run_dir("transformer", description, f"{args.run_tag or 'run'}_seed{args.seed}")

    train_ds = CachedBagDataset(os.path.join(args.split_path, args.train_json), args.cache_dir, args.num_frames, train=True,
                                views=args.views, random_segment_sample=not args.uniform_train,
                                bag_consistent_views=args.bag_consistent_views, view_mix=args.view_mix,
                                frame_dropout=args.frame_dropout, min_frames=args.min_frames, min_bag_size=args.min_bag_size)
    val_ds = CachedBagDataset(os.path.join(args.split_path, args.val_json), args.cache_dir, args.num_frames)
    if args.mask_stills > 0:
        print(f"mask_stills {args.mask_stills}: {train_ds.mask_stills(args.split_path, args.mask_stills):.1%} of the "
              f"training videos' frames (videos with stills) masked")
    train_loader, val_loader = make_balanced_loaders(train_ds, val_ds, binary, num_workers=args.num_workers,
                                                     sampler_power=args.sampler_power)

    stats = feature_stats(train_ds)
    mean, proj, pca_n, pca_kept = fit_pca(stats, args.pca_dim, args.pca_whiten)
    feat_std = torch.tensor(stats.std(0), dtype=torch.float32, device=device)

    config = dict(in_dim=stats.shape[1], num_heads=args.num_heads, ffn_dim=args.ffn_dim, dropout=args.dropout,
                  attn_dropout=args.attn_dropout, num_blocks=args.num_blocks, num_frames=args.num_frames,
                  num_classes=1 if binary else 5, head_dim=args.head_dim,
                  pca_dim=args.pca_dim, pca_whiten=args.pca_whiten, ordinal=args.ordinal)
    with open(os.path.join(run_dir, "architecture_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    model = CachedBagTransformer(**config)
    if args.pca_dim:
        model.pca_mean.copy_(torch.from_numpy(mean).float())
        model.pca_proj.copy_(torch.from_numpy(proj.copy()).float())
    model.to(device)
    print(f"Head params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    ema = None
    if args.ema_epochs:
        decay = 1 - 1 / (args.ema_epochs * steps_per_epoch)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)
        print(f"EMA decay {decay:.4f} per step ({steps_per_epoch} steps/epoch)")
    eval_model = ema.module if ema is not None else model

    train_criterion = make_criterion(train_ds, binary, args.class_weight_power).to(device)
    criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    logger = CachedLogger(run_name, run_dir, args, binary, "best_sa_model.pth", selection=args.selection_metric)
    raw_hist, eval_hist, med_hist = [], [], []
    for epoch in range(args.epochs):
        t_loss = train_epoch(model, ema, train_loader, train_criterion, optimizer, args, device, feat_std)
        m = validate_extended_trans(eval_model, val_loader, criterion, binary, device)
        extra = {}
        if ema is not None:
            raw = validate_extended_trans(model, val_loader, criterion, binary, device)
            raw_hist.append(raw)
            extra = {f"Raw/{k}": v for k, v in raw.items() if k in ("qwk", "acc", "loss", "referable_recall", "pr_auc")}
        eval_hist.append(m)
        if not binary and epoch >= args.epochs - 3:
            med_hist.append(median_qwk(predict(eval_model, val_ds, device), np.array(val_ds.grades)))
        scheduler.step()
        logger.log_epoch(epoch, eval_model, optimizer, t_loss, m, extra)

    key = "pr_auc" if binary else "qwk"
    row = {f"last3_{key}": float(np.mean([h[key] for h in eval_hist[-3:]])),
           "last3_loss": float(np.mean([h["loss"] for h in eval_hist[-3:]])),
           "pca_n95": pca_n[0.95], **({"last3_qwk_median": float(np.mean(med_hist))} if med_hist else {}), "pca_kept_var": pca_kept, "steps_per_epoch": steps_per_epoch}
    if raw_hist:
        torch.save(model.state_dict(), os.path.join(run_dir, "raw_last_sa_model.pth"))
        row.update({f"raw_last_{k}": float(raw_hist[-1][k]) for k in (key, "acc", "loss")},
                   **{f"raw_best_{key}": float(max(h[key] for h in raw_hist))})
    if not binary:
        probs = predict(eval_model, val_ds, device)  # last epoch, uniform clean bags
        with open(os.path.join(run_dir, "val_predictions_last.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vid_name", "grade", "pred"] + [f"p{c}" for c in range(5)])
            for d, p in zip(val_ds.items, probs):
                w.writerow([d["vid_name"], d["grade"], int(p.argmax())] + [f"{v:.5f}" for v in p])
        if args.eval_bags:
            row.update(bag_stability(eval_model, val_ds, args.eval_bags, device))
            print("Bag stability (last):", {k: round(v, 3) for k, v in row.items() if k.startswith("bag_")})
    logger.extra_row = row
    logger.close()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True, help="dino/cache_cls_features.py output dir.")
    p.add_argument("--split_path", required=True)
    p.add_argument("--train_json", default="mil_train.json")
    p.add_argument("--val_json", default="mil_val.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--binary_classification", action="store_true")
    # optimization (defaults: dfz)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1, help="Bags per optimizer step (gradients averaged).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad", type=float, default=0.0, help="Max grad norm per step; 0 = off.")
    p.add_argument("--ema_epochs", type=float, default=0.0,
                   help="EMA of the weights with a horizon of this many epochs (decay = 1 - 1/(h*steps_per_epoch)); 0 = off.")
    p.add_argument("--distance_scale_weight", type=float, default=0.3)
    p.add_argument("--selection_metric", default="combined", choices=SELECTION_METRICS)
    p.add_argument("--ordinal", default=None, choices=["coral"], help="Ordinal head (5-class only); default plain softmax.")
    p.add_argument("--ordinal_loss", default="coral", choices=["coral", "nll"],
                   help="coral: sum of the 4 binary BCEs; nll: NLL of the implied class probabilities (cumulative-link likelihood).")
    # architecture
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ffn_dim", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--head_dim", type=int, default=None, help="Learned LayerNorm + Linear projection to this width.")
    p.add_argument("--pca_dim", type=int, default=None, help="Fixed PCA projection (fitted on train frames) to this width.")
    p.add_argument("--pca_whiten", action="store_true")
    # data / augmentation
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--views", type=int, default=4, help="Augmented cache views used for training (0 = clean view only).")
    p.add_argument("--bag_consistent_views", action="store_true", help="One augmented view per bag instead of per frame.")
    p.add_argument("--view_mix", action="store_true", help="Interpolate two augmented views per frame (lambda ~ U(0, 1)).")
    p.add_argument("--uniform_train", action="store_true", help="Uniform train bags instead of random-segment sampling.")
    p.add_argument("--min_bag_size", type=int, default=None)
    p.add_argument("--frame_dropout", type=float, default=0.0, help="Drop each train frame token with this probability.")
    p.add_argument("--min_frames", type=int, default=8, help="Frames kept at least under --frame_dropout.")
    p.add_argument("--mask_stills", type=int, default=0,
                   help="Train bags skip frames within this many clip frames of the split's training stills (0 = off).")
    p.add_argument("--feat_noise", type=float, default=0.0, help="Gaussian noise sigma, in units of the per-dim train std.")
    p.add_argument("--feat_dropout", type=float, default=0.0, help="Dropout on the feature dimensions of the train tokens.")
    p.add_argument("--eval_bags", type=int, default=8, help="Random val bags for the bag-stability metrics (0 = off).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--run_tag", default=None)
    add_balancing_args(p)  # --sampler_power / --class_weight_power (crop/zoom flags: fixed by the cache)
    p.set_defaults(sampler_power=0.5, class_weight_power=0.5)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    width = args.head_dim or args.pca_dim or 1024
    if width % args.num_heads:
        parser.error(f"--num_heads {args.num_heads} must divide the head width {width}")
    if args.ordinal and args.binary_classification:
        parser.error("--ordinal is 5-class only")
    if args.selection_metric == "combined" and args.binary_classification:
        parser.error("--selection_metric combined is 5-class only")
    main(args)
