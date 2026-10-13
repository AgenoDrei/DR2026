"""
Hierarchical DINO MIL training script: anchor-centered regions experiment.

This is a deliberately non-flexible, single-purpose variant of
train_dino_transformer.py, built to test ONE specific idea:

Instead of feeding the transformer 32 single-frame embeddings (one per
anchor frame, as MILVideoDataset does), each of the same 32 anchor frames
is expanded into a 16-frame "region" of its surrounding, temporally-nearby
informative frames (padding by duplication where the clip is too short in
one direction). A two-stage attention model then:

  1. (local stage) runs one self-attention block WITHIN each 16-frame
     region, independently, pooling it down to a single region-summary
     token via a shared local CLS token -- this is the "denoise a region
     before trusting it" step.
  2. (global stage) runs one self-attention block over the resulting 32
     region-summary tokens, with a global CLS token, exactly like the
     existing flat model did over 32 frame tokens -- this is the MIL
     aggregation step, just operating on region tokens instead of frame
     tokens.

Everything about this is hardcoded on purpose (32 regions, 16 frames/region,
exactly one local + one global attention block, frozen backbone, no
positional embedding, no alternate sampling modes): this script exists to
answer one question cheaply, not to be a general framework. If the region
window size, region count, or block count ever need to vary, that's a sign
this script has served its purpose and the idea should be promoted back
into the flexible pipeline.

Dataset note: this script reads the SAME clip + frame-map layout as
MILVideoDatasetRope in dataset.py (a "cleaned_videos" dir of informative
clips, plus a sibling "frame_maps" dir of per-video JSON files giving each
informative frame's index in the ORIGINAL, pre-filtering video). The
anchor-region grouping logic needs that original-video mapping to decide,
per anchor, how many real frames actually exist before/after it in the
clip -- it does not need or use the "gap between frames" sliding-window
idea at all (that's the other, unused approach), so a plain
`informative_to_original` list is all it reads from the frame map.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import save_confusion_matrix

# --------------------------------------------------------------------------
# Fixed experiment configuration. Not exposed as CLI flags on purpose --
# this script tests exactly this configuration and nothing else.
# --------------------------------------------------------------------------
NUM_REGIONS = 32          # number of anchor-centered regions per video (was: num_frames=32)
FRAMES_PER_REGION = 16    # frames making up each region
IMG_SIZE = 512
USE_COMPLEX_AUGS = True
FREEZE_BACKBONE = True
NUM_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1
# The backbone sees NUM_REGIONS * FRAMES_PER_REGION = 512 images in one
# forward pass (16x what the flat 32-frame model sees). Run it through in
# chunks so a single video doesn't blow up GPU memory; this is purely an
# implementation/memory detail, not a modeling choice.
BACKBONE_MICRO_BATCH = 64


# ==========================================================================
# Anchor-centered region grouping (merged in from the standalone windowed
# dataset prototype; the "sliding_gap" mode from that prototype is
# intentionally not included here -- this script only ever uses the
# anchor-centered windows).
# ==========================================================================

def _select_uniform_anchor_local_indices(total_frames, num_anchors=NUM_REGIONS):
    """Same deterministic anchor selection as MILVideoDataset's uniform
    sampling (np.linspace over the clip). When total_frames < num_anchors
    this naturally repeats indices, same as the existing behavior for
    short clips."""
    return np.linspace(0, total_frames - 1, num_anchors, dtype=int).tolist()


def _build_anchor_centered_groups(informative_to_original, anchor_local_indices,
                                   group_size=FRAMES_PER_REGION):
    """
    For each anchor's clip-local index, build a group_size-frame window
    straddling it. The anchor itself takes one slot; the remaining
    (group_size - 1) slots are split into a before/after count. Every
    possible split is scored by (a) how many padding duplicates it would
    need given how many real frames actually exist on each side, then (b)
    how balanced before/after is, then (c) a slight preference for more
    `before` than `after` on remaining ties -- and the best-scoring split
    is used. See train_dino_hierarchical.py's module docstring / the
    original design discussion for examples (an anchor at the very start
    of a clip ends up as 0 before / 15 after; a well-centered anchor in a
    long clip ends up 8 before / 7 after; an anchor one frame from the end
    ends up 14 before / 1 after; etc).

    Returns a list of length NUM_REGIONS, each entry a list of
    `group_size` clip-local frame indices (chronologically ordered,
    boundary frames repeated as needed for padding).
    """
    n = len(informative_to_original)
    groups = []
    remaining = group_size - 1

    for p in anchor_local_indices:
        available_before = p
        available_after = (n - 1) - p

        best_key, best_split = None, None
        for before in range(remaining + 1):
            after = remaining - before
            dup = max(0, before - available_before) + max(0, after - available_after)
            balance = abs(before - after)
            key = (dup, balance, -before)
            if best_key is None or key < best_key:
                best_key, best_split = key, (before, after)

        before, after = best_split
        before_real = min(before, available_before)
        after_real = min(after, available_after)
        before_pad = before - before_real
        after_pad = after - after_real

        group = (
            [0] * before_pad +
            list(range(p - before_real, p)) +
            [p] +
            list(range(p + 1, p + 1 + after_real)) +
            [n - 1] * after_pad
        )
        assert len(group) == group_size
        groups.append(group)

    return groups


class MILVideoDatasetAnchorRegions(Dataset):
    """
    Same video/frame-map matching convention as MILVideoDatasetRope in
    dataset.py. Instead of a [NUM_REGIONS, C, H, W] flat bag, returns a
    [NUM_REGIONS, FRAMES_PER_REGION, C, H, W] tensor: one anchor-centered
    16-frame region per anchor, always NUM_REGIONS=32 regions.

    Frame map JSON, one per video, matched to the video the same way videos
    themselves are matched (base-name regex), living under
    `frame_map_dir_paths`:

        {
          "total_original_frames": 842,
          "informative_to_original": [5, 6, 7, 9, 10, 13, 14, ...]
        }
    """

    def __init__(self, json_path, transform=None, search_dir_paths=None, frame_map_dir_paths=None):
        with open(json_path, 'r') as f:
            self.label_map = json.load(f)

        self.transform = transform

        if search_dir_paths is None:
            search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos"]
        if frame_map_dir_paths is None:
            frame_map_dir_paths = [p.replace("cleaned_videos", "frame_maps") for p in search_dir_paths]

        all_video_paths = []
        for search_dir_path in search_dir_paths:
            search_dir = Path(search_dir_path)
            if search_dir.exists():
                all_video_paths.extend(search_dir.glob("*.mp4"))
                all_video_paths.extend(search_dir.glob("*.MOV"))

        all_frame_map_paths = []
        for frame_map_dir_path in frame_map_dir_paths:
            frame_map_dir = Path(frame_map_dir_path)
            if frame_map_dir.exists():
                all_frame_map_paths.extend(frame_map_dir.glob("*.json"))

        self.data = []
        missing_count = corrupt_count = missing_map_count = bad_map_count = 0

        print(f"Mapping anchor-region dataset from {json_path}...")

        for vid_name, grade in self.label_map.items():
            base_name = vid_name.replace("CLEAN_", "").replace('.mp4', '').replace('.MOV', '')
            pattern = re.compile(rf"{re.escape(base_name)}(?![a-zA-Z0-9])")

            matched_path = next((p for p in all_video_paths if pattern.search(p.name)), None)
            if not matched_path or not matched_path.exists():
                print(f"Missing: {vid_name}")
                missing_count += 1
                continue
            if matched_path.stat().st_size <= 1000:
                print(f"Corrupt: {matched_path}")
                corrupt_count += 1
                continue

            matched_map_path = next((p for p in all_frame_map_paths if pattern.search(p.name)), None)
            if not matched_map_path or not matched_map_path.exists():
                print(f"Missing frame map: {vid_name}")
                missing_map_count += 1
                continue

            try:
                with open(matched_map_path, 'r') as f:
                    frame_map = json.load(f)
                total_original_frames = int(frame_map["total_original_frames"])
                informative_to_original = list(frame_map["informative_to_original"])
                if total_original_frames <= 0 or len(informative_to_original) == 0:
                    raise ValueError("empty or non-positive fields")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"Bad frame map for {vid_name} ({matched_map_path}): {e}")
                bad_map_count += 1
                continue

            self.data.append({
                'vid_name': vid_name,
                'path': str(matched_path),
                'label': 1.0 if grade >= 2 else 0.0,
                'grade': grade,
                'informative_to_original': informative_to_original,
            })

        print(f"Successfully mapped {len(self.data)} videos.")
        if missing_count:
            print(f"Skipped {missing_count} videos (video file not found).")
        if corrupt_count:
            print(f"Skipped {corrupt_count} videos (found but appear empty/corrupt).")
        if missing_map_count:
            print(f"Skipped {missing_map_count} videos (frame map not found).")
        if bad_map_count:
            print(f"Skipped {bad_map_count} videos (frame map malformed).")

    @property
    def labels(self):
        return [int(item['label']) for item in self.data]

    @property
    def grades(self):
        return [int(item['grade']) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        informative_to_original = item['informative_to_original']
        n = len(informative_to_original)

        anchor_local_indices = _select_uniform_anchor_local_indices(n)
        groups = _build_anchor_centered_groups(informative_to_original, anchor_local_indices)

        regions = self._load_regions(item['path'], groups)  # [NUM_REGIONS, FRAMES_PER_REGION, C, H, W]

        return (
            regions,
            torch.tensor(item['label'], dtype=torch.float32),
            torch.tensor(item['grade']),
        )

    def _load_regions(self, path, groups):
        """Single sequential decode pass over the clip; each clip-local
        frame index is decoded at most once even though it may be reused
        across several regions' slots (padding duplicates, and overlap
        between nearby anchors' windows)."""
        needed = set(i for g in groups for i in g)
        decoded = {}

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        i = 0
        last_valid = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in needed:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                if self.transform:
                    img = self.transform(img)
                decoded[i] = img
                last_valid = img
            if len(decoded) == len(needed):
                break
            i += 1
        cap.release()

        missing = needed - decoded.keys()
        if missing:
            if last_valid is None:
                raise RuntimeError(f"Could not decode any needed frame from {path}.")
            for m in missing:
                decoded[m] = last_valid

        region_tensors = [torch.stack([decoded[i] for i in g]) for g in groups]  # each [FRAMES_PER_REGION, C, H, W]
        return torch.stack(region_tensors)  # [NUM_REGIONS, FRAMES_PER_REGION, C, H, W]


# ==========================================================================
# Model: local (within-region) attention + global (across-region) attention
# ==========================================================================

class TransformerBlock(nn.Module):
    """A single MHA + FFN block (post-norm). Returns output and the raw
    attention weights so the caller can inspect CLS->token attention."""

    def __init__(self, dim, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class DinoHierarchicalAttention(nn.Module):
    """
    Two-stage hierarchical MIL model:

      backbone (frozen) -> [NUM_REGIONS, FRAMES_PER_REGION, L]
      -> local block (batch=NUM_REGIONS, seq=FRAMES_PER_REGION+1 incl. local
         CLS) -> region tokens [NUM_REGIONS, L]
      -> global block (batch=1, seq=NUM_REGIONS+1 incl. global CLS)
      -> classifier(global CLS output)

    Exactly one local block and one global block -- "two layers" -- by
    design, not a configurable num_blocks.
    """

    def __init__(self, num_classes=1, classifier_checkpoint_path=None):
        super().__init__()

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim (e.g., 1024 for ViT-L)
        self.num_classes = num_classes

        self.local_cls_token = nn.Parameter(torch.zeros(1, 1, self.L))
        self.global_cls_token = nn.Parameter(torch.zeros(1, 1, self.L))

        self.local_block = TransformerBlock(self.L)
        self.global_block = TransformerBlock(self.L)

        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

        if classifier_checkpoint_path is not None:
            self._load_classifier_backbone(classifier_checkpoint_path)

        if FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _load_classifier_backbone(self, path):
        """Load only the backbone weights from a DinoClassifier checkpoint.
        Strict: the checkpoint's backbone.* keys must exactly match this
        model's backbone's own key set, or this raises."""
        print(f"Loading backbone from DinoClassifier checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")

        backbone_state = {
            k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")
        }
        if not backbone_state:
            raise ValueError(
                f"No keys starting with 'backbone.' found in {path}. "
                f"Is this really a DinoClassifier checkpoint?"
            )

        self.backbone.load_state_dict(backbone_state, strict=True)
        print(f"Backbone weights loaded strictly ({len(backbone_state)} tensors).")

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for block in (self.local_block, self.global_block):
            for layer in block.ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

            nn.init.xavier_uniform_(block.mha.in_proj_weight)
            if block.mha.in_proj_bias is not None:
                nn.init.constant_(block.mha.in_proj_bias, 0)
            nn.init.xavier_uniform_(block.mha.out_proj.weight)
            if block.mha.out_proj.bias is not None:
                nn.init.constant_(block.mha.out_proj.bias, 0)

        nn.init.normal_(self.local_cls_token, std=1e-6)
        nn.init.normal_(self.global_cls_token, std=1e-6)

        print(f"--- DINO Hierarchical Attention [{self.L} features, "
              f"{NUM_REGIONS} regions x {FRAMES_PER_REGION} frames, "
              f"1 local + 1 global block] weights initialized ---")

    def extract_features(self, x):
        """x: [N, 3, H, W] -> [N, L]. Runs the (frozen) backbone in chunks
        of BACKBONE_MICRO_BATCH to bound memory when N = NUM_REGIONS *
        FRAMES_PER_REGION (512 by default) images are passed at once."""
        feats = []
        for start in range(0, x.size(0), BACKBONE_MICRO_BATCH):
            chunk = x[start:start + BACKBONE_MICRO_BATCH]
            feats.append(self.backbone.forward_features(chunk)["x_norm_clstoken"])
        return torch.cat(feats, dim=0)

    def forward(self, x):
        """
        x: [NUM_REGIONS, FRAMES_PER_REGION, 3, H, W]
        Returns: logits [1, num_classes], region_weights [NUM_REGIONS],
        local_frame_weights [NUM_REGIONS, FRAMES_PER_REGION].
        """
        num_regions, frames_per_region = x.shape[0], x.shape[1]

        flat = x.view(num_regions * frames_per_region, *x.shape[2:])
        h = self.extract_features(flat)  # [num_regions * frames_per_region, L]
        h = h.view(num_regions, frames_per_region, self.L)

        # --- Local stage: one self-attention block per region, run as a
        # batch of `num_regions` independent sequences of length
        # frames_per_region + 1 (region's own local CLS token). ---
        local_cls = self.local_cls_token.expand(num_regions, -1, -1)  # [num_regions, 1, L]
        x_local = torch.cat((local_cls, h), dim=1)  # [num_regions, frames_per_region + 1, L]
        x_local, local_attn = self.local_block(x_local)

        region_tokens = x_local[:, 0, :]  # [num_regions, L]
        local_frame_weights = local_attn[:, 0, 1:]  # [num_regions, frames_per_region]
        local_frame_weights = local_frame_weights / (local_frame_weights.sum(dim=1, keepdim=True) + 1e-12)

        # --- Global stage: one self-attention block over the region
        # tokens, exactly like the flat model's single block over frame
        # tokens, just at region granularity. ---
        region_tokens = region_tokens.unsqueeze(0)  # [1, num_regions, L]
        global_cls = self.global_cls_token.expand(1, -1, -1)  # [1, 1, L]
        x_global = torch.cat((global_cls, region_tokens), dim=1)  # [1, num_regions + 1, L]
        x_global, global_attn = self.global_block(x_global)

        bag_representation = x_global[:, 0, :]  # [1, L]
        logits = self.classifier(bag_representation)  # [1, num_classes]

        region_weights = global_attn[0, 0, 1:]  # [num_regions]
        region_weights = region_weights / (region_weights.sum() + 1e-12)

        return logits, region_weights, local_frame_weights


# ==========================================================================
# Train / validate (self-contained -- not reusing transformer/utils.py's
# train_one_epoch_trans / validate_extended_trans, since those assume the
# flat [num_frames, C, H, W] model interface)
# ==========================================================================

def train_one_epoch(model, loader, criterion, optimizer, binary, device):
    model.train()
    total_loss, n_batches = 0.0, 0

    for regions, label, grade in tqdm(loader, desc="Training"):
        regions = regions.squeeze(0).to(device)  # [NUM_REGIONS, FRAMES_PER_REGION, C, H, W]

        optimizer.zero_grad()
        logits, _, _ = model(regions)

        if binary:
            loss = criterion(logits.view(-1), label.to(device))
        else:
            loss = criterion(logits, grade.to(device).long())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, binary, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for regions, label, grade in loader:
            regions = regions.squeeze(0).to(device)
            logits, _, _ = model(regions)

            if binary:
                loss = criterion(logits.view(-1), label.to(device))
                prob = torch.sigmoid(logits.view(-1)).item()
                y_true.append(int(label.item()))
                y_pred.append(int(prob >= 0.5))
                y_score.append(prob)
            else:
                loss = criterion(logits, grade.to(device).long())
                y_true.append(int(grade.item()))
                y_pred.append(int(torch.argmax(logits, dim=1).item()))

            total_loss += loss.item()
            n_batches += 1

    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'acc': accuracy_score(y_true, y_pred),
        'qwk': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'y_true': y_true,
        'y_pred': y_pred,
    }
    if binary:
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['pr_auc'] = average_precision_score(y_true, y_score)
    else:
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return metrics


def train_hierarchical(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    class_mode = "binary" if args.binary_classification else "all"
    split_name = os.path.basename(args.split_path.rstrip("/"))
    train_tag = "full" if args.train_json == "mil_train.json" else args.train_json.split(".")[0].split("_")[-1]
    run_name = f"dino_hier_anchor32x16_{class_mode}_{split_name}_{timestamp}_{train_tag}"

    run_dir = os.path.join("transformer", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # --- Model ---
    model = DinoHierarchicalAttention(
        num_classes=1 if args.binary_classification else 5,
        classifier_checkpoint_path=args.classifier_checkpoint,
    ).to(device)

    # --- Transforms ---
    train_trans = make_train_transform_dino(IMG_SIZE, USE_COMPLEX_AUGS)
    val_trans = make_val_transform_dino(IMG_SIZE, USE_COMPLEX_AUGS)

    # --- Datasets ---
    if args.fused_dataset:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos", "data/ensemble_results_paxos2020/cleaned_videos"]
    else:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos"]

    train_ds = MILVideoDatasetAnchorRegions(
        os.path.join(args.split_path, args.train_json),
        transform=train_trans,
        search_dir_paths=search_dir_paths,
    )
    val_ds = MILVideoDatasetAnchorRegions(
        os.path.join(args.split_path, args.val_json),
        transform=val_trans,
        search_dir_paths=search_dir_paths,
    )

    # --- Setup Oversampling ---
    train_targets = np.array(train_ds.labels if args.binary_classification else train_ds.grades)
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[train_targets]).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    best_selection_metric = 0.0

    print(f"Starting Hierarchical (anchor-region) Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.binary_classification, device)
        m = validate(model, val_loader, criterion, args.binary_classification, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        class_names = ['non-referable', 'referable'] if args.binary_classification else ['0', '1', '2', '3', '4']
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir, class_names=class_names)

        selection_metric = m['qwk'] if not args.binary_classification else m['pr_auc']
        if selection_metric > best_selection_metric:
            best_selection_metric = selection_metric
            torch.save(model.state_dict(), os.path.join(run_dir, "best_hier_model.pth"))
            writer.add_scalar('Meta/Best_Selection_Metric', best_selection_metric, epoch)

        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)

        skip_keys = {'loss', 'y_true', 'y_pred'}
        for key, value in m.items():
            if key in skip_keys:
                continue
            writer.add_scalar(f'Metric/{key}', value, epoch)

        f1_headline = m['f1'] if args.binary_classification else m['f1_macro']
        headline_metric_name = 'PR-AUC' if args.binary_classification else 'QWK'
        headline_metric_value = m['pr_auc'] if args.binary_classification else m['qwk']
        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {t_loss:.3f} | "
            f"Val Loss: {m['loss']:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {f1_headline:.3f} | "
            f"{headline_metric_name}: {headline_metric_value:.3f}"
        )

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical (32 anchor-centered regions x 16 frames) DINO MIL training. "
                    "Architecture and windowing are fixed by design -- see module docstring."
    )

    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained DinoClassifier model (.pth) to warm-start the backbone from')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--fused_dataset', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')

    args = parser.parse_args()

    train_hierarchical(args)