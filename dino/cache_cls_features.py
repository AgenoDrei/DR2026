"""Cache frozen DINOv3 CLS tokens of every `--frame_stride`-th frame of each clip in a split dir's MIL jsons:
view 0 = clean (val transform), views 1..K = independent train-augmented views (per frame, no zoom).
Writes <out_dir>/<vid_name stem>.npy ([1+K, F, 1024] fp16) + <stem>.json (frame indices, grade); resumable per clip.
Usage (repo root): gpu run python -m dino.cache_cls_features --split_dir data/stratified_splits_paxos2025_patient_q5_v3
  --out_dir data/cls_cache/lvd_q5v3 [--classifier_checkpoint ...] --shard_index 0 --num_shards 2"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.dataset import MILVideoDataset, crop_black_borders
from misc.training import seed_everything, video_search_dirs
from .utils import load_dino_backbone, load_backbone_from_classifier, make_train_transform_dino, make_val_transform_dino


class ClipFrameStream(IterableDataset):
    """Decodes each clip once (workers split the clips) and yields chunks
    (clip_idx, frame indices, [n, 1+K, 3, H, W] views, is_last)."""

    def __init__(self, items, stride, views, img_size, chunk, seed):
        self.items, self.stride, self.views, self.chunk, self.seed = items, stride, views, chunk, seed
        self.clean = make_val_transform_dino(img_size, complex_augs=True, zoom=1.0)
        self.aug = make_train_transform_dino(img_size, train_zoom=None)

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for ci in range(wid, len(self.items), nw):
            torch.manual_seed(self.seed + ci)  # augmented views reproducible per clip
            cap = cv2.VideoCapture(self.items[ci]["path"])
            idx, buf, i = [], [], 0
            while True:
                if i % self.stride:
                    if not cap.grab():
                        break
                    i += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                img = Image.fromarray(crop_black_borders(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                buf.append(torch.stack([self.clean(img)] + [self.aug(img) for _ in range(self.views)]))
                idx.append(i)
                i += 1
                if len(buf) == self.chunk:
                    yield ci, idx, torch.stack(buf), False
                    idx, buf = [], []
            cap.release()
            yield ci, idx, (torch.stack(buf) if buf else torch.empty(0)), True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, help="Dir with split_<n>/mil_{train,val}.json; the union of clips is cached.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--classifier_checkpoint", default=None, help="DinoClassifier .pth (backbone.* keys); default plain LVD.")
    p.add_argument("--views", type=int, default=4, help="Augmented views K (+ 1 clean view).")
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--chunk", type=int, default=8, help="Frames per worker chunk (x (1+K) images).")
    p.add_argument("--batch", type=int, default=40, help="Images per backbone forward.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)

    labels = {}
    for f in sorted(glob.glob(os.path.join(args.split_dir, "split_*", "mil_train.json"))
                    + glob.glob(os.path.join(args.split_dir, "split_*", "mil_val.json"))):
        with open(f) as fh:
            labels.update(json.load(fh))
    tmp_json = os.path.join(args.out_dir, f".labels_shard{args.shard_index}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(tmp_json, "w") as fh:
        json.dump(labels, fh)
    ds = MILVideoDataset(tmp_json, search_dir_paths=video_search_dirs(False))  # vid_name -> clip path matching
    os.remove(tmp_json)
    items = sorted(ds.data, key=lambda d: d["vid_name"])
    stem = lambda d: os.path.splitext(d["vid_name"])[0]
    items = [d for k, d in enumerate(items) if k % args.num_shards == args.shard_index
             and not os.path.exists(os.path.join(args.out_dir, stem(d) + ".npy"))]
    print(f"{len(labels)} clips in {args.split_dir}; shard {args.shard_index}/{args.num_shards}: {len(items)} to do")

    device = torch.device("cuda")
    backbone = load_dino_backbone()
    if args.classifier_checkpoint:
        load_backbone_from_classifier(backbone, args.classifier_checkpoint)
    backbone.eval().to(device)
    meta_common = dict(views=args.views, frame_stride=args.frame_stride, img_size=args.img_size,
                       backbone=args.classifier_checkpoint or "lvd",
                       clean="make_val_transform_dino(complex_augs, zoom 1)",
                       aug="make_train_transform_dino(complex_augs, no zoom), per frame", crop_borders=True)

    loader = DataLoader(ClipFrameStream(items, args.frame_stride, args.views, args.img_size, args.chunk, args.seed),
                        batch_size=None, num_workers=args.num_workers, prefetch_factor=4)
    feats, idxs, done, t0 = {}, {}, 0, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for ci, idx, x, last in loader:
            if x.numel():
                n, v = x.shape[:2]
                flat = x.flatten(0, 1)
                out = torch.cat([backbone.forward_features(flat[s:s + args.batch].to(device, non_blocking=True))
                                 ["x_norm_clstoken"] for s in range(0, len(flat), args.batch)])
                feats.setdefault(ci, []).append(out.view(n, v, -1).half().cpu())
                idxs.setdefault(ci, []).extend(idx)
            if last:
                item = items[ci]
                arr = torch.cat(feats.pop(ci)).transpose(0, 1).numpy()  # [1+K, F, 1024]
                base = os.path.join(args.out_dir, stem(item))
                with open(base + ".json", "w") as fh:
                    json.dump(dict(meta_common, vid_name=item["vid_name"], path=item["path"], grade=item["grade"],
                                   frame_indices=idxs.pop(ci)), fh)
                np.save(base + ".tmp.npy", arr)
                os.replace(base + ".tmp.npy", base + ".npy")
                done += 1
                el = time.time() - t0
                print(f"[{done}/{len(items)}] {item['vid_name']}: {arr.shape[1]} frames, "
                      f"{el / 60:.1f} min, ETA {el / done * (len(items) - done) / 60:.0f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
