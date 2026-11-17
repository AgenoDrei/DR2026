"""Leakage probe on cached clean CLS tokens (view 0 of dino/cache_cls_features.py caches; CPU only).
Has the clf_pgv3 backbone pre-fit its training videos (its stills are frames of them)?
1. head:  the split's classifier head on its own cache = its frame predictions; train vs val videos, and accuracy vs distance
          (clip frames) to the nearest still (val stills = quality control: the classifier never saw them).
2. probe: logistic regression on per-video mean-pooled CLS, LVD vs classifier cache, train (resubstitution) vs val per C.
3. cross: videos that are train under one split's backbone and val under another (all available clf caches): paired difference.
Usage (repo root): python -m dino.probe_cached_features --splits 1 2 [--out_dir results/probe/<date>]"""
import argparse
import glob
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

from .utils import make_mlp_head

TXT_DIR = "data/ensemble_results_paxos2025/txt_files"
BINS = [0, 1, 3, 11, 31, 101, 301, 10 ** 9]
BIN_NAMES = ["0", "1-2", "3-10", "11-30", "31-100", "101-300", ">300"]


def stem(v):
    return os.path.splitext(v)[0]


def load_cache(cache_dir):
    """-> {stem: (clean feats [F, 1024] float32, clip frame indices [F])}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npy"))):
        if f.endswith(".tmp.npy"):
            continue
        with open(f[:-4] + ".json") as fh:
            meta = json.load(fh)
        out[stem(meta["vid_name"])] = (np.asarray(np.load(f, mmap_mode="r")[0], dtype=np.float32),
                                       np.asarray(meta["frame_indices"]))
    return out


def still_clip_indices(split_path):
    """-> {video stem: {'train': [clip idx], 'val': [clip idx]}} of the frame_{train,val}.json stills (raw frame -> clip index
    via the informative flags of the extraction txt; stills on non-informative raw frames are dropped)."""
    out, lost, n = {}, 0, 0
    cache_txt = {}
    for role in ["train", "val"]:
        with open(os.path.join(split_path, f"frame_{role}.json")) as fh:
            names = list(json.load(fh))
        for name in names:
            v, rest = name.rsplit("_frame", 1)
            raw = int(rest.split("_")[0])
            if v not in cache_txt:
                p = os.path.join(TXT_DIR, v + ".txt")
                cache_txt[v] = pd.read_csv(p, sep="\t") if os.path.exists(p) else None
            t = cache_txt[v]
            n += 1
            if t is None or raw >= len(t) or not bool(t.informative.iloc[raw]):
                lost += 1
                continue
            clip_idx = int(t.informative.iloc[:raw].sum())
            out.setdefault(v, {"train": [], "val": []})[role].append(clip_idx)
    return out, lost, n


def load_head(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    head = make_mlp_head(1024, 5)
    head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
    return head.eval()


def qwk(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


def part_head(s, split_path, cache, labels, stills, out_rows, frame_rows, video_rows):
    ckpt = json.load(open(next(iter(glob.glob(os.path.join(cache["_dir"], "*.json"))))))["backbone"]
    head = load_head(ckpt)
    for role in ["train", "val"]:
        for v, g in labels[role].items():
            if v not in cache:
                continue
            x, idx = cache[v]
            with torch.no_grad():
                lp = torch.log_softmax(head(torch.from_numpy(x)), -1).numpy()
            p = np.exp(lp)
            st = stills.get(v, {"train": [], "val": []})
            st_all = np.array(st["train"] + st["val"])
            dist = (np.abs(idx[:, None] - st_all[None]).min(1) if len(st_all) else np.full(len(idx), -1))
            group = role + ("_still" if len(st_all) else "_nostill")
            frame_rows.append(pd.DataFrame({"split": s, "video": v, "group": group, "grade": g, "dist": dist,
                                            "correct": p.argmax(1) == g, "nll": -lp[:, g], "maxprob": p.max(1),
                                            "p_true": p[:, g]}))
            video_rows.append(dict(split=s, video=v, role=role, group=group, grade=g, frame_acc=float((p.argmax(1) == g).mean()),
                                   frame_nll=float(-lp[:, g].mean()), maxprob=float(p.max(1).mean()),
                                   vid_pred=int(p.mean(0).argmax()), vid_nll=float(-np.log(p.mean(0)[g] + 1e-9))))


def part_probe(s, caches, labels, Cs, rows):
    for name, cache in caches.items():
        X, y = {}, {}
        for role in ["train", "val"]:
            vids = [v for v in labels[role] if v in cache]
            X[role] = np.stack([cache[v][0].mean(0) for v in vids])
            y[role] = np.array([labels[role][v] for v in vids])
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        Xt, Xv = (X["train"] - mu) / sd, (X["val"] - mu) / sd
        cnt = np.bincount(y["train"], minlength=5).astype(float)
        cw = np.where(cnt > 0, cnt, 1) ** -0.5
        cw = cw / (cw[y["train"]].mean())
        for C in Cs:
            m = LogisticRegression(C=C, max_iter=5000).fit(Xt, y["train"], sample_weight=cw[y["train"]])
            r = dict(split=s, features=name, C=C)
            for role, Xr in [("train", Xt), ("val", Xv)]:
                lp = m.predict_log_proba(Xr)
                full = np.full((len(Xr), 5), -30.0)
                full[:, m.classes_] = lp
                r[f"{role}_nll"] = float(-full[np.arange(len(Xr)), y[role]].mean())
                r[f"{role}_qwk"] = qwk(y[role], full.argmax(1))
                r[f"{role}_acc"] = float((full.argmax(1) == y[role]).mean())
            r["gap_nll"] = r["val_nll"] - r["train_nll"]
            rows.append(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, nargs="+", default=[1, 2])
    p.add_argument("--split_dir", default="data/stratified_splits_paxos2025_patient_q5_v3")
    p.add_argument("--clf_cache", default="data/cls_cache/clf_pgv3_split{}")
    p.add_argument("--lvd_cache", default="data/cls_cache/lvd_q5v3")
    p.add_argument("--out_dir", default=f"results/probe/{date.today()}")
    p.add_argument("--Cs", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(8)

    lvd = load_cache(args.lvd_cache)
    probe_rows, frame_rows, video_rows = [], [], []
    for s in args.splits:
        sp = os.path.join(args.split_dir, f"split_{s}")
        labels = {r: {stem(k): v for k, v in json.load(open(os.path.join(sp, f"mil_{r}.json"))).items()} for r in ["train", "val"]}
        clf = load_cache(args.clf_cache.format(s))
        clf["_dir"] = args.clf_cache.format(s)
        stills, lost, n = still_clip_indices(sp)
        print(f"split {s}: {len(clf) - 1} clf clips, {len(lvd)} lvd clips; {n} stills, {lost} not on an informative frame")
        part_head(s, sp, clf, labels, stills, None, frame_rows, video_rows)
        clf.pop("_dir")
        part_probe(s, {"lvd": lvd, "clf": clf}, labels, args.Cs, probe_rows)

    frames = pd.concat(frame_rows)
    videos = pd.DataFrame(video_rows)
    probe = pd.DataFrame(probe_rows)
    videos.to_csv(os.path.join(args.out_dir, "head_videos.csv"), index=False)
    probe.to_csv(os.path.join(args.out_dir, "probe.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n== 1. classifier head on its own cache: per video group (frame-level means, video-level QWK of mean probs)")
    g = videos.groupby(["split", "group"])
    t = g[["frame_acc", "frame_nll", "maxprob", "vid_nll"]].mean()
    t["vid_qwk"] = g.apply(lambda d: qwk(d.grade, d.vid_pred))
    t["n"] = g.size()
    print(t.round(3).to_string())
    f = frames[frames.dist >= 0].copy()
    f["dist_bin"] = pd.cut(f.dist, BINS, right=False, labels=BIN_NAMES)
    prof = f.groupby(["group", "dist_bin"], observed=True)[["correct", "nll", "maxprob", "p_true"]].mean()
    prof["n_frames"] = f.groupby(["group", "dist_bin"], observed=True).size()
    prof.to_csv(os.path.join(args.out_dir, "head_distance_profile.csv"))
    print("\n== 1b. frame metrics vs distance (clip frames) to the nearest still, all splits pooled")
    print(prof.round(3).to_string())

    print("\n== 2. logistic-regression probe on mean-pooled CLS (train = resubstitution)")
    print(probe.groupby(["features", "C"])[["train_nll", "val_nll", "gap_nll", "train_qwk", "val_qwk"]].mean().round(3).to_string())

    cross = videos.pivot_table(index="video", columns="role", values=["frame_acc", "frame_nll", "maxprob"], aggfunc="mean").dropna()
    if len(cross):
        print(f"\n== 3. cross-split paired: {len(cross)} videos seen as train under one backbone and as val under another")
        for m in ["frame_acc", "frame_nll", "maxprob"]:
            d = cross[(m, "train")] - cross[(m, "val")]
            print(f"  {m}: train-backbone minus val-backbone {d.mean():+.3f} (SE {d.std() / np.sqrt(len(d)):.3f}), "
                  f"train {cross[(m, 'train')].mean():.3f} vs val {cross[(m, 'val')].mean():.3f}")
        cross.to_csv(os.path.join(args.out_dir, "cross_split.csv"))
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
