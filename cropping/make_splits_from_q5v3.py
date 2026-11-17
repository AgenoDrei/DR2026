"""Build the 5-fold splits for the cropping and informative-frame stages from the
supervisor's patient-level q5_v3 splits, extended with the remaining annotated videos.

Basis: data/stratified_splits_paxos2025_patient_q5_v3/split_<i>/frame_{train,val}_2025.json
(clip ids -> DR grade). The clip ids are joined at the VIDEO level (their naming differs
from our frame names: `<video>_frame<N>_score<S>.png` vs our `<video>_f<N>.png` /
`<video>_frame_<N>.png`, and the frame indices are not comparable -- cleaned vs raw videos),
so only the video stem is used.

Videos missing from q5_v3 (dropped by its quality filter, e.g. clips with <= 139 cleaned
frames) are added without patient-level leakage:

- If the video's PATIENT already has videos in q5_v3, the video joins the SAME fold as its
  patient (a patient whose right eye was removed for quality but whose left eye is in
  q5_v3 fold i goes to fold i too -- never a different fold, which would put the patient
  in both train and val of some fold).
- Patients without any q5_v3 video are distributed over the 5 folds with
  StratifiedGroupKFold (patient groups, stratified by the patient's mean informative-frame
  label), the same patient-level mechanism q5_v3 itself was built with.

Outputs -- both per-row CSVs in the same format (one row per frame per fold,
names indicate the q5_v3 basis):
    cropping/split_cropping_5fold_paxos2025_from_q5v3.csv
                              (filename,video_id,fold,split; consumed by train_segmentation,
                               evaluate_segmentation and extract_inf_frames via --splits_csv)
    cropping/split_cropping_5fold_paxos2025_from_q5v3_metadata.csv
    informative_frames/split_info_5fold_paxos2025_from_q5v3.csv
                              (filename,video_id,label,fold,split; consumed by crop_frames /
                               classify_frames / evaluate_classification)

Run from the repo root:
    python -u -m cropping.make_splits_from_q5v3
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

FRAME_RE = re.compile(r"^(?P<stem>.*)_frame_(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE)
IMG_RE = re.compile(r"IMG_(\d+)", re.IGNORECASE)
EYE_RE = re.compile(r"^([A-Za-z]+\d+)[LR]\d*$")

Q5_FRAME_RE = re.compile(r"^(?P<stem>.+?)_frame\d+(?:_score[\d.]+)?\.png$", re.IGNORECASE)
INF_FRAME_RE = re.compile(r"^(?P<stem>.+?)_f\d+\.png$", re.IGNORECASE)


def frame_stem(filename):
    """Video stem of a cropping annotation filename (<video>_frame_<idx>.png)."""
    m = FRAME_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def subject_of(stem):
    """Patient id of a video stem: R008R/R008R2/R008L -> R008, ..._IMG_4608 ... -> IMG_4608."""
    m = IMG_RE.search(stem)
    if m:
        return f"IMG_{m.group(1)}"
    m = EYE_RE.match(stem)
    if m:
        return m.group(1).upper()
    return stem


def q5_video_stem(clip_id):
    m = Q5_FRAME_RE.match(clip_id)
    return m.group("stem") if m else Path(clip_id).stem


def inf_video_stem(filename):
    m = INF_FRAME_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def load_q5_video_folds(q5_dir, n_splits):
    """Video stem -> fold, from the q5_v3 frame_val lists (video-level join).

    A video's fold is the split whose VAL part contains it; in the other splits it is
    train, which is the normal k-fold scheme. Verified: no video is in the val part of
    more than one split, and nothing is in both train and val of the same split."""
    video_fold = {}
    for fold in range(1, n_splits + 1):
        val_path = Path(q5_dir) / f"split_{fold}" / "frame_val_2025.json"
        train_path = Path(q5_dir) / f"split_{fold}" / "frame_train_2025.json"
        if not val_path.exists():
            continue
        val_stems = {q5_video_stem(k) for k in json.load(open(val_path))}
        train_stems = {q5_video_stem(k) for k in json.load(open(train_path))} if train_path.exists() else set()
        overlap = val_stems & train_stems
        assert not overlap, f"{q5_dir}/split_{fold}: videos in both train and val: {sorted(overlap)[:5]}"
        for stem in val_stems:
            assert stem not in video_fold, f"{q5_dir}: video {stem!r} in the val part of multiple folds"
            video_fold[stem] = fold - 1
    return video_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q5_dir", type=str, default="data/stratified_splits_paxos2025_patient_q5_v3")
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv",
                        help="Cropping annotations (all annotated videos come from this file).")
    parser.add_argument("--inf_csv", type=str, default="informative_frames/annotations_informative_2025.csv",
                        help="Informative-frame labels (filename,label); the stratification source "
                             "for patients without q5_v3 videos.")
    parser.add_argument("--out_dir", type=str, default="cropping")
    parser.add_argument("--inf_out_dir", type=str, default="informative_frames")
    parser.add_argument("--cropping_csv_name", type=str,
                        default="split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_splits = args.n_splits

    # 1. q5_v3 basis: video -> fold, patient -> fold
    q5_video_fold = load_q5_video_folds(args.q5_dir, n_splits)
    q5_patient_fold = {}
    for video, fold in q5_video_fold.items():
        patient = subject_of(video)
        if patient in q5_patient_fold and q5_patient_fold[patient] != fold:
            raise AssertionError(
                f"q5_v3 itself leaks: patient {patient!r} has videos in fold "
                f"{q5_patient_fold[patient] + 1} and fold {fold + 1} -- cannot extend safely")
        q5_patient_fold[patient] = fold
    print(f"q5_v3 basis: {len(q5_video_fold)} videos / {len(q5_patient_fold)} patients")

    # 2. All annotated videos (cropping annotations) + informative labels
    ann = pd.read_csv(args.ann_csv)
    crop_stem_of = {fn: frame_stem(fn) for fn in ann["filename"].tolist()}
    all_videos = sorted(set(crop_stem_of.values()))
    print(f"annotated videos: {len(all_videos)}")

    inf = pd.read_csv(args.inf_csv)
    inf["video_id"] = inf["filename"].map(inf_video_stem)
    label_of = dict(zip(inf["video_id"], inf["label"]))
    video_label_mean = inf.groupby("video_id")["label"].mean().to_dict()

    # 3. Fold assignment
    video_fold = {}
    new_patients = []  # patients without any q5_v3 video, to be group-folded
    for video in all_videos:
        patient = subject_of(video)
        if patient in q5_patient_fold:
            video_fold[video] = q5_patient_fold[patient]
        elif video in q5_video_fold:
            video_fold[video] = q5_video_fold[video]
        else:
            video_fold[video] = None  # resolved below
    unresolved = sorted(v for v, f in video_fold.items() if f is None)
    unresolved_patients = sorted({subject_of(v) for v in unresolved})
    print(f"videos not in q5_v3: {len(unresolved)} (patients: {len(unresolved_patients)})")

    if unresolved:
        # patients -> one row each; stratify by mean informative label; groups = patient
        rows = []
        for patient in unresolved_patients:
            vids = [v for v in unresolved if subject_of(v) == patient]
            mean_label = float(np.mean([video_label_mean.get(v, 0.0) for v in vids]))
            rows.append((patient, vids, int(round(mean_label))))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        patients = np.array([r[0] for r in rows])
        y = np.array([r[2] for r in rows])
        groups = patients
        assigned = {}
        for fold, (_, val_idx) in enumerate(sgkf.split(patients, y, groups)):
            for i in val_idx:
                assigned[rows[i][0]] = fold
        for video in unresolved:
            video_fold[video] = assigned[subject_of(video)]

    # 4. Leakage checks
    patient_folds = defaultdict(set)
    for video, fold in video_fold.items():
        patient_folds[subject_of(video)].add(fold)
    multi = {p: sorted(f + 1 for f in fs) for p, fs in patient_folds.items() if len(fs) > 1}
    assert not multi, f"patient in multiple folds: {dict(list(multi.items())[:5])}"
    for fold in range(n_splits):
        val_videos = {v for v, f in video_fold.items() if f == fold}
        train_patients = {subject_of(v) for v, f in video_fold.items() if f != fold}
        val_patients = {subject_of(v) for v in val_videos}
        shared = train_patients & val_patients
        assert not shared, f"fold {fold}: patients in both train and val: {sorted(shared)[:5]}"
    print("leakage checks passed: every patient is in exactly one fold, "
          "never in both train and val of any fold")

    # 5. Cropping split CSV (same per-row format as the informative-frame split CSV)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / args.cropping_csv_name
    with open(crop_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "video_id", "fold", "split"])
        writer.writeheader()
        for fn, stem in sorted(crop_stem_of.items()):
            fold = video_fold[stem]
            for fi in range(n_splits):
                writer.writerow({"filename": fn, "video_id": stem,
                                 "fold": fi, "split": "val" if fi == fold else "train"})
    print(f"Saved {crop_path} ({len(crop_stem_of)} frames x {n_splits} folds)")
    val_frames = sum(1 for fn, stem in crop_stem_of.items() if video_fold[stem] is not None)
    assert val_frames == len(ann), "val rows must cover each frame exactly once"

    meta_rows = []
    for f in range(n_splits):
        val_stems = {s for s, fl in video_fold.items() if fl == f}
        meta_rows.append({
            "fold": f + 1, "set": "train",
            "total": sum(1 for s in crop_stem_of.values() if s not in val_stems),
            "n_videos": len(all_videos) - len(val_stems),
            "n_patients": len({subject_of(s) for s in all_videos if s not in val_stems}),
        })
        meta_rows.append({
            "fold": f + 1, "set": "val",
            "total": sum(1 for s in crop_stem_of.values() if s in val_stems),
            "n_videos": len(val_stems),
            "n_patients": len({subject_of(s) for s in val_stems}),
        })
    meta_path = out_dir / (Path(args.cropping_csv_name).stem + "_metadata.csv")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    print(f"Saved {meta_path}")

    # 6. Informative-frame split CSV (filename,video_id,label,fold,split)
    inf_out_dir = Path(args.inf_out_dir)
    inf_out_dir.mkdir(parents=True, exist_ok=True)
    inf_path = inf_out_dir / "split_info_5fold_paxos2025_from_q5v3.csv"
    with open(inf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "video_id", "label", "fold", "split"])
        writer.writeheader()
        for row in inf.itertuples():
            fold = video_fold[row.video_id]
            for fi in range(n_splits):
                writer.writerow({
                    "filename": row.filename,
                    "video_id": row.video_id,
                    "label": row.label,
                    "fold": fi,
                    "split": "val" if fi == fold else "train",
                })
    print(f"Saved {inf_path} ({len(inf)} frames x {n_splits} folds)")

    # 7. Summary
    print("\nfold summary (videos):")
    for f in range(n_splits):
        vids = [v for v, fl in video_fold.items() if fl == f]
        from_q5 = sum(1 for v in vids if v in q5_video_fold)
        labels = Counter(label_of.get(v) for v in vids)
        print(f"  fold_{f}: {len(vids)} videos ({from_q5} from q5_v3, "
              f"{len(vids) - from_q5} added), {len({subject_of(v) for v in vids})} patients")


if __name__ == "__main__":
    main()
