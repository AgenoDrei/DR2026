"""Post-hoc variants of a vis_video_results.py run's 5-class predictions, written as a new eval dir in the same CSV format:
- <model>_ecd: expected-cost decoding, pred = argmin_j sum_k p_k (j - k)^2 (QWK-aware, untuned; probabilities unchanged);
- ensembles: mean probabilities of several models (videos a member didn't score use the remaining members), + their _ecd.
Usage: python post_hoc.py --input_dir <eval dir> --output_dir <new dir> [--ensemble NAME M1 M2 ...]
Then make_binary.py / analyse_runs.py / paired_tests.py on the new dir.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

NUM_CLASSES = 5
DEFAULT_ENSEMBLES = [["Ens4", "DINO", "DINO_FT", "CNX", "CNX_mix"], ["Ens4_Clf", "DINO", "DINO_FT", "CNX", "CNX_mix", "Clf"]]


def probs_of(df, m):
    return df[[f"{m}_prob_{c}" for c in range(NUM_CLASSES)]].to_numpy()


def ecd(p):
    cost = (np.arange(NUM_CLASSES)[:, None] - np.arange(NUM_CLASSES)[None, :]) ** 2  # [pred j, true k]
    return np.argmin(p @ cost.T, axis=1)


def add_model(df, name, p, pred, split):
    for c in range(NUM_CLASSES):
        df[f"{name}_prob_{c}"] = p[:, c]
    df[f"{name}_pred_grade_split_{split}"] = pd.array(np.where(np.isnan(p).any(1), pd.NA, pred), dtype="Int64")


def main(args):
    ensembles = args.ensemble or DEFAULT_ENSEMBLES
    os.makedirs(args.output_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(args.input_dir, "predictions_split_*.csv"))):
        split = re.search(r"split_(\d+)", path).group(1)
        df = pd.read_csv(path, index_col="video")
        models = [c[:-len(f"_pred_grade_split_{split}")] for c in df.columns if c.endswith(f"_pred_grade_split_{split}")]
        for name, *members in ensembles:
            p = np.nanmean(np.stack([probs_of(df, m) for m in members]), axis=0)
            add_model(df, name, p, p.argmax(1), split)
            models.append(name)
        for m in models:
            p = probs_of(df, m)
            add_model(df, f"{m}_ecd", p, ecd(np.nan_to_num(p)), split)
        df.to_csv(os.path.join(args.output_dir, os.path.basename(path)))
        print(f"{path} -> {args.output_dir}: {len(models)} models + _ecd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ensemble", nargs="+", action="append", metavar="NAME MODEL",
                        help="Repeatable: NAME followed by member models (default: Ens4 = the four bag models, Ens4_Clf = + Clf).")
    main(parser.parse_args())
