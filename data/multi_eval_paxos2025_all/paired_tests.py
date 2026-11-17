"""Paired model comparisons on the 5-class prediction CSVs of a vis_video_results.py run.

Two levels, every model pair on the videos both models scored:
- split level (n = 5 folds): mean paired difference, SE, wins, paired t-test and exact Wilcoxon (min p = 0.0625 at n = 5);
- pooled video level: difference with a 95% patient-clustered bootstrap CI and a paired permutation test that swaps the two
  models' predictions per patient (both eyes of a patient are correlated), plus exact McNemar on per-video correctness
  (ignores the clustering). Holm correction over the model pairs, per metric.
Usage: python paired_tests.py --input_dir <eval dir> [--pairs A:B ...]  ->  <eval dir>/<--out_name>/{paired_tests.csv,summary.csv,summary.md}
"""
import argparse
import glob
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_data import make_patient_id_fn  # noqa: E402

# rr/spec: referral = predicted grade >= 2 (as referable_recall / fixed_decision_*.csv); auc: P(grade >= 2) = sum of probs 2-4;
# off2: share of videos >= 2 grades off (lower is better)
METRICS = ["qwk", "acc", "f1_macro", "rr", "spec", "auc", "off2"]
LOWER_BETTER = {"off2"}


def qwk(y, p, k=5):
    o = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (w * o).sum() / (w * e).sum()


def f1_macro(y, p, k=5):
    f = []
    for c in range(k):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        if tp + fp + fn:  # classes absent from both are skipped, as in sklearn's labels=present
            f.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(f)


def auc(r, s):
    pos, neg = s[r], s[~r]
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics(y, p, s):
    r = y >= 2
    return {"qwk": qwk(y, p), "acc": np.mean(y == p), "f1_macro": f1_macro(y, p),
            "rr": np.mean(p[r] >= 2), "spec": np.mean(p[~r] < 2), "auc": auc(r, s),
            "off2": np.mean(np.abs(y - p) >= 2)}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def load(input_dir):
    dfs = []
    for path in sorted(glob.glob(os.path.join(input_dir, "predictions_split_*.csv"))):
        df = pd.read_csv(path)
        df["split"] = int(re.search(r"split_(\d+)", path).group(1))
        df.columns = [re.sub(r"_split_\d+$", "", c) for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    models = [c[:-len("_pred_grade")] for c in df.columns if c.endswith("_pred_grade")]
    for m in models:
        df[f"{m}_pref"] = df[[f"{m}_prob_{g}" for g in (2, 3, 4)]].sum(axis=1, min_count=3)
    df["patient"] = df["video"].map(make_patient_id_fn())
    return df, models


def pair_arrays(d, a, b):
    y = d["grade"].to_numpy(int)
    pa, pb = d[f"{a}_pred_grade"].to_numpy(int), d[f"{b}_pred_grade"].to_numpy(int)
    sa, sb = d[f"{a}_pref"].to_numpy(), d[f"{b}_pref"].to_numpy()
    return y, pa, pb, sa, sb


def mcnemar(ca, cb):
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def compare(df, a, b, n_boot, n_perm, rng):
    d = df.dropna(subset=[f"{a}_pref", f"{b}_pref"]).reset_index(drop=True)
    y, pa, pb, sa, sb = pair_arrays(d, a, b)
    rows = []

    # split level
    per_split = []
    for s, ds in d.groupby("split"):
        ys, pas, pbs, sas, sbs = pair_arrays(ds, a, b)
        ma, mb = metrics(ys, pas, sas), metrics(ys, pbs, sbs)
        per_split.append({m: ma[m] - mb[m] for m in METRICS})
    per_split = pd.DataFrame(per_split)

    # pooled, patient-clustered bootstrap + per-patient swap permutation
    pat = pd.factorize(d["patient"])[0]
    groups = [np.flatnonzero(pat == g) for g in range(pat.max() + 1)]
    obs_a, obs_b = metrics(y, pa, sa), metrics(y, pb, sb)
    obs = np.array([obs_a[m] - obs_b[m] for m in METRICS])
    boot = np.empty((n_boot, len(METRICS)))
    for i in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(len(groups), size=len(groups))])
        ma, mb = metrics(y[idx], pa[idx], sa[idx]), metrics(y[idx], pb[idx], sb[idx])
        boot[i] = [ma[m] - mb[m] for m in METRICS]
    perm = np.empty((n_perm, len(METRICS)))
    for i in range(n_perm):
        swap = rng.random(len(groups)) < 0.5
        sw = swap[pat]
        qa, qb = np.where(sw, pb, pa), np.where(sw, pa, pb)
        ta, tb = np.where(sw, sb, sa), np.where(sw, sa, sb)
        ma, mb = metrics(y, qa, ta), metrics(y, qb, tb)
        perm[i] = [ma[m] - mb[m] for m in METRICS]
    p_perm = (1 + np.sum(np.abs(perm) >= np.abs(obs) - 1e-12, axis=0)) / (1 + n_perm)

    # McNemar on per-video correctness: exact grade, and the referral decision
    mc = {"acc": mcnemar(pa == y, pb == y),
          "rr": mcnemar((pa >= 2) == (y >= 2), (pb >= 2) == (y >= 2))}

    for j, m in enumerate(METRICS):
        x = per_split[m].to_numpy()
        better = -x if m in LOWER_BETTER else x
        t_p = stats.ttest_1samp(x, 0).pvalue if np.std(x) > 0 else 1.0
        w_p = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
        n01, n10, mc_p = mc.get(m, (np.nan, np.nan, np.nan))
        rows.append({
            "model_a": a, "model_b": b, "metric": m, "n_videos": len(d), "n_patients": len(groups),
            "a_pooled": obs_a[m], "b_pooled": obs_b[m], "diff_pooled": obs[j],
            "ci_low": np.percentile(boot[:, j], 2.5), "ci_high": np.percentile(boot[:, j], 97.5),
            "p_perm": p_perm[j],
            "diff_split_mean": x.mean(), "diff_split_se": x.std(ddof=1) / np.sqrt(len(x)),
            "a_wins": int(np.sum(better > 0)), "ties": int(np.sum(x == 0)), "n_splits": len(x),
            "p_ttest": t_p, "p_wilcoxon": w_p,
            "mcnemar_a_only": n01, "mcnemar_b_only": n10, "p_mcnemar": mc_p,
        })
    return rows


def main(args):
    df, models = load(args.input_dir)
    pairs = [tuple(pr.split(":")) for pr in args.pairs] if args.pairs else None
    models = list(dict.fromkeys(m for pr in pairs for m in pr)) if pairs else (args.models or models)
    out_dir = os.path.join(args.input_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for a, b in pairs or itertools.combinations(models, 2):
        print(f"{a} vs {b}")
        rows += compare(df, a, b, args.n_boot, args.n_perm, rng)
    res = pd.DataFrame(rows)
    for col in ["p_perm", "p_ttest", "p_wilcoxon", "p_mcnemar"]:
        res[col + "_holm"] = res.groupby("metric")[col].transform(
            lambda p: pd.Series(holm(p.fillna(1)), index=p.index).where(p.notna()))
    res.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)

    summary = res[["model_a", "model_b", "metric", "n_videos", "a_pooled", "b_pooled", "diff_pooled", "ci_low", "ci_high",
                   "p_perm", "p_perm_holm", "a_wins", "n_splits"]].copy()
    summary["significant"] = summary["p_perm_holm"] < args.alpha
    summary.round(4).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    md = summary_markdown(df, models, res, args)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    print(md)


METRIC_NAMES = {"qwk": "QWK", "acc": "Accuracy", "f1_macro": "Macro F1", "rr": "Referable recall", "spec": "Referral specificity",
                "auc": "Referable AUC", "off2": "≥ 2 grades off"}


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def fmt_p(p):
    return "–" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def summary_markdown(df, models, res, args):
    sig = res[res.p_perm_holm < args.alpha]
    lines = [f"# Paired model comparisons", "",
             f"`{args.input_dir}`: {len(df)} videos, {df['patient'].nunique()} patients, {df['split'].nunique()} splits.", "",
             f"- Δ = A − B on the pooled videos both models scored; 95% CI from {args.n_boot} patient-clustered bootstraps.",
             f"- p: paired permutation test ({args.n_perm} per-patient swaps of A/B); Holm over the {res.groupby('metric').size().iloc[0]} "
             f"pairs per metric. **Bold** = Holm p < {args.alpha}.",
             "- Wins: splits where A is better (out of 5). Referral = predicted grade ≥ 2; AUC on P(grade ≥ 2). "
             "≥ 2 grades off: lower is better.",
             "- McNemar (accuracy / referral correctness, ignores patient clustering), split-level t / Wilcoxon: `paired_tests.csv`.", "",
             "## Pooled scores per model", ""]
    rows = []
    for mod in models:
        d = df.dropna(subset=[f"{mod}_pref"])
        sc = metrics(d["grade"].to_numpy(int), d[f"{mod}_pred_grade"].to_numpy(int), d[f"{mod}_pref"].to_numpy())
        rows.append([mod, str(len(d))] + [f"{sc[m]:.3f}" for m in METRICS])
    lines += [md_table(["Model", "n"] + [METRIC_NAMES[m] for m in METRICS], rows), "",
              f"## Significant after Holm (p < {args.alpha})", ""]
    if len(sig):
        lines += [f"- {METRIC_NAMES[r.metric]}: {r.model_a} vs {r.model_b} Δ {r.diff_pooled:+.3f} "
                  f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}], p = {fmt_p(r.p_perm_holm)}, {r.a_wins}/{r.n_splits} wins" for r in sig.itertuples()]
    else:
        lines.append("- none")
    for m in METRICS:
        lines += ["", f"## {METRIC_NAMES[m]}", ""]
        rows = []
        for r in res[res.metric == m].itertuples():
            cells = [f"{r.model_a} vs {r.model_b}", str(r.n_videos), f"{r.a_pooled:.3f}", f"{r.b_pooled:.3f}", f"{r.diff_pooled:+.3f}",
                     f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]", fmt_p(r.p_perm), fmt_p(r.p_perm_holm), f"{r.a_wins}/{r.n_splits}"]
            rows.append([f"**{c}**" for c in cells] if r.p_perm_holm < args.alpha else cells)
        lines.append(md_table(["A vs B", "n", "A", "B", "Δ", "95% CI", "p", "p (Holm)", "Wins A"], rows))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--models", nargs="+", help="Subset/order of models (default: all in the CSVs).")
    parser.add_argument("--pairs", nargs="+", metavar="A:B", help="Only these comparisons (Holm over them), instead of all model pairs.")
    parser.add_argument("--out_name", default="paired_tests", help="Output subdirectory of --input_dir.")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for the summary (Holm-corrected).")
    main(parser.parse_args())
