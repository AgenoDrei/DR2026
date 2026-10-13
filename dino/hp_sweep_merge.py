"""
Merge the per-shard CSVs written by dino.hp_sweep (one per GPU/shard) into a
single sorted results table.

  python -u -m dino.hp_sweep_merge --pattern "dino/hp_sweep_results_shard*.csv"
"""
import argparse
import csv
import glob


def main(args):
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"No files matched: {args.pattern}")
        return

    rows = []
    seen = set()  # (trial_idx, run_name) - guards against a shard re-run appending dupes
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["trial_idx"], row["run_name"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    ok.sort(key=lambda r: float(r["best_qwk"]), reverse=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(ok + failed)

    print(f"Merged {len(paths)} file(s): {len(ok)} ok, {len(failed)} failed -> {args.output_csv}")
    print("\n--- Leaderboard (all shards) ---")
    for r in ok[:args.top_n]:
        print(
            f"QWK {r['best_qwk']} acc={r['acc']} f1_macro={r['f1_macro']} "
            f"ref_recall={r['referable_recall']} roc_auc={r['roc_auc_macro']} | "
            f"heads={r['num_heads']} frozen={r['freeze_backbone']} "
            f"bb_lr_mult={r['backbone_lr_mult']} dist_w={r['distance_scale_weight']} "
            f"ent_w={r['attn_entropy_weight']} blocks={r['num_blocks']} | {r['run_name']}"
        )

    if failed:
        print(f"\n{len(failed)} failed trial(s):")
        for r in failed:
            print(f"  trial {r['trial_idx']}: heads={r['num_heads']} frozen={r['freeze_backbone']} "
                  f"bb_lr_mult={r['backbone_lr_mult']} dist_w={r['distance_scale_weight']} "
                  f"ent_w={r['attn_entropy_weight']} blocks={r['num_blocks']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="dino/hp_sweep_results_shard*.csv")
    parser.add_argument("--output_csv", type=str, default="dino/hp_sweep_results_merged.csv")
    parser.add_argument("--top_n", type=int, default=20)

    args = parser.parse_args()
    main(args)
