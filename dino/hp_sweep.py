"""
Hyperparameter search for dino.train_dino_transformer, on a single split
(fold0 = split_1 by default).

Fixed: epochs=6, lr_step=3, lr=5e-5, complex_augs + random_segment_sample on
(matches the current best-run setup), 5-class, Paxos2025-only.

Search space:
  num_heads            [2, 4, 8]
  freeze_backbone       [True, False]
  backbone_lr_mult      [0.1, 0.3]   (only sampled/used when freeze_backbone=False)
  distance_scale_weight [0.2, 0.5, 1.0]
  attn_entropy_weight   [0.0, 0.1, 0.2]
  num_blocks            [1, 2]

162 unique effective configs (backbone_lr_mult collapses to a single N/A
case when freeze_backbone=True, so it isn't double-counted). Trials are
drawn without replacement from the full enumerated grid (shuffled once
with --sample_seed), so no config repeats - requesting --n_trials >= 162
runs the full grid exactly once.

Every trial trains with the same --train_seed, so differences between
trials reflect the hyperparameters, not random weight-init/sampler noise.

At ~35-40 min/epoch, each 6-epoch trial takes ~3.5-4h, so this supports
sharding across GPUs: launch one process per free GPU with the same
--n_trials/--sample_seed and a distinct --shard_index, e.g.

  gpu run python -u -m dino.hp_sweep --num_shards 5 --shard_index 0   # worker0
  gpu run python -u -m dino.hp_sweep --num_shards 5 --shard_index 1   # worker1
  ...

All shards enumerate the identical shuffled trial list, each runs only
the trials where trial_idx % num_shards == shard_index, and writes its own
CSV (results merge cleanly afterward, no write contention).
"""
import argparse
import csv
import itertools
import os
import random
import traceback

import torch

from dino.train_dino_transformer import train_self_attention, build_parser


SEARCH_SPACE = {
    "num_heads": [2, 4, 8],
    "freeze_backbone": [True, False],
    "backbone_lr_mult": [0.1, 0.3],
    "distance_scale_weight": [0.2, 0.5, 1.0],
    "attn_entropy_weight": [0.0, 0.1, 0.2],
    "num_blocks": [1, 2],
}


def enumerate_all_configs():
    """All unique effective configs. backbone_lr_mult is only varied when
    freeze_backbone=False (it has no effect on a frozen backbone), so the
    frozen case contributes one config per other-dim combination instead of
    len(backbone_lr_mult) duplicates of it."""
    configs = []
    for num_heads, freeze_backbone, distance_scale_weight, attn_entropy_weight, num_blocks in itertools.product(
        SEARCH_SPACE["num_heads"], SEARCH_SPACE["freeze_backbone"],
        SEARCH_SPACE["distance_scale_weight"], SEARCH_SPACE["attn_entropy_weight"],
        SEARCH_SPACE["num_blocks"],
    ):
        base = {
            "num_heads": num_heads, "freeze_backbone": freeze_backbone,
            "distance_scale_weight": distance_scale_weight,
            "attn_entropy_weight": attn_entropy_weight, "num_blocks": num_blocks,
        }
        if freeze_backbone:
            configs.append({**base, "backbone_lr_mult": None})
        else:
            for backbone_lr_mult in SEARCH_SPACE["backbone_lr_mult"]:
                configs.append({**base, "backbone_lr_mult": backbone_lr_mult})
    return configs


def build_args(trial, base):
    """train_dino_transformer's own parser defaults + this sweep's fixed setup + the trial's HPs."""
    args = build_parser().parse_args([
        "--split_path", base.split_path,
        "--seed", str(base.train_seed),
        "--epochs", "6", "--lr_step", "3", "--lr", "5e-5",
        "--complex_augs", "--random_segment_sample",
        "--num_blocks", str(trial["num_blocks"]),
        "--num_heads", str(trial["num_heads"]),
        "--attn_entropy_weight", str(trial["attn_entropy_weight"]),
        "--distance_scale_weight", str(trial["distance_scale_weight"]),
    ])
    args.classifier_checkpoint = base.classifier_checkpoint
    args.freeze_backbone = trial["freeze_backbone"]
    args.backbone_lr_mult = trial["backbone_lr_mult"] if trial["backbone_lr_mult"] is not None else 1.0
    return args


def main(args):
    all_configs = enumerate_all_configs()
    random.Random(args.sample_seed).shuffle(all_configs)

    n_trials = args.n_trials
    if n_trials >= len(all_configs):
        if n_trials > len(all_configs):
            print(f"--n_trials {n_trials} >= {len(all_configs)} unique configs; running the full grid.")
        n_trials = len(all_configs)
    all_trials = all_configs[:n_trials]  # sampled without replacement - no repeats

    my_trials = [
        (i, t) for i, t in enumerate(all_trials) if i % args.num_shards == args.shard_index
    ]

    print(f"Shard {args.shard_index}/{args.num_shards}: running {len(my_trials)} of {n_trials} trials "
          f"(train_seed={args.train_seed}, out of {len(all_configs)} unique configs).")

    # Headline metrics captured alongside QWK, read straight off the
    # best-QWK-epoch metrics dict train_self_attention already returns (no
    # separate TensorBoard read-back needed). Per-class breakdowns still
    # only live in that run's own TensorBoard log if you need them.
    metric_fields = ["best_qwk", "acc", "f1_macro", "referable_recall", "roc_auc_macro"]
    fieldnames = [
        "trial_idx", "num_heads", "freeze_backbone", "backbone_lr_mult",
        "distance_scale_weight", "attn_entropy_weight", "num_blocks",
        *metric_fields, "run_name", "status",
    ]
    out_path = args.output_csv.format(shard_index=args.shard_index)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_header = not os.path.exists(out_path)

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for trial_idx, trial in my_trials:
            print(f"\n=== Trial {trial_idx}: {trial} ===")
            row = {"trial_idx": trial_idx, **trial}

            try:
                run_name, best_qwk, best_metrics = train_self_attention(build_args(trial, args))
                row["best_qwk"] = round(best_qwk, 4)
                row["acc"] = round(best_metrics["acc"], 4)
                row["f1_macro"] = round(best_metrics["f1_macro"], 4)
                row["referable_recall"] = round(best_metrics["referable_recall"], 4)
                row["roc_auc_macro"] = round(best_metrics["roc_auc_macro"], 4)
                row["run_name"] = run_name
                row["status"] = "ok"
            except Exception:
                print(f"Trial {trial_idx} FAILED:")
                traceback.print_exc()
                for field in metric_fields:
                    row[field] = ""
                row["run_name"] = ""
                row["status"] = "failed"
            finally:
                torch.cuda.empty_cache()

            writer.writerow(row)
            f.flush()

    print(f"\nShard {args.shard_index} done. Results: {out_path}")
    completed = [r for r in csv.DictReader(open(out_path)) if r["status"] == "ok"]
    completed.sort(key=lambda r: float(r["best_qwk"]), reverse=True)
    print("\n--- Leaderboard (this shard) ---")
    for r in completed[:10]:
        print(
            f"QWK {r['best_qwk']} acc={r['acc']} f1_macro={r['f1_macro']} "
            f"ref_recall={r['referable_recall']} roc_auc={r['roc_auc_macro']} | "
            f"heads={r['num_heads']} frozen={r['freeze_backbone']} "
            f"bb_lr_mult={r['backbone_lr_mult']} dist_w={r['distance_scale_weight']} "
            f"ent_w={r['attn_entropy_weight']} blocks={r['num_blocks']} | {r['run_name']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_checkpoint", type=str,
                        default="classifier/models/dino_all_split_1_1002_1543_full/best_model.pth")
    parser.add_argument("--split_path", type=str, default="data/stratified_splits_paxos2025/split_1",
                        help="fold0")
    parser.add_argument("--n_trials", type=int, default=15,
                        help="Configs to run, drawn without replacement (capped at the 108-config full grid).")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Seeds which configs get drawn/shuffled (shared across shards).")
    parser.add_argument("--train_seed", type=int, default=42,
                        help="Passed to every trial's --seed, so differences reflect the HPs, not training noise.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="dino/hp_sweep_results_shard{shard_index}.csv")

    args = parser.parse_args()
    main(args)
