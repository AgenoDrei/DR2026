import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold


def load_dataset(name, matching_path, clip_labels_path):
    """Load one dataset (matches + clip_labels) and return synced,
    labeled arrays ready for StratifiedKFold, plus the raw dicts."""
    with open(matching_path, "r") as f:
        data = json.load(f)
        matches = data.get("matches", data)

    with open(clip_labels_path, "r") as f:
        clip_labels = json.load(f)

    labeled_clip_ids = []
    stratify_labels = []

    for cid in matches.keys():
        clip_filename = f"{cid}.mp4"
        if clip_filename in clip_labels:
            labeled_clip_ids.append(cid)
            stratify_labels.append(clip_labels[clip_filename])

    return {
        "name": name,
        "matches": matches,
        "clip_labels": clip_labels,
        "ids": np.array(labeled_clip_ids),
        "labels": np.array(stratify_labels),
    }


def create_kfold_splits(
    dataset_configs,
    n_splits=5,
    output_dir="data/stratified_splits",
    random_seed=42,
):
    """
    dataset_configs: list of dicts, e.g.
        [
            {"name": "2020", "matching_path": "data/matching_results_2020.json",
             "clip_labels_path": "data/clip_labels_2020.json"},
            {"name": "2025", "matching_path": "data/matching_results_2025.json",
             "clip_labels_path": "data/clip_labels_2025.json"},
        ]

    Each dataset is stratified-K-folded independently (by grade), then
    fold i of every dataset is merged together into the final fold i.
    This keeps both the grade distribution AND the dataset-origin ratio
    stable across folds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load every dataset
    datasets = [
        load_dataset(cfg["name"], cfg["matching_path"], cfg["clip_labels_path"])
        for cfg in dataset_configs
    ]

    # 2. Compute independent StratifiedKFold splits per dataset
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    per_dataset_folds = []  # list (per dataset) of list (per fold) of (train_idx, val_idx)
    for ds in datasets:
        folds = list(skf.split(ds["ids"], ds["labels"]))
        per_dataset_folds.append(folds)

    stats_log = []

    print(f"--- Starting {n_splits}-Fold Stratified Split across {len(datasets)} dataset(s) ---")

    for fold in range(n_splits):
        fold_num = fold + 1
        fold_dir = os.path.join(output_dir, f"split_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        file_keys = ["mil_train.json", "mil_val.json", "frame_train.json", "frame_val.json"]
        structures = {key: {} for key in file_keys}  # merged, across all datasets
        per_dataset_structures = {
            ds["name"]: {key: {} for key in file_keys} for ds in datasets
        }  # kept separate too, e.g. frame_train_2020.json

        # Helper: populate clip/image dicts and return grades. No dataset
        # prefixing here — confirmed there are no id clashes between years.
        def process_subset(ds, ids, clip_dict, image_dict):
            grades = []
            for cid in ids:
                filename = f"{cid}.mp4"
                grade = ds["clip_labels"][filename]

                if filename in clip_dict:
                    raise ValueError(
                        f"Duplicate clip id across datasets: {filename}. "
                        "Expected no clashes between 2020/2025 — check your data."
                    )

                grades.append(grade)
                clip_dict[filename] = grade
                if cid in ds["matches"]:
                    for img_name in ds["matches"][cid]:
                        image_dict[img_name] = grade
            return grades

        # Track grades per dataset too, so we can verify/report balance
        per_dataset_counts = {"train": {}, "val": {}}
        train_grades_all, val_grades_all = [], []

        for ds, folds in zip(datasets, per_dataset_folds):
            train_idx, val_idx = folds[fold]
            train_ids, val_ids = ds["ids"][train_idx], ds["ids"][val_idx]
            ds_structs = per_dataset_structures[ds["name"]]

            train_grades = process_subset(ds, train_ids, ds_structs["mil_train.json"], ds_structs["frame_train.json"])
            val_grades = process_subset(ds, val_ids, ds_structs["mil_val.json"], ds_structs["frame_val.json"])

            # fold this dataset's contribution into the merged structures too
            for key in file_keys:
                structures[key].update(ds_structs[key])

            per_dataset_counts["train"][ds["name"]] = len(train_grades)
            per_dataset_counts["val"][ds["name"]] = len(val_grades)

            train_grades_all.extend(train_grades)
            val_grades_all.extend(val_grades)

        # Save the 4 merged JSON files for this fold
        for filename, content in structures.items():
            with open(os.path.join(fold_dir, filename), "w") as f:
                json.dump(content, f, indent=4)

        # Save the per-dataset JSON files too, e.g. frame_train_2020.json
        for ds_name, ds_structs in per_dataset_structures.items():
            for filename, content in ds_structs.items():
                base, ext = filename.rsplit(".", 1)
                out_filename = f"{base}_{ds_name}.{ext}"
                with open(os.path.join(fold_dir, out_filename), "w") as f:
                    json.dump(content, f, indent=4)

        # Collect metadata/stats (grade dist + binary dist + per-dataset counts)
        for name, g_list in [("train", train_grades_all), ("val", val_grades_all)]:
            counts = Counter(g_list)
            binary_counts = Counter([1 if g >= 2 else 0 for g in g_list])

            entry = {
                "fold": fold_num,
                "set": name,
                "total": len(g_list),
                **{f"grade_{i}": counts.get(i, 0) for i in range(5)},
                "binary_0": binary_counts.get(0, 0),
                "binary_1": binary_counts.get(1, 0),
            }
            for ds_name, cnt in per_dataset_counts[name].items():
                entry[f"n_{ds_name}"] = cnt

            stats_log.append(entry)

    # 3. Save metadata CSV
    df_stats = pd.DataFrame(stats_log)
    df_stats.to_csv(os.path.join(output_dir, "split_metadata.csv"), index=False)

    # 4. Generate distribution plots
    dataset_names = [ds["name"] for ds in datasets]
    generate_plots(df_stats, output_dir, dataset_names)

    print(f"Success! Splits and metadata saved to: {output_dir}")
    return df_stats


def generate_plots(df, output_dir, dataset_names):
    df_train = df[df['set'] == 'train']

    n_panels = 3 if len(dataset_names) > 1 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 6))

    # Plot 1: 0-4 Grades (train sets only, for clarity)
    grade_cols = [f"grade_{i}" for i in range(5)]
    df_train.set_index('fold')[grade_cols].plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title("Grade Distribution (0-4) per Fold (Train)")
    axes[0].set_ylabel("Count")

    # Plot 2: Binary Distribution (train sets)
    df_train.set_index('fold')[['binary_0', 'binary_1']].plot(kind='bar', ax=axes[1])
    axes[1].set_title("Binary Distribution (Referable vs Not) per Fold")
    axes[1].set_ylabel("Count")

    # Plot 3: Per-dataset composition per fold (train sets), if multiple datasets
    if len(dataset_names) > 1:
        ds_cols = [f"n_{name}" for name in dataset_names]
        df_train.set_index('fold')[ds_cols].plot(kind='bar', stacked=True, ax=axes[2])
        axes[2].set_title("Dataset Origin per Fold (Train)")
        axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_plot.png"))
    plt.close()


if __name__ == '__main__':
    create_kfold_splits(
        dataset_configs=[
            {
                "name": "2025",
                "matching_path": "data/matching_results_2025.json",
                "clip_labels_path": "data/clip_labels_2025.json",
            },
            {
                "name": "2020",
                "matching_path": "data/matching_results_2020.json",
                "clip_labels_path": "data/clip_labels_2020.json",
            },
        ],
        n_splits=5,
        output_dir="data/stratified_splits",
        random_seed=42,
    )