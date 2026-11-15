import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="Input CSV files")
parser.add_argument("--outdir", default="label_plots")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

for file in args.files:
    path = Path(file)
    df = pd.read_csv(path)

    # Calculate label distributions
    counts = pd.crosstab([df["fold"], df["split"]], df["label"])
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    print(f"\n=== {path.name} ===")
    print("\nLabel counts:")
    print(counts)
    print("\nLabel percentages:")
    print(percentages.round(2))

    # Generate separate plots
    for data, suffix, title, ylabel in [
        (counts, "counts", "Sample Counts", "Number of samples"),
        (percentages, "percentages", "Label Percentages", "Percentage (%)")
    ]:
        ax = data.plot(kind="bar", figsize=(10, 5), width=0.8)

        ax.set_title(f"{path.stem} - {title}")
        ax.set_xlabel("Fold / Split")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(
            [f"Fold {fold}\n{split}" for fold, split in data.index],
            rotation=0
        )
        ax.legend(title="Label")
        ax.grid(axis="y", alpha=0.3)

        if suffix == "percentages":
            ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(outdir / f"{path.stem}_{suffix}.png", dpi=300)
        plt.show()
        plt.close()

print(f"\nPlots saved to: {outdir}")
