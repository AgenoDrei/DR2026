import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_frames(video_path, indices):
    wanted = set(indices)
    frames = {}
    cap = cv2.VideoCapture(str(video_path))
    for i in range(max(indices) + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames


def plot_video_grid(video_path, cluster_info, mode, out_path, cols=8, tile_size=256):
    clusters = cluster_info["clusters"]
    if mode == "medoid":
        picks = list(cluster_info["medoids"])
    else:
        picks = [int(np.random.choice(members)) for members in clusters]
    order = np.argsort(picks)
    picks = [picks[i] for i in order]
    sizes = [len(clusters[i]) for i in order]

    frames = read_frames(video_path, picks)
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx, size in zip(axes.flat, picks, sizes):
        if idx in frames:
            ax.imshow(cv2.resize(frames[idx], (tile_size, tile_size)))
        ax.set_title(f"f{idx} | n={size}", fontsize=8)
    fig.suptitle(f"{video_path.stem}  ({cluster_info['num_frames']} frames, {mode})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save one image grid of the 32 cluster-sampled frames per video for random videos.")
    parser.add_argument('--video_dir', type=str, default="data/ensemble_results_paxos2025/cleaned_videos")
    parser.add_argument('--cluster_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/clusters")
    parser.add_argument('--out_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/vis")
    parser.add_argument('--num_videos', type=int, default=20)
    parser.add_argument('--mode', type=str, default="medoid", choices=["medoid", "random"],
                        help="medoid = validation sampling, random = one training draw")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_paths = random.sample(sorted(Path(args.cluster_dir).glob("*.json")), args.num_videos)
    for cluster_path in cluster_paths:
        video_path = Path(args.video_dir) / f"{cluster_path.stem}.mp4"
        cluster_info = json.loads(cluster_path.read_text())
        plot_video_grid(video_path, cluster_info, args.mode, out_dir / f"{cluster_path.stem}_{args.mode}.jpg")
        print(f"Saved {cluster_path.stem}")


if __name__ == "__main__":
    main()
