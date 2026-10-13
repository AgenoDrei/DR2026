import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from tqdm import tqdm


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def cluster_video(emb_path, out_dir, k, seed):
    """Spherical k-means (cosine distance) over one video's frame embeddings; writes cluster members + medoids."""
    emb = l2_normalize(np.load(emb_path).astype(np.float64))
    n = emb.shape[0]
    if n < k:
        raise ValueError(f"{emb_path.stem}: {n} frames < k={k}")

    # k-means on unit vectors (Euclidean on the sphere is monotone in cosine distance),
    # then assign by cosine similarity to the re-normalized centroids.
    with threadpool_limits(1):  # parallelism comes from the process pool
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
    centroids = l2_normalize(km.cluster_centers_)
    sims = emb @ centroids.T
    assign = sims.argmax(axis=1)

    clusters, medoids = [], []
    for c in range(k):
        members = np.flatnonzero(assign == c)
        if len(members) == 0:
            continue
        clusters.append(members.tolist())
        medoids.append(int(members[sims[members, c].argmax()]))

    order = np.argsort(medoids)
    result = {
        "num_frames": n,
        "k": len(clusters),
        "clusters": [clusters[i] for i in order],
        "medoids": [medoids[i] for i in order],
    }
    (out_dir / f"{emb_path.stem}.json").write_text(json.dumps(result))
    return emb_path.stem, n, [len(c) for c in clusters]


def main():
    parser = argparse.ArgumentParser(description="Cluster each video's contrastive frame embeddings into k groups (cosine k-means).")
    parser.add_argument('--embeddings_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/contrastive/embeddings")
    parser.add_argument('--out_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/clusters")
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_paths = sorted(Path(args.embeddings_dir).glob("*.npy"))

    fn = partial(cluster_video, out_dir=out_dir, k=args.k, seed=args.seed)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(fn, emb_paths), total=len(emb_paths), desc="Clustering"))

    sizes = np.concatenate([s for _, _, s in results])
    short = [stem for stem, _, s in results if len(s) < args.k]
    print(f"Clustered {len(results)} videos into {out_dir}")
    print(f"Cluster sizes: min={sizes.min()} median={np.median(sizes):.0f} max={sizes.max()}")
    if short:
        print(f"WARNING: {len(short)} videos ended up with fewer than {args.k} non-empty clusters: {short}")


if __name__ == "__main__":
    main()
