import argparse
import json
import random
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

class ContrastiveProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_hidden_layers):
        super().__init__()
        blocks = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            blocks += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        blocks.append(nn.Linear(in_dim, embedding_dim))
        self.mlp = nn.Sequential(*blocks)

    def forward(self, x):
        return F.normalize(self.mlp(F.normalize(x, dim=-1)), dim=-1)


def load_features(features_dir):
    return {p.stem: np.load(p).astype(np.float32) for p in sorted(Path(features_dir).glob("*.npy"))}


def sample_pair_for_anchor(n, anchor, pos_window, neg_min_distance, num_negatives, rng):
    """Positive: random frame within pos_window (in cleaned-video frame indices); negatives: frames >= neg_min_distance away."""
    lo, hi = max(0, anchor - pos_window), min(n - 1, anchor + pos_window)
    n_pos = hi - lo  # candidates in [lo, hi] excluding the anchor itself
    left_end = anchor - neg_min_distance + 1   # negatives on the left: [0, left_end)
    right_start = anchor + neg_min_distance    # negatives on the right: [right_start, n)
    n_left, n_right = max(0, left_end), max(0, n - right_start)
    if n_pos == 0 or n_left + n_right == 0:
        return None

    pos = lo + int(rng.integers(n_pos))
    if pos >= anchor:
        pos += 1

    neg = rng.integers(n_left + n_right, size=num_negatives)
    neg = np.where(neg < n_left, neg, neg - n_left + right_start)
    return pos, neg


def build_batch(video_features, anchor_chunk, args, rng, device):
    anchors, positives, negatives = [], [], []
    n_skipped = 0
    for stem, anchor in anchor_chunk:
        arr = video_features[stem]
        pair = sample_pair_for_anchor(arr.shape[0], anchor, args.pos_window, args.neg_min_distance, args.num_negatives, rng)
        if pair is None:
            n_skipped += 1
            continue
        pos, neg = pair
        anchors.append(arr[anchor])
        positives.append(arr[pos])
        negatives.append(arr[neg])

    if not anchors:
        return None, n_skipped
    batch = tuple(torch.from_numpy(np.stack(x)).to(device) for x in (anchors, positives, negatives))
    return batch, n_skipped


def info_nce_loss(anchor_emb, pos_emb, neg_emb, temperature):
    sim_pos = (anchor_emb * pos_emb).sum(-1) / temperature
    sim_neg = torch.einsum("bd,bkd->bk", anchor_emb, neg_emb) / temperature
    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def train(model, video_features, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    all_anchors = [(stem, i) for stem, arr in video_features.items() for i in range(arr.shape[0])]
    loss_history = []

    for epoch in range(args.epochs):
        order = [all_anchors[i] for i in rng.permutation(len(all_anchors))]
        epoch_losses, total_skipped = [], 0
        for i in tqdm(range(0, len(order), args.batch_size), desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch, n_skipped = build_batch(video_features, order[i:i + args.batch_size], args, rng, device)
            total_skipped += n_skipped
            if batch is None:
                continue
            anchors, positives, negatives = batch
            anchor_emb = model(anchors)
            pos_emb = model(positives)
            neg_emb = model(negatives.reshape(-1, negatives.shape[-1])).reshape(*negatives.shape[:2], -1)
            loss = info_nce_loss(anchor_emb, pos_emb, neg_emb, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = statistics.mean(epoch_losses)
        loss_history.append(mean_loss)
        print(f"Epoch {epoch + 1}/{args.epochs} mean_loss={mean_loss:.4f} skipped_anchors={total_skipped}")

    return loss_history


def save_outputs(model, video_features, args, loss_history, out_dir, device):
    embeddings_dir = out_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        for stem, arr in video_features.items():
            emb = model(torch.from_numpy(arr).to(device)).cpu().numpy().astype(np.float32)
            np.save(embeddings_dir / f"{stem}.npy", emb)

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": next(iter(video_features.values())).shape[1],
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "num_hidden_layers": args.num_hidden_layers,
    }, out_dir / "projector.pt")

    params = {k: v for k, v in vars(args).items()}
    params.update(final_loss=loss_history[-1], loss_history=loss_history, timestamp=datetime.now().isoformat())
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train a temporal contrastive projection head on per-frame DINOv3 features.")
    parser.add_argument('--features_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/features")
    parser.add_argument('--out_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/contrastive")
    parser.add_argument('--pos_window', type=int, default=5, help="Max frame distance counted as a positive pair")
    parser.add_argument('--neg_min_distance', type=int, default=30, help="Min frame distance counted as a negative pair")
    parser.add_argument('--num_negatives', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.neg_min_distance <= args.pos_window:
        parser.error("--neg_min_distance must be > --pos_window")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_features = load_features(args.features_dir)
    input_dim = next(iter(video_features.values())).shape[1]
    n_frames = sum(arr.shape[0] for arr in video_features.values())
    print(f"Training on {len(video_features)} videos / {n_frames} frames, input_dim={input_dim}, device={device}")

    model = ContrastiveProjector(input_dim, args.hidden_dim, args.embedding_dim, args.num_hidden_layers).to(device)
    loss_history = train(model, video_features, args, device)
    save_outputs(model, video_features, args, loss_history, out_dir, device)
    print(f"Wrote projector and embeddings to {out_dir}")


if __name__ == "__main__":
    main()
