import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dino.utils import DinoBackbone, make_val_transform_dino


def iter_frame_batches(video_path, transform, batch_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    cap.release()
    if batch:
        yield torch.stack(batch)


@torch.inference_mode()
def extract_video_features(model, video_path, transform, batch_size, device):
    feats = []
    for batch in iter_frame_batches(video_path, transform, batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats.append(model(batch.to(device, non_blocking=True)).float().cpu())
    return torch.cat(feats).numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Extract pretrained DINOv3 CLS features for every frame of the cleaned videos.")
    parser.add_argument('--video_dir', type=str, default="data/ensemble_results_paxos2025/cleaned_videos")
    parser.add_argument('--out_dir', type=str, default="data/ensemble_results_paxos2025/frame_clusters/features")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=12, help="CPU threads for decoding/transforms (avoid oversubscription when sharding).")
    parser.add_argument('--force', action='store_true', help="Recompute features that already exist.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    cv2.setNumThreads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(Path(args.video_dir).glob("*.mp4")) + list(Path(args.video_dir).glob("*.MOV")))
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_index]
    todo = [v for v in videos if args.force or not (out_dir / f"{v.stem}.npy").exists()]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(videos)} videos, {len(todo)} to process")

    model = DinoBackbone(freeze=True).to(device).eval()
    transform = make_val_transform_dino(args.img_size, complex_augs=False)

    for video_path in tqdm(todo, desc="Extracting features"):
        feats = extract_video_features(model, video_path, transform, args.batch_size, device)
        cap = cv2.VideoCapture(str(video_path))
        n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if feats.shape[0] != n_meta:
            print(f"WARNING: {video_path.name}: decoded {feats.shape[0]} frames, metadata says {n_meta}")
        np.save(out_dir / f"{video_path.stem}.npy", feats)


if __name__ == "__main__":
    main()
