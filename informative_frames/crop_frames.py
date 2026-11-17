"""Crop the raw Paxos2025 informative frames with the 5-fold cropping models.

Every labeled frame (from the split CSV built by cropping/make_splits_from_q5v3.py,
rows with split == "val") is cropped and written to <output_dir>/<label>/<filename>
(ImageFolder layout with classes 0/1, so classify_frames.py can consume the folder directly).

Each frame is cropped by the fold-i segmentation model of its own val fold
(cropping/models/seg_fold_<i>.pth) -- the model that never saw the frame during training,
exactly the anti-leakage rule of cropping/evaluate_segmentation.py.

Pre-processing (same as cropping/segment_frames.py and informative_frames/extract_inf_frames.py):
the raw 1080x1920 frame is padded to a square (pad_to_square_info), the model predicts at
768x768, the probability mask is mapped back through the padding offsets, the largest
connected component is kept, and a square crop is taken around its bounding box (crop_and_pad).

Run from the repo root:
    python -u -m informative_frames.crop_frames \
      --input_dir data/informative_frames/frames_raw_extract \
      --split_csv informative_frames/split_info_5fold_paxos2025_from_q5v3.csv \
      --model_dir cropping/models \
      --output_dir data/informative_frames/frames_cropped
"""

import argparse
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=str, default="data/informative_frames/frames_raw_extract",
                        help="Raw Paxos2025 frame folder.")
    parser.add_argument("--split_csv", type=str, default="informative_frames/split_info_5fold_paxos2025_from_q5v3.csv",
                        help="Split CSV (filename,video_id,label,fold,split) with label + val fold per frame.")
    parser.add_argument("--model_dir", type=str, default="cropping/models",
                        help="Checkpoint dir with seg_fold_<i>.pth.")
    parser.add_argument("--output_dir", type=str, default="data/informative_frames/frames_cropped",
                        help="Output folder; cropped frames land in <output_dir>/<label>/<filename>.")
    parser.add_argument("--img_size", type=int, default=768, help="Model input size (as trained).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    return parser.parse_args()


def get_inference_transforms(img_size=768):
    return albu.Compose([
        albu.Resize(img_size, img_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def load_fold_models(model_dir, n_splits, encoder_name, device):
    model_dir = Path(model_dir)
    models = []
    for fold in range(n_splits):
        model_paths = sorted(model_dir.glob(f"seg_fold_{fold}*.pth"))
        if not model_paths:
            model_paths = sorted(model_dir.glob(f"*fold_{fold}*"))
        if not model_paths:
            raise FileNotFoundError(f"Could not find model for fold {fold} in {model_dir} "
                                    f"(tried seg_fold_{fold}*.pth and *fold_{fold}*)")
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(torch.load(model_paths[0], map_location=device))
        model.to(device).eval()
        models.append(model)
    return models


def pad_to_square_info(image):
    h, w = image.shape[:2]
    side = max(h, w)
    pad_img = np.zeros((side, side, 3), dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    pad_img[y_off:y_off + h, x_off:x_off + w] = image
    return pad_img, y_off, x_off, side


def get_largest_component_mask(binary_mask):
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
    if num_labels <= 1:
        return None, None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = (labels == largest).astype(np.uint8)
    x, y, w, h = stats[largest, :4]
    H, W = clean_mask.shape
    box = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
    return clean_mask, box


def crop_and_pad(frame, cx, cy, size):
    h, w = frame.shape[:2]
    half_size = int(size // 2)
    x1, y1 = int(cx - half_size), int(cy - half_size)
    x2, y2 = x1 + size, y1 + size
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    src_x1, src_x2 = max(0, x1), min(w, x2)
    src_y1, src_y2 = max(0, y1), min(h, y2)
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
    return canvas


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    val_df = df[df["split"] == "val"]
    frame_label = dict(zip(val_df["filename"], val_df["label"].astype(int)))
    frame_fold = dict(zip(val_df["filename"], val_df["fold"].astype(int)))

    on_disk = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    on_disk_names = {p.name for p in on_disk}
    unlabeled = sorted(on_disk_names - set(frame_label))
    missing = sorted(set(frame_label) - on_disk_names)
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} frames in {input_dir} have no label in {args.split_csv} -- skipped, "
              f"e.g. {unlabeled[:3]}")
    if missing:
        print(f"WARNING: {len(missing)} labeled frames not found in {input_dir} -- skipped, "
              f"e.g. {missing[:3]}")
    files = [p for p in on_disk if p.name in frame_label]
    print(f"cropping {len(files)} frames from {input_dir} into {output_dir}")

    models = load_fold_models(args.model_dir, args.n_splits, args.encoder_name, device)
    print(f"Loaded {len(models)} fold models from {args.model_dir} on {device}")
    print("each frame is cropped by the fold model of its val fold (never trained on it)")

    transforms = get_inference_transforms(args.img_size)
    for label in sorted({frame_label[p.name] for p in files}):
        (output_dir / str(label)).mkdir(parents=True, exist_ok=True)

    total_written = total_skipped = 0
    for fold in range(args.n_splits):
        group_files = [p for p in files if frame_fold[p.name] == fold]
        if not group_files:
            continue
        model = models[fold]
        written = skipped = 0
        for start in tqdm(range(0, len(group_files), args.batch_size), desc=f"fold_{fold}", leave=False):
            batch_paths = group_files[start:start + args.batch_size]
            batch = []
            for p in batch_paths:
                out_path = output_dir / str(frame_label[p.name]) / p.name
                if out_path.exists():
                    skipped += 1
                    continue
                frame = cv2.imread(str(p))
                if frame is None:
                    print(f"WARNING: cannot read {p} -- skipped")
                    skipped += 1
                    continue
                sq, y_off, x_off, side = pad_to_square_info(frame)
                tensor = transforms(image=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))["image"]
                batch.append((p, frame, y_off, x_off, side, tensor, out_path))
            if not batch:
                continue
            x = torch.stack([b[5] for b in batch]).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(x))[:, 0]
            probs = probs.float().cpu().numpy()
            for (p, frame, y_off, x_off, side, _, out_path), prob in zip(batch, probs):
                h, w = frame.shape[:2]
                full_mask = cv2.resize(prob, (side, side))[y_off:y_off + h, x_off:x_off + w]
                clean_mask, box = get_largest_component_mask((full_mask > 0.5).astype(np.uint8))
                if clean_mask is None:
                    print(f"WARNING: no retina detected for {p.name} -- skipped")
                    skipped += 1
                    continue
                x1, y1, x2, y2 = box
                masked = cv2.bitwise_and(frame, frame, mask=clean_mask)
                cropped = crop_and_pad(masked, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1))
                cv2.imwrite(str(out_path), cropped)
                written += 1
        print(f"  fold_{fold}: wrote {written}, skipped {skipped} (existing/unreadable/no-retina)")
        total_written += written
        total_skipped += skipped

    print(f"Done: {total_written} cropped frames in {output_dir} ({total_skipped} skipped)")


if __name__ == "__main__":
    main()
