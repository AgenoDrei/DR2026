import cv2
import torch
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from misc.utils import apply_clahe_cv2
from .crop_frames import crop_and_pad, get_largest_component_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Full Ensemble Production Pipeline")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None, help="Max videos per worker.")
    parser.add_argument('--video_dir', type=str, default='data/dr_videos')
    parser.add_argument('--seg_model_dir', type=str, default='cropping/models')
    parser.add_argument('--cls_model_dir', type=str, default='informative_frames/models')
    parser.add_argument('--out_base', type=str, default='data/ensemble_results')
    parser.add_argument('--min_informative_frames', type=int, default=32,
                        help="Guaranteed minimum length of the cleaned clip.")
    parser.add_argument('--infer_batch_size', type=int, default=16,
                        help="Frames batched per ensemble forward pass in PASS 1.")
    return parser.parse_args()


def get_seg_transforms(size=512):
    return albu.Compose([
        albu.Resize(size, size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def select_informative_frames(probs, start_threshold, min_threshold, step, min_frames):
    """
    Select informative frame indices by thresholding the ensemble probability.
    If the starting threshold doesn't yield enough frames, step the threshold
    down (never below min_threshold) until min_frames is reached. If even
    min_threshold isn't enough, fall back to the top-`min_frames` frames by
    probability so the clip is never shorter than min_frames (as long as the
    video itself has that many frames).

    Returns: (sorted_indices, threshold_used, used_fallback)
    """
    probs = np.asarray(probs)
    n_total = len(probs)

    threshold = start_threshold
    idx = np.where(probs >= threshold)[0]

    while len(idx) < min_frames and threshold > min_threshold:
        threshold = round(max(threshold - step, min_threshold), 4)
        idx = np.where(probs >= threshold)[0]

    used_fallback = False
    if len(idx) < min_frames:
        n_take = min(min_frames, n_total)
        idx = np.sort(np.argsort(probs)[::-1][:n_take])
        used_fallback = True

    return idx, threshold, used_fallback


def main():
    args = parse_args()
    DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    VIDEO_DIR = Path(args.video_dir)
    SEG_MODEL_DIR = Path(args.seg_model_dir)
    CLS_MODEL_DIR = Path(args.cls_model_dir)

    OUT_BASE = Path(args.out_base)
    TXT_DIR = OUT_BASE / 'txt_files'
    PLOT_DIR = OUT_BASE / 'plots'
    VID_DIR = OUT_BASE / 'cleaned_videos'

    for d in [TXT_DIR, PLOT_DIR, VID_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    IMG_SIZE = 512
    CONF_THRESHOLD = 0.5  # starting / preferred threshold
    MIN_CONF_THRESHOLD = 0.2  # floor - never go below this
    THRESHOLD_STEP = 0.05  # how much to relax the threshold per attempt
    MIN_INFORMATIVE_FRAMES = args.min_informative_frames
    INFER_BATCH_SIZE = args.infer_batch_size

    # 1. Load Ensembles
    print(f"Loading 5-Fold Unet++ (B4) and Classification (B4) on {DEVICE}...")
    seg_ensemble = []
    cls_ensemble = []
    for fold in range(5):
        # Seg Model
        sm = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1).to(DEVICE).eval()
        sm.load_state_dict(torch.load(list(SEG_MODEL_DIR.glob(f"*fold_{fold}*.pth"))[0], map_location=DEVICE))
        seg_ensemble.append(sm)
        # Cls Model
        cm = models.efficientnet_b4()
        cm.classifier[1] = torch.nn.Linear(cm.classifier[1].in_features, 1)
        cm.load_state_dict(torch.load(list(CLS_MODEL_DIR.glob(f"*Fold{fold}*.pth"))[0], map_location=DEVICE))
        cm.to(DEVICE).eval()
        cls_ensemble.append(cm)

    seg_trans = get_seg_transforms(IMG_SIZE)
    cls_trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- File Discovery ---
    all_vids = sorted([p for p in VIDEO_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"}])
    my_vids = [v for i, v in enumerate(all_vids) if i % args.total_gpus == args.gpu_id]
    if args.limit: my_vids = my_vids[:args.limit]

    print(f"Worker {args.gpu_id}: Processing {len(my_vids)} videos.")

    for v_path in tqdm(my_vids, desc=f"Overall GPU {args.gpu_id}"):
        vid_id = v_path.stem
        out_vid_path = VID_DIR / f"CLEAN_{vid_id}.mp4"
        txt_path = TXT_DIR / f"{vid_id}.txt"
        plot_path = PLOT_DIR / f"{vid_id}.png"

        if out_vid_path.exists() and txt_path.exists() and plot_path.exists():
            print(f"Skipping {vid_id}")
            continue
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # PASS 1: Ensemble Analysis
        frame_results = []
        global_max_dim = 0

        amp_ctx = lambda: torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == 'cuda'))

        with tqdm(total=total_f, desc=f"Analysis: {vid_id}", leave=False) as pbar:
            f_idx = 0
            while True:
                # --- Read up to INFER_BATCH_SIZE raw frames ---
                batch_frames = []
                batch_f_indices = []
                for _ in range(INFER_BATCH_SIZE):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    batch_f_indices.append(f_idx)
                    f_idx += 1
                if not batch_frames:
                    break

                h, w = batch_frames[0].shape[:2]

                # --- Batched segmentation: one forward pass per ensemble member for the whole batch ---
                seg_in = torch.stack([
                    seg_trans(image=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))['image'] for fr in batch_frames
                ]).to(DEVICE)

                with torch.no_grad(), amp_ctx():
                    m_preds = torch.stack([torch.sigmoid(m(seg_in)) for m in seg_ensemble])  # [5, B, 1, H, W]
                    mean_mask_batch = torch.mean(m_preds, dim=0).squeeze(1).float().cpu().numpy()  # [B, H, W]

                # --- Per-frame CPU postprocessing (connected components can't batch) ---
                per_frame_info = [(None, None, None)] * len(batch_frames)  # (clean_mask, cx, cy)
                crops, crop_slots = [], []

                for bi, mask_small in enumerate(mean_mask_batch):
                    full_mask = cv2.resize(mask_small, (w, h))
                    clean_mask, mw, mh, cx, cy = get_largest_component_mask(full_mask > 0.5)
                    per_frame_info[bi] = (clean_mask, cx, cy)

                    if clean_mask is not None:
                        global_max_dim = max(global_max_dim, max(mw, mh))
                        # Masked square crop, same recipe as crop_frames.py (classifier training data)
                        masked = cv2.bitwise_and(batch_frames[bi], batch_frames[bi], mask=clean_mask)
                        crop = crop_and_pad(masked, cx, cy, int(max(mw, mh) * 1.05))

                        if crop.size > 0:
                            crops.append(cls_trans(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                            crop_slots.append(bi)

                # --- Batched classification on whichever frames produced a valid crop ---
                probs = [0.0] * len(batch_frames)
                if crops:
                    cls_in = torch.stack(crops).to(DEVICE)
                    with torch.no_grad(), amp_ctx():
                        c_preds = torch.stack([torch.sigmoid(m(cls_in)) for m in cls_ensemble])  # [5, Bc, 1]
                        mean_probs = torch.mean(c_preds, dim=0).squeeze(1).float().cpu().numpy()  # [Bc]
                    for slot_idx, bi in enumerate(crop_slots):
                        probs[bi] = float(mean_probs[slot_idx])

                for bi in range(len(batch_frames)):
                    clean_mask, cx, cy = per_frame_info[bi]
                    frame_results.append({
                        'f': batch_f_indices[bi], 'prob': probs[bi], 'cx': cx, 'cy': cy,
                        'mask': clean_mask, 'frame': batch_frames[bi],
                    })

                pbar.update(len(batch_frames))

        # --- Adaptive informative-frame selection ---
        # Start at CONF_THRESHOLD and relax down to MIN_CONF_THRESHOLD until we
        # have at least MIN_INFORMATIVE_FRAMES. If that's still not enough,
        # fall back to the top-N frames by probability.
        all_probs = [r['prob'] for r in frame_results]
        informative_idx, threshold_used, used_fallback = select_informative_frames(
            all_probs,
            start_threshold=CONF_THRESHOLD,
            min_threshold=MIN_CONF_THRESHOLD,
            step=THRESHOLD_STEP,
            min_frames=MIN_INFORMATIVE_FRAMES
        )
        informative_idx_set = set(int(i) for i in informative_idx)

        status = "fallback top-N" if used_fallback else f"threshold={threshold_used}"
        print(f"{vid_id}: {len(informative_idx_set)}/{len(frame_results)} informative frames ({status})")

        # PASS 2: Export Cropped Video (informative frames only)
        if global_max_dim > 0 and len(informative_idx_set) > 0:
            final_sz = int(global_max_dim * 1.05)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(out_vid_path),
                fourcc,
                fps,
                (final_sz, final_sz)
            )

            for res in frame_results:
                # Skip frames that didn't make the informative cut
                if res['f'] not in informative_idx_set:
                    continue

                # If segmentation exists, crop around it
                if res['mask'] is not None:
                    masked = cv2.bitwise_and(res['frame'], res['frame'], mask=res['mask'])
                    cropped = crop_and_pad(
                        masked,
                        res['cx'],
                        res['cy'],
                        final_sz
                    )

                # Otherwise output a black frame of the same size
                else:
                    cropped = np.zeros((final_sz, final_sz, 3), dtype=np.uint8)

                out.write(cropped)

            out.release()

        # PASS 3: Txt & Plot
        df_res = pd.DataFrame([{'frame': r['f'], 'prob': r['prob']} for r in frame_results])
        df_res['informative'] = df_res['frame'].isin(informative_idx_set)
        df_res.to_csv(TXT_DIR / f"{vid_id}.txt", sep='\t', index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(df_res['frame'], df_res['prob'], label='Ensemble Prob')
        plt.axhline(y=CONF_THRESHOLD, color='r', linestyle='--', alpha=0.5, label=f'Preferred ({CONF_THRESHOLD})')
        if threshold_used != CONF_THRESHOLD and not used_fallback:
            plt.axhline(y=threshold_used, color='orange', linestyle='--', label=f'Used ({threshold_used})')
        plt.fill_between(df_res['frame'], 0, 1.1, where=df_res['informative'],
                         color='green', alpha=0.1, label='Informative')
        plt.title(f"Informative Frames: {vid_id} ({len(informative_idx_set)} frames, {status})")
        plt.xlabel("Frame Index")
        plt.ylabel("Avg Probability")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(PLOT_DIR / f"{vid_id}.png")
        plt.close()

        cap.release()


if __name__ == "__main__":
    main()
