import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# LABEL_CSV = 'informative_frames/split_info_full.csv'
LABEL_CSV = 'informative_frames/manual_labels_binary_paxos2020.csv'
# INPUT_DIR = Path('data/informative_frames/frames_raw_extract')
INPUT_DIR = Path('data/informative_frames/frames_raw_extract_paxos2020')
OUTPUT_DIR = Path('data/informative_frames/frames_cropped')
# OUTPUT_DIR = Path('data/informative_frames/frames_cropped_paxos2020')
MODEL_FOLDER = Path('cropping/models')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 768
N_FOLDS = 5

(OUTPUT_DIR / '1').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / '0').mkdir(parents=True, exist_ok=True)

def get_inference_transforms():
    return albu.Compose([
        albu.Resize(IMG_SIZE, IMG_SIZE),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def load_ensemble():
    models = []
    for i in range(N_FOLDS):
        model_path = MODEL_FOLDER / f'best_model_fold_{i}.pth'
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE).eval()
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
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)
    if num_labels <= 1: return None, None, None, None, None
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.uint8(labels == largest_label) * 255
    x, y, w, h, _ = stats[largest_label]
    cx, cy = centroids[largest_label]
    return clean_mask, w, h, cx, cy

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

def process_frames():
    models = load_ensemble()
    transforms = get_inference_transforms()
    df = pd.read_csv(LABEL_CSV)

    print(f"Processing {len(df)} images from CSV...")

    for _, row in df.iterrows():
        img_name = row['filename']
        label = str(int(row['label'])) # Uses the 0 or 1 from your CSV
        img_path = INPUT_DIR / img_name

        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        height, width = frame.shape[:2]

        sq_frame, y_off, x_off, side = pad_to_square_info(frame)
        img_rgb = cv2.cvtColor(sq_frame, cv2.COLOR_BGR2RGB)
        input_tensor = transforms(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)

        ensemble_prob = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        with torch.no_grad():
            for model in models:
                output = model(input_tensor)
                ensemble_prob += (torch.sigmoid(output).squeeze().cpu().numpy() / N_FOLDS)

        sq_mask = cv2.resize(ensemble_prob, (side, side))
        rect_mask = sq_mask[y_off:y_off + height, x_off:x_off + width]
        binary_mask = (rect_mask > 0.5).astype(np.uint8)

        clean_mask, w, h, cx, cy = get_largest_component_mask(binary_mask)

        if clean_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=clean_mask)
            crop_dim = int(max(w, h) * 1.05)
            cropped_img = crop_and_pad(masked_frame, cx, cy, crop_dim)

            save_name = Path(img_name).stem + ".png"
            cv2.imwrite(str(OUTPUT_DIR / label / save_name), cropped_img)
        else:
            print(f"Skipping {img_name}: No retina detected.")

if __name__ == "__main__":
    process_frames()