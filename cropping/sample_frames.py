import cv2
import numpy as np
from pathlib import Path

VIDEO_DIR = Path('data/dr_videos')
# VIDEO_DIR = Path('/home/user/mueller9/Data/2020_Paxos_Videos/data')
OUTPUT_DIR = Path('cropping/frames_to_annotate_larger')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pad_to_square(image):
    # Adds black padding to make the image 1:1 without stretching.
    h, w = image.shape[:2]
    max_side = max(h, w)
    squared_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)

    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2

    squared_img[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return squared_img


def extract_and_pad_sequential():
    print(f"Searching for videos in: {VIDEO_DIR.resolve()}")
    video_files = list(VIDEO_DIR.glob('*.mp4')) + list(VIDEO_DIR.glob('*.MOV')) + list(VIDEO_DIR.glob('*.mov'))

    if not video_files:
        print(f"Error: No videos found at {VIDEO_DIR}")
        return

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping {video_path.name}: Error opening.")
            continue

        # 1. Determine target frame numbers
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Skipping {video_path.name}: Invalid frame count.")
            continue

        targets = [int(total_frames * f) for f in [0.50]]
        max_target = max(targets)

        print(f"Processing: {video_path.name} ({total_frames} frames)...")

        # 2. Read frames sequentially
        current_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in targets:
                # 3. Square and Save
                square_frame = pad_to_square(frame)
                filename = f"{video_path.stem}_frame_{current_idx}.png"
                save_path = OUTPUT_DIR / filename

                cv2.imwrite(str(save_path), square_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                print(f"  -> Extracted frame {current_idx}")

            current_idx += 1

            if current_idx > max_target:
                break

        cap.release()

    print(f"\nSuccess! All cleaned frames are in: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    extract_and_pad_sequential()