import cv2
from pathlib import Path

VIDEO_DIR = Path('data/dr_videos')
# VIDEO_DIR = Path('/home/user/mueller9/Data/2020_Paxos_Videos/data')
OUTPUT_DIR = Path('data/informative_frames/frames_raw_extract')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_fast_lossless():
    print(f"Searching for videos in: {VIDEO_DIR.resolve()}")
    video_files = [f for f in VIDEO_DIR.iterdir() if f.is_file() and f.suffix.lower() in {".mp4", ".mov"}]

    if not video_files:
        print(f"Error: No videos found at {VIDEO_DIR}")
        return

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping {video_path.name}: Error opening.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 5:
            cap.release()
            continue

        # 7-frame scheme matches informative_frames/split_info_full.csv (Paxos2025 labels)
        targets = [int(total_frames * k / 8) for k in range(1, 8)]
        # 4-frame scheme matches informative_frames/manual_labels_binary_paxos2020.csv (Paxos2020 labels)
        # targets = [int(total_frames * f) for f in [0.2, 0.4, 0.6, 0.8]]

        print(f"Processing: {video_path.name}")

        for t in targets:
            # Direct seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Switched extension to .png for lossless storage
                filename = f"{video_path.stem}_f{t}.png"
                save_path = OUTPUT_DIR / filename

                # OpenCV uses compression level (0-9) for PNG.
                # 0 is fastest/largest file, 3 is a good balance.
                cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                print(f"  [Warning] Failed to read frame {t} in {video_path.name}")

        cap.release()

    print(f"\nExtraction finished. Lossless PNG frames saved in: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    extract_fast_lossless()
