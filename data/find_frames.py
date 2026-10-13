import argparse
import re
from pathlib import Path
import cv2

cv2.setNumThreads(1)  # avoid each parallel worker oversubscribing CPU cores internally

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    SUCCESS, WARNING, ERROR, INFO = Fore.GREEN, Fore.YELLOW, Fore.RED, Fore.CYAN
    RESET = Style.RESET_ALL
except ImportError:
    SUCCESS = WARNING = ERROR = INFO = RESET = ""

IMAGE_DIR = Path("data/2024_Paxos_Frames/frames")
VIDEO_DIR = Path("data/dr_videos")
OUTPUT_DIR = Path("data/results")
FRAMES_DIR = Path("data/matched_frames")  # New folder for raw matched frames

INITIAL_SAMPLE_EVERY = 30  # Start coarse: test 1 frame per ~1s (at 30fps)
SCORE_THRESHOLD = 0.99     # Success threshold; if below, retry with smaller step
REFINE_RADIUS = 30        # Local neighborhood search radius (± frames)
TOP_K = 3

OUTPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)


def key(path):
    """Make image/video filenames comparable by stripping common prefixes/suffixes."""
    s = path.stem
    s = re.sub(r"\.(MOV|MP4|AVI|MKV|M4V)-\d+$", "", s, flags=re.I)
    s = re.sub(r"^CLEAN_", "", s, flags=re.I)
    return s.lower().strip()


videos = {
    key(p): p for p in VIDEO_DIR.iterdir()
    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
}


def match(template, frame):
    """Find template at its original size using normalized cross-correlation."""
    if template is None or frame is None:
        return -1, None

    if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
        return -1, None

    a = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    return score, location


def search_at_step(template, video, sample_every):
    """Runs a single coarse-to-fine pass at a given sample_every stride."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    if not cap.isOpened() or total <= 0:
        cap.release()
        return -1, None, None, None, fps

    coarse = []

    # 1. Coarse Sampling
    for n in range(0, total, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if ok:
            score, loc = match(template, frame)
            if loc is not None:
                coarse.append((score, n))

    if not coarse:
        cap.release()
        return -1, None, None, None, fps

    # Keep top K coarse candidates
    coarse.sort(reverse=True, key=lambda x: x[0])
    candidates = coarse[:TOP_K]

    best_score = -1
    best_frame_no = None
    best_loc = None
    best_frame = None
    checked = set()

    # 2. Local Refinement Phase around candidate centers
    for _, center in candidates:
        start_frame = max(0, center - REFINE_RADIUS)
        end_frame = min(total - 1, center + REFINE_RADIUS)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for n in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break

            if n in checked:
                continue
            checked.add(n)

            score, loc = match(template, frame)
            if loc is not None and score > best_score:
                best_score = score
                best_frame_no = n
                best_loc = loc
                best_frame = frame.copy()

    cap.release()
    return best_score, best_frame_no, best_loc, best_frame, fps


def adaptive_search(template, video):
    """Dynamically halves sample_every from 30 down to 1 until score >= 0.99."""
    step = INITIAL_SAMPLE_EVERY

    while True:
        score, frame_no, loc, frame, fps = search_at_step(template, video, step)

        if score >= SCORE_THRESHOLD or step <= 1:
            return score, frame_no, loc, frame, fps, step

        # Halve the step size (minimum 1) and try again
        new_step = max(1, step // 2)
        print(f"  ├─ {WARNING}Score {score:.4f} < {SCORE_THRESHOLD} with step={step}. Retrying with step={new_step}...{RESET}")
        step = new_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--skip_first', type=int, default=0,
                        help="Skip this many images at the start of this shard's list (for resuming).")
    args = parser.parse_args()

    all_images = sorted(IMAGE_DIR.rglob("*.png"))
    images = all_images[args.shard_index::args.num_shards][args.skip_first:]
    print(f"{INFO}Found {len(all_images)} PNG images in '{IMAGE_DIR}' "
          f"(shard {args.shard_index}/{args.num_shards}: {len(images)} images). "
          f"Matching against '{VIDEO_DIR}'...{RESET}\n")

    for idx, image_path in enumerate(images, start=1):
        video = videos.get(key(image_path))
        print(f"[{idx}/{len(images)}] Processing: {image_path.name}")

        if video is None:
            print(f"  └─ {WARNING}No matching video found for key: '{key(image_path)}'{RESET}")
            continue

        raw_image = cv2.imread(str(image_path))
        if raw_image is None:
            print(f"  └─ {ERROR}Failed to load image file.{RESET}")
            continue

        # Fixed 90-degree clockwise rotation
        rotated_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

        score, frame_no, loc, frame, fps, used_step = adaptive_search(rotated_image, video)

        if frame is None or loc is None:
            print(f"  └─ {ERROR}No suitable frame match found in '{video.name}'.{RESET}")
            continue

        x, y = loc
        h, w = rotated_image.shape[:2]

        # 1. Save pure matched video frame to the dedicated matched_frames folder
        raw_frame_filename = (
            FRAMES_DIR / f"{video.stem}_frame{frame_no}_score{score:.4f}.png"
        )
        cv2.imwrite(str(raw_frame_filename), frame)

        # 2. Draw bounding box on a frame copy for the side-by-side result
        marked_frame = frame.copy()
        cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Build side-by-side comparison visualization
        H = max(rotated_image.shape[0], marked_frame.shape[0])
        img_padded = cv2.copyMakeBorder(
            rotated_image, 0, H - rotated_image.shape[0], 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        frame_padded = cv2.copyMakeBorder(
            marked_frame, 0, H - marked_frame.shape[0], 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        combined = cv2.hconcat([img_padded, frame_padded])

        timestamp = (frame_no / fps) if fps > 0 else 0.0
        out_filename = OUTPUT_DIR / f"{image_path.stem}_BEST.png"

        cv2.imwrite(str(out_filename), combined)

        print(f"  └─ {SUCCESS}Matched! Frame: {frame_no} | Time: {timestamp:.2f}s | "
              f"Pos: ({x}, {y}) | Score: {score:.4f} (Step: {used_step}){RESET}")
        print(f"  └─ Saved frame: {raw_frame_filename}")
        print(f"  └─ Saved comparison: {out_filename}\n")


if __name__ == "__main__":
    main()