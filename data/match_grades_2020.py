import json
import csv
import os

cleaned_videos_dir = "data/ensemble_results_paxos2020/cleaned_videos"
annotations_csv = os.path.expanduser("~/Data/2020_Paxos_Videos/annotations.csv")

# 1. Load grades from annotations.csv, keyed by base video id (no extension).
grades = {}
with open(annotations_csv, mode='r') as f:
    for row in csv.DictReader(f):
        video_id = os.path.splitext(row['video_name'].strip())[0]
        grades[video_id] = int(row['dr_grade'])

print(f"Loaded {len(grades)} grades from {annotations_csv}")

# 2. Walk the cleaned clips, matching each to its grade by base video id.
clip_ids = sorted(
    os.path.splitext(f)[0][len("CLEAN_"):]
    for f in os.listdir(cleaned_videos_dir)
    if f.startswith("CLEAN_") and f.lower().endswith(".mp4")
)

matches = {}
clip_labels = {}
unmatched_clips = []

for clip_id in clip_ids:
    matches[clip_id] = []  # no still-image counterpart for 2020
    if clip_id in grades:
        clip_labels[f"{clip_id}.mp4"] = grades[clip_id]
    else:
        unmatched_clips.append(clip_id)

results = {
    "matches": matches,
    "unmatched_clips": unmatched_clips,
    "unmatched_images": [],
}

with open("data/matching_results_2020.json", "w") as f:
    json.dump(results, f, indent=4)

with open("data/clip_labels_2020.json", "w") as f:
    json.dump(clip_labels, f, indent=4)

print(f"Clips found: {len(clip_ids)}")
print(f"Clips graded: {len(clip_labels)}")
print(f"Clips with no grade in {annotations_csv}: {len(unmatched_clips)}")
if unmatched_clips:
    print(f"  -> {unmatched_clips}")
print("Success!")
