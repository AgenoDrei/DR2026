#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Crop the raw Paxos2025 informative frames with the 5-fold cropping models
# (each frame by the fold model of its own val fold -> never trained on it) into
# data/informative_frames/frames_cropped/<label>/.
#
# Prerequisites (run from the repo root first):
#   python -u -m cropping.make_splits_from_q5v3
#   trained checkpoints in cropping/models (seg_fold_<i>.pth)
#   raw frames in data/informative_frames/frames_raw_extract
#
# Usage: sbatch scripts/crop_frames.sh

source ~/.bashrc
conda activate job_new

python -u -m informative_frames.crop_frames \
  --input_dir data/informative_frames/frames_raw_extract \
  --split_csv informative_frames/split_info_5fold_paxos2025_from_q5v3.csv \
  --model_dir cropping/models \
  --output_dir data/informative_frames/frames_cropped
