#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Evaluate the Paxos2025 informative-frame classifier experiment: every trained
# classifier (informative_frames/models/cls_fold_<i>.pth) is evaluated on the
# fold-i val rows of the split CSV, on the frames cropped by scripts/crop_frames.sh.
# Writes per-image metrics, summary CSVs and classification_eval.png
# (one bar per metric, points = the 5 fold values) to informative_frames/eval/.
#
# Usage: sbatch scripts/eval_classify.sh

source ~/.bashrc
conda activate job_new

python -u -m informative_frames.evaluate_classification \
  --models_dir informative_frames/models \
  --img_base data/informative_frames \
  --split_dir informative_frames \
  --out_dir informative_frames/eval
