#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Evaluate the Paxos2025 cropping experiment: every trained model
# (cropping/models/seg_fold_<i>.pth) is evaluated on the fold-i val split of
# cropping/splits.json. Writes per-image metrics, summary CSVs and
# segmentation_eval.png (one bar per metric, points = the 5 fold values)
# to cropping/eval/.
#
# Usage: sbatch scripts/eval_cropping.sh

source ~/.bashrc
conda activate job_new

python -u -m cropping.evaluate_segmentation \
  --models_dir cropping/models \
  --img_dir cropping/frames_to_annotate_larger \
  --out_dir cropping/eval
