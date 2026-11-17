#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Paxos2025 cropping experiment: train the Unet++ (B4) segmentation model over the
# 5 subject-level folds of cropping/splits.json.
#
# Prerequisite (run from the repo root first; builds the 5 subject-level folds from
# cropping/annotations_cropping_2025.csv -- the annotated frames must exist in
# cropping/frames_to_annotate_larger/):
#   python -u -m cropping.make_splits_from_q5v3
#
# Usage: sbatch scripts/train_cropping.sh
#        FOLDS="0" bash scripts/train_cropping.sh   (subset of folds)

source ~/.bashrc
conda activate job_new

FOLDS=${FOLDS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-20}
IMG_DIR=${IMG_DIR:-cropping/frames_to_annotate_larger}
OUT_DIR=${OUT_DIR:-cropping/models}
BATCH_SIZE=${BATCH_SIZE:-8}     # raise (16/32) if the GPU has headroom; A100 + AMP fits easily
NUM_WORKERS=${NUM_WORKERS:-16}
IMG_SIZE=${IMG_SIZE:-768}

for fold in $FOLDS; do
  echo "=== train fold=$fold epochs=$EPOCHS ==="
  python -u -m cropping.train_segmentation \
    --fold "$fold" --epochs "$EPOCHS" \
    --img_dir "$IMG_DIR" --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --img_size "$IMG_SIZE" --amp
done

echo "All cropping training runs finished. Models in $OUT_DIR"
