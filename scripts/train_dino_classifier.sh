#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Frame-level DINOv3 classifier (5-class, unfrozen backbone), one split; the `clf` job of train_pgcw.sh (script defaults).
# Usage (from scripts/): TAG=<tag> SPLIT=1 sbatch train_dino_classifier.sh   (SP= other split dir, EXTRA= extra args)
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
[ -n "$TAG" ] || { echo "set TAG=<run tag>"; exit 1; }

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_classifier --split_path "$SP/split_$SPLIT" --run_tag "clf_$TAG" $EXTRA
