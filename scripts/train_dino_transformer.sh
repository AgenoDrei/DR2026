#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# DINOv3 transformer aggregator (5-class, sparse pooling), one split, initialized from a trained classifier
# (train_dino_classifier.sh); the `dfz` (frozen, 8 epochs) / `dft` (FT=1: backbone lr x0.1, 15 epochs) jobs of train_pgcw.sh.
# Usage (from scripts/): CLF=classifier/models/<run>/best_model.pth TAG=<tag> SPLIT=1 sbatch train_dino_transformer.sh   (SP=, EXTRA= as in train_dino_classifier.sh)
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
[ -n "$TAG" ] || { echo "set TAG=<run tag>"; exit 1; }
[ -n "$CLF" ] || { echo "set CLF=<classifier best_model.pth>"; exit 1; }
if [ -n "$FT" ]; then mode="--epochs 15"; name=dino_ft_sparse_$TAG
else mode="--freeze_backbone --epochs 8"; name=dino_sparse_$TAG; fi

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_transformer --classifier_checkpoint "$CLF" --split_path "$SP/split_$SPLIT" $mode --run_tag "$name" $EXTRA
