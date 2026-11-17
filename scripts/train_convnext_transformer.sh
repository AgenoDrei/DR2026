#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# DINOv3 ConvNeXt-L transformer (stage-3 features, backbone lr x0.1, sparse pooling), one split; the `cnxp` job of
# train_pgcw.sh (MIXED=1: + the 53 Paxos2020 training videos = the `cnx` job).
# Usage (from scripts/): TAG=<tag> SPLIT=1 sbatch train_convnext_transformer.sh   (SP=, EXTRA= as in train_dino_classifier.sh)
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
[ -n "$TAG" ] || { echo "set TAG=<run tag>"; exit 1; }
if [ -n "$MIXED" ]; then data="--train_json mil_train_mixed.json --fused_dataset"; name=cnx_sparse_mixed_$TAG
else data=""; name=cnx_sparse_$TAG; fi

source ~/.bashrc
conda activate job_new
cd ..

python -u -m convnext.train_convnext_transformer --split_path "$SP/split_$SPLIT" $data --run_tag "$name" $EXTRA
