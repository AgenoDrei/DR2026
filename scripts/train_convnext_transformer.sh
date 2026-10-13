#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodelist=node-03
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

python -u -m convnext.train_convnext_transformer \
  --split_path "data/stratified_splits/split_1" \
  --train_json mil_train_2025.json \
  --val_json mil_val_2025.json \
  --adapter linear_attnpool \
  --complex_augs \
  --random_segment_sample
