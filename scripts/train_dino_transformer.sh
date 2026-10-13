#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

source ~/.bashrc
conda activate job_new
cd ..

#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_1_0906_1934_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_1" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --num_blocks 2 \
#  --num_frames=-1
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_2_0906_2005_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_2" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --num_blocks 2 \
#  --num_frames=-1
#
#CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_3_0906_2036_full/best_model.pth
#python -u -m dino.train_dino_transformer \
#  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
#  --split_path "data/stratified_splits/split_3" \
#  --freeze_backbone \
#  --lr 1e-4 \
#  --epochs 5 \
#  --lr_step 3 \
#  --complex_augs \
#  --random_segment_sample \
#  --num_blocks 2 \
#  --num_frames=-1

CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_4_0906_2107_full/best_model.pth
python -u -m dino.train_dino_transformer \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_4" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --num_blocks 2

CLASSIFIER_WEIGHTS=classifier/models/dino_all_split_5_0906_2138_full/best_model.pth
python -u -m dino.train_dino_transformer \
  --classifier_checkpoint $CLASSIFIER_WEIGHTS \
  --split_path "data/stratified_splits/split_5" \
  --freeze_backbone \
  --lr 1e-4 \
  --epochs 5 \
  --lr_step 3 \
  --complex_augs \
  --random_segment_sample \
  --num_blocks 2
