#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Baseline (set 2026-10-13): frozen-backbone transformer with standard random segment sampling, Paxos2025 split_1, 5-class.
# Reference run: 2026-10-14_08-21_dino_sa_all_split_1_full_baseline
#   best (epoch 3): QWK 0.850 | acc 0.650 | macro F1 0.593 | referable recall 0.848
#   seed 43 (run_tag baseline_seed43), best (epoch 2): QWK 0.881 | acc 0.667 | macro F1 0.581 | referable recall 0.788
# Same config with --cluster_sample instead (2026-10-13_12-30_dino_sa_all_split_1_full_cluster_frozen_b2h8_ent0_dist0.3):
#   best (epoch 5): QWK 0.862 | acc 0.717 | macro F1 0.672 | referable recall 0.848

source ~/.bashrc
conda activate job_new
cd ..

python -u -m dino.train_dino_transformer \
  --classifier_checkpoint classifier/models/dino_all_split_1_1002_1543_full/best_model.pth \
  --split_path data/stratified_splits_paxos2025/split_1 \
  --train_json mil_train.json --val_json mil_val.json \
  --seed 42 \
  --freeze_backbone \
  --epochs 6 --lr_step 3 --lr 5e-5 \
  --img_size 512 --complex_augs \
  --num_blocks 2 --num_heads 8 --ffn_dim 2048 --dropout 0.1 --num_frames 32 \
  --attn_entropy_weight 0.0 --distance_scale_weight 0.3 \
  --random_segment_sample \
  --run_tag baseline
