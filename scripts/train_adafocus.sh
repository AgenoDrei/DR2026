#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Uni-AdaFocus (adafocus/): MobileNetV2 glance over 32 of 64 frames -> spatial/temporal policy -> 16 patches through
# the plain DINOv3 ViT-L (frozen), 5-class, one split.
# Usage (from scripts/): SPLIT=1 sbatch train_adafocus.sh
#   GLOBAL=dinov3_convnext_base|dinov3_vitl swaps the glance encoder, LOCAL=dinov3_convnext_large|_base the patch encoder; FT=1 fine-tunes the local ViT-L (lr x0.1).
#   SP=, TAG=, EXTRA= as in train_dino_transformer.sh.
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
SPLIT=${SPLIT:-1}
GLOBAL=${GLOBAL:-mobilenetv2}
LOCAL=${LOCAL:-dinov3_vitl}
TAG=${TAG:-uaf}
if [ -n "$FT" ]; then mode="--unfreeze_local --local_lr_mult 0.1"; else mode=""; fi

source ~/.bashrc
conda activate job_new
cd ..

python -u -m adafocus.train_adafocus \
  --split_path "$SP/split_$SPLIT" \
  --global_encoder "$GLOBAL" --local_encoder "$LOCAL" \
  --epochs 20 --lr 1e-4 --min_lr 1e-7 \
  --img_size 512 --complex_augs --random_segment_sample \
  --num_frames 64 --num_glance 32 --num_focus 16 --patch_size 224 \
  --distance_scale_weight 0.3 --selection_metric combined \
  --sampler_power 0.5 --class_weight_power 0.5 \
  $mode --run_tag "$TAG" $EXTRA
