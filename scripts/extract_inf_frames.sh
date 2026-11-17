#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Paxos2025 production pipeline: score every video in data/dr_videos with the
# 5-fold seg + cls models (per video, the fold model of its val fold from
# cropping/splits.json -- never trained on that video; unknown videos get the
# full 5-fold probability ensembles) and write the cleaned clips,
# per-frame probability txt files and plots to data/ensemble_results_paxos2025/.
#
# Prerequisites: trained checkpoints in cropping/models (seg_fold_<i>.pth)
# and informative_frames/models (cls_fold_<i>.pth).
#
# Usage: sbatch scripts/extract_inf_frames.sh
#        TOTAL_GPUS=4 sbatch --array=0-3 scripts/extract_inf_frames.sh   (multi-GPU)

source ~/.bashrc
conda activate job_new

GPU_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_GPUS=${TOTAL_GPUS:-1}

python -u -m informative_frames.extract_inf_frames \
  --gpu_id "$GPU_ID" \
  --total_gpus "$TOTAL_GPUS" \
  --video_dir data/dr_videos \
  --seg_model_dir cropping/models \
  --cls_model_dir informative_frames/models \
  --out_base data/ensemble_results_paxos2025
