#!/bin/bash
# LoRA self-supervised (DINO + iBOT + KoLeo + Gram) domain pretraining of DINOv3 ViT-L/16 on Paxos clips (dino/ssl_lora.py).
# Local tmux job (not sbatch). Needs the frames first: python -m data.extract_ssl_frames (CPU, ~80k frames).
# Usage (from scripts/): NGPU=4 TAG=v2 EXTRA="--frame_list data/ensemble_results_paxos2025/ssl_frame_quality/keep_p10.txt" bash train_ssl_lora.sh
# v1 (2026-11-10) used the old defaults: EXTRA="--global_size 256 --batch_size 32 --steps 5000" (+ the old step-based schedule).
# v3 (2026-11-11): Gram anchoring (--gram_weight 1, teacher views 512 px) + local DINO loss weight 0.5 are now defaults
# (v2 = --gram_weight 0 --local_loss_weight 1); Paxos2025 + Paxos2020 frames, p10 quality filter per cohort (~9 h on 4 GPUs):
#   NGPU=4 TAG=v3_p2520 EXTRA="--frame_dir data --frame_list data/ssl_frame_quality_paxos2025_2020/keep_p10.txt" bash train_ssl_lora.sh
# Output: dino/ssl_lora/<run>/merged_backbone.pth -> BW=<that path> ONLY="clf dfz dft" bash train_pgcw.sh
NGPU=${NGPU:-5}
TAG=${TAG:-v3}
EXTRA=${EXTRA:-}
SESSION=${SESSION:-dr_hiwi_stage2}
cd "$(dirname "$0")/.."
command -v gpu >/dev/null || { echo "needs the local 'gpu' wrapper + tmux (GPU workstation)"; exit 1; }

win=ssl_lora_$TAG
log=dino/logs/${win}_$(date +%Y-%m-%d_%H-%M).log
mkdir -p dino/logs
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
tmux new-window -d -t "$SESSION" -n "$win" -c "$PWD" \
  "eval \"\$(conda shell.bash hook)\"; conda activate job_new; gpu -n $NGPU run env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node $NGPU \
 -m dino.ssl_lora --run_tag $TAG $EXTRA 2>&1 | tee $log; exec env TMOUT=3600 bash"
echo "launched $win ($NGPU GPUs) -> $log"
