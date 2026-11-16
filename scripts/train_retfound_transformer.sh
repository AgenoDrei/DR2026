#!/bin/bash
# RETFound_dinov2_meh (dino/RETFound_dinov2_meh.pth, gated HF repo YukunZhou/RETFound_dinov2_meh) as transformer backbone
# (2026-10-30): frozen (= dino_sparse_pgl HPs, 8 ep) and unfrozen x0.1 (= dino_ft_sparse_pgl HPs, 15 ep), no classifier init,
# at RETFound input 224 (native) and 448 (resampled pos. embedding), patient-level q5 splits (split 1 only by default). Local queue (not sbatch): each job opens a window in tmux session dr_hiwi_stage2 once this user
# holds < MAX_GPUS cards. Usage (from scripts/): SPLITS="1" RES="224 448" bash train_retfound_transformer.sh
SPLITS=(${SPLITS:-1})
RES=(${RES:-224 448})
MAX_GPUS=${MAX_GPUS:-5}
SESSION=${SESSION:-dr_hiwi_stage2}
cd "$(dirname "$0")/.."
command -v gpu >/dev/null || { echo "needs the local 'gpu' wrapper + tmux (GPU workstation); on SLURM use the sbatch scripts"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
mkdir -p dino/logs convnext/logs results
SP=data/stratified_splits_paxos2025_patient_q5
[ -f dino/RETFound_dinov2_meh.pth ] || { echo "missing dino/RETFound_dinov2_meh.pth"; exit 1; }

common="--backbone retfound_dinov2 --train_json mil_train.json --val_json mil_val.json --seed 42 --lr 1e-5 --lr_schedule cosine \
 --min_lr 1e-7 --optimizer adamw --weight_decay 0.0 --img_size 512 --complex_augs --num_frames 32 --random_segment_sample \
 --num_blocks 2 --num_heads 8 --ffn_dim 2048 --dropout 0.1 --aggregation_mode sparse_pool --attn_entropy_weight 0.0 \
 --distance_scale_weight 0.3 --selection_metric combined"

JOBS=()  # name|split|python -m args
for s in "${SPLITS[@]}"; do
  for r in "${RES[@]}"; do
    JOBS+=("rfz$r|$s|dino.train_dino_transformer --split_path $SP/split_$s $common --backbone_img_size $r --freeze_backbone \
 --epochs 8 --run_tag retfound${r}_sparse_pgl")
    JOBS+=("rft$r|$s|dino.train_dino_transformer --split_path $SP/split_$s $common --backbone_img_size $r --backbone_lr_mult 0.1 \
 --epochs 15 --run_tag retfound${r}_ft_sparse_pgl")
  done
done

mine() { gpu status | grep -c "$USER"; }
free() { gpu status | grep -c "free"; }
for job in "${JOBS[@]}"; do
  IFS='|' read -r name s cmd <<< "$job"
  until [ "$(mine)" -lt "$MAX_GPUS" ] && [ "$(free)" -ge 1 ]; do sleep 60; done
  win=${name}_s$s
  log=dino/logs/${win}_$(date +%Y-%m-%d_%H-%M).log
  tmux new-window -d -t "$SESSION" -n "$win" -c "$PWD" \
    "eval \"\$(conda shell.bash hook)\"; conda activate job_new; gpu run python -u -m $cmd 2>&1 | tee $log; exec env TMOUT=3600 bash"
  echo "$(date '+%m-%d %H:%M') launched $win -> $log"
  sleep 90  # let the run claim its card before the next check
done
echo "all launched"
