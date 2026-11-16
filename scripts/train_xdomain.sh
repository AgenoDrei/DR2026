#!/bin/bash
# Cross-domain validation (2026-10-29): train on 100% of the Paxos2025 q5 data, test on Paxos2020
# (split: `python -m data.make_xdomain_split --num_seeds N`). HPs = the *_pgl runs; evaluate last_*.pth only
# (Paxos2020 is passed as val_json purely as a monitor). One pseudo-split per seed: seed i of SEEDS -> split_i.
# Local queue (not sbatch): each job opens a window in tmux session dr_hiwi_stage2 once this user holds < MAX_GPUS cards.
# DINO transformers wait for the classifier of their seed. Usage (from scripts/): SEEDS="42 43 44" bash train_xdomain.sh
# EXTRA="<args>" is appended to every run, XTAG (default xdom) names them, e.g. the 2026-10-30 rerun:
# XTAG=xdomcw EXTRA="--sampler_power 0.5 --class_weight_power 0.5 --crop_borders" bash train_xdomain.sh
# Paxos2020-style augs (2026-11-02, CNX only): ONLY=cnx XTAG=xdp20 EXTRA="--no_clahe --bag_consistent_augs --extra_augs p20_color p20_quality p20_light"
# --crop_borders is the default since 2026-11-01 (evaluate with CROP=1); the original xdom runs = EXTRA="--no_crop_borders".
SEEDS=(${SEEDS:-42})
MAX_GPUS=${MAX_GPUS:-5}
SESSION=${SESSION:-dr_hiwi_stage2}
XTAG=${XTAG:-xdom}
EXTRA=${EXTRA:-}
ONLY=${ONLY:-clf cnx dft dfz}  # job names to launch
cd "$(dirname "$0")/.."
command -v gpu >/dev/null || { echo "needs the local 'gpu' wrapper + tmux (GPU workstation); on SLURM use the sbatch scripts"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
mkdir -p dino/logs convnext/logs results
SP=data/stratified_splits_paxos2025_q5_full_xdom

common="--train_json mil_train.json --val_json mil_val.json --fused_dataset --lr 1e-5 --min_lr 1e-7 --weight_decay 0.0 \
 --img_size 512 --complex_augs --num_frames 32 --random_segment_sample --aggregation_mode sparse_pool --selection_metric combined $EXTRA"
dino_args="--lr_schedule cosine --num_blocks 2 --num_heads 8 --ffn_dim 2048 --dropout 0.1 --attn_entropy_weight 0.0 \
 --distance_scale_weight 0.3"
cnx_args="--adapter linear_attnpool --feature_stage 3 --feature_block -1 --epochs 25 --backbone_lr_mult 0.1 --num_blocks 1 \
 --distance_scale_weight 0.2"

JOBS=()  # name|seed index|log dir|python -m args
for i in "${!SEEDS[@]}"; do
  s=$((i + 1)); seed=${SEEDS[$i]}
  JOBS+=("clf|$s|dino|dino.train_dino_classifier --split_path $SP/split_$s --seed $seed --lr 5e-6 --batch_size 2 --epochs 15 \
 --img_size 512 --complex_augs --lr_schedule cosine --min_lr 1e-7 --distance_scale_weight 0.2 --weight_decay 0.0 $EXTRA --run_tag clf_${XTAG}_s$seed")
  JOBS+=("cnx|$s|convnext|convnext.train_convnext_transformer --split_path $SP/split_$s --seed $seed $common $cnx_args \
 --run_tag cnx_sparse_${XTAG}_s$seed")
done
for i in "${!SEEDS[@]}"; do
  s=$((i + 1)); seed=${SEEDS[$i]}
  JOBS+=("dft|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s --seed $seed $common \
 $dino_args --backbone_lr_mult 0.1 --epochs 15 --run_tag dino_ft_sparse_${XTAG}_s$seed")
  JOBS+=("dfz|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s --seed $seed $common \
 $dino_args --freeze_backbone --epochs 8 --run_tag dino_sparse_${XTAG}_s$seed")
done

mine() { gpu status | grep -c "$USER"; }
free() { gpu status | grep -c "free"; }
for job in "${JOBS[@]}"; do
  IFS='|' read -r name s pkg cmd <<< "$job"
  [[ " $ONLY " == *" $name "* ]] || continue
  seed=${SEEDS[$((s - 1))]}
  if [[ $cmd == *" CLF "* ]]; then
    until grep -q "_full_clf_${XTAG}_s$seed," results/run_results.csv; do sleep 60; done
    ckpt=$(ls -d classifier/models/*_split_${s}_full_clf_${XTAG}_s$seed | tail -1)/last_model.pth
    [ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
    cmd=${cmd/ CLF / $ckpt }
  fi
  until [ "$(mine)" -lt "$MAX_GPUS" ] && [ "$(free)" -ge 1 ]; do sleep 60; done
  win=${name}_${XTAG}_s$seed
  log=$pkg/logs/${win}_$(date +%Y-%m-%d_%H-%M).log
  tmux new-window -d -t "$SESSION" -n "$win" -c "$PWD" \
    "eval \"\$(conda shell.bash hook)\"; conda activate job_new; gpu run python -u -m $cmd 2>&1 | tee $log; exec env TMOUT=3600 bash"
  echo "$(date '+%m-%d %H:%M') launched $win -> $log"
  sleep 90  # let the run claim its card before the next check
done
echo "all launched"
