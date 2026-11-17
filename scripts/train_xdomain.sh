#!/bin/bash
# Cross-domain validation: train on 100% of the Paxos2025 q5 data, test on Paxos2020
# (split: `python -m data.make_xdomain_split --num_seeds N`). Hyperparameters are the training scripts' defaults; evaluate
# last_*.pth only (Paxos2020 is passed as val_json purely as a monitor). One pseudo-split per seed: seed i of SEEDS -> split_i.
# Local queue (not sbatch): each job opens a window in tmux session dr_hiwi_stage2 once this user holds < MAX_GPUS cards.
# DINO transformers wait for the classifier of their seed. Usage (from scripts/): XTAG=<tag> SEEDS="42 43 44" bash train_xdomain.sh
# EXTRA="<args>" is appended to every run, ONLY="clf cnx dft dfz" selects jobs. Evaluate: XDOM=1 XTAG=<tag> CROP=1 ZOOM=1 sbatch vis_video_results.sh
SEEDS=(${SEEDS:-42})
MAX_GPUS=${MAX_GPUS:-5}
SESSION=${SESSION:-dr_hiwi_stage2}
[ -n "$XTAG" ] || { echo "set XTAG=<run tag>"; exit 1; }
EXTRA=${EXTRA:-}
ONLY=${ONLY:-clf cnx dft dfz}  # job names to launch
cd "$(dirname "$0")/.."
command -v gpu >/dev/null || { echo "needs the local 'gpu' wrapper + tmux (GPU workstation); on SLURM use the sbatch scripts"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
mkdir -p dino/logs convnext/logs results
SP=data/stratified_splits_paxos2025_q5_full_xdom

common="--train_json mil_train.json --val_json mil_val.json --fused_dataset $EXTRA"

JOBS=()  # name|seed index|log dir|python -m args
for i in "${!SEEDS[@]}"; do
  s=$((i + 1)); seed=${SEEDS[$i]}
  JOBS+=("clf|$s|dino|dino.train_dino_classifier --split_path $SP/split_$s --seed $seed $EXTRA --run_tag clf_${XTAG}_s$seed")
  JOBS+=("cnx|$s|convnext|convnext.train_convnext_transformer --split_path $SP/split_$s --seed $seed $common \
 --run_tag cnx_sparse_${XTAG}_s$seed")
done
for i in "${!SEEDS[@]}"; do
  s=$((i + 1)); seed=${SEEDS[$i]}
  JOBS+=("dft|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s --seed $seed $common \
 --epochs 15 --run_tag dino_ft_sparse_${XTAG}_s$seed")
  JOBS+=("dfz|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s --seed $seed $common \
 --freeze_backbone --epochs 8 --run_tag dino_sparse_${XTAG}_s$seed")
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
