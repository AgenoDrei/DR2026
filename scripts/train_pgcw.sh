#!/bin/bash
# Trains the reference models on every split of the current dataset (v3): frame classifier (clf) -> DINO FT (dft) and
# DINO frozen (dfz), both initialized from the best classifier of their split -> ConvNeXt + Paxos2020 mixing (cnx) and
# ConvNeXt (cnxp). Hyperparameters are the scripts' defaults.
# Local queue (not sbatch): each job opens a window in tmux session dr_hiwi_stage2 once this user holds < MAX_GPUS cards.
# Usage (from scripts/): TAG=<tag> bash train_pgcw.sh; evaluate with PGCW=1 PTAG=<tag> Q5_SPLITS=<SP> ZOOM=1 sbatch vis_video_results.sh
#   SPLITS="1 2 3 4 5", ONLY="clf dft cnx dfz cnxp" (job names), SP=<split dir>, EXTRA="<args>" (appended to every job),
#   CLFTAG=<tag>: init the transformers from the clf_<CLFTAG> classifiers (default: $TAG).
SPLITS=(${SPLITS:-1 2 3 4 5})
MAX_GPUS=${MAX_GPUS:-5}
ONLY=${ONLY:-clf dft cnx dfz cnxp}
SESSION=${SESSION:-dr_hiwi_stage2}
[ -n "$TAG" ] || { echo "set TAG=<run tag> (new tag: the DINO jobs wait for clf_<TAG> in results/run_results.csv)"; exit 1; }
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
EXTRA=${EXTRA:-}
CLFTAG=${CLFTAG:-$TAG}
cd "$(dirname "$0")/.."
command -v gpu >/dev/null || { echo "needs the local 'gpu' wrapper + tmux (GPU workstation); on SLURM use the sbatch scripts"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
mkdir -p dino/logs convnext/logs results

JOBS=()  # name|split|log dir|python -m args
for s in "${SPLITS[@]}"; do
  JOBS+=("clf|$s|dino|dino.train_dino_classifier --split_path $SP/split_$s --run_tag clf_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("dft|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s --epochs 15 \
 --run_tag dino_ft_sparse_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("cnx|$s|convnext|convnext.train_convnext_transformer --split_path $SP/split_$s --train_json mil_train_mixed.json \
 --fused_dataset --run_tag cnx_sparse_mixed_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("dfz|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s --freeze_backbone \
 --epochs 8 --run_tag dino_sparse_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("cnxp|$s|convnext|convnext.train_convnext_transformer --split_path $SP/split_$s --run_tag cnx_sparse_$TAG")
done

mine() { gpu status | grep -c "$USER"; }
free() { gpu status | grep -c "free"; }
for job in "${JOBS[@]}"; do
  IFS='|' read -r name s pkg cmd <<< "$job"
  [[ " $ONLY " == *" $name "* ]] || continue
  if [[ $cmd == *" CLF "* ]]; then
    until grep -q "_split_${s}_full_clf_$CLFTAG," results/run_results.csv; do sleep 60; done
    ckpt=$(ls -d classifier/models/*_split_${s}_full_clf_$CLFTAG | tail -1)/best_model.pth
    [ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
    cmd=${cmd/ CLF / $ckpt }
  fi
  until [ "$(mine)" -lt "$MAX_GPUS" ] && [ "$(free)" -ge 1 ]; do sleep 60; done
  win=${name}_${TAG}_s$s
  log=$pkg/logs/${win}_$(date +%Y-%m-%d_%H-%M).log
  tmux new-window -d -t "$SESSION" -n "$win" -c "$PWD" \
    "eval \"\$(conda shell.bash hook)\"; conda activate job_new; gpu run python -u -m $cmd $EXTRA 2>&1 | tee $log; exec env TMOUT=3600 bash"
  echo "$(date '+%m-%d %H:%M') launched $win -> $log"
  sleep 90  # let the run claim its card before the next check
done
echo "all launched"
