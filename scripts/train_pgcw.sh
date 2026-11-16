#!/bin/bash
# *_pgcw (2026-10-30): the *_pgl runs (patient q5 splits, same HPs) + softer oversampling (--sampler_power 0.5),
# class-weighted loss (--class_weight_power 0.5) and black-border cropping (--crop_borders).
# Order: classifier -> DINO FT (init: best classifier of its split) -> ConvNeXt mixed -> DINO frozen -> ConvNeXt (2025 only).
# Local queue (not sbatch): each job opens a window in tmux session dr_hiwi_stage2 once this user holds < MAX_GPUS cards.
# Usage (from scripts/): SPLITS="1 2 3 4 5" bash train_pgcw.sh (ONLY="dfz cnxp": only those job names); evaluate with PGCW=1 sbatch vis_video_results.sh
# Ablations: TAG=pgc NEW="--crop_borders" ONLY="clf dft cnx" bash train_pgcw.sh (--crop_borders is the default since 2026-11-01;
# NEW="--no_crop_borders" = plain *_pgl config).
# EXTRA: appended to every job (overrides earlier args, e.g. EXTRA="--distance_scale_weight 0").
# BW: --backbone_weights for the classifier jobs only (SSL-adapted backbone, dino/ssl_lora.py); DINO transformers inherit it.
# Other split dir: SP=data/stratified_splits_paxos2025_patient_q5_v2 TAG=pgv2 NEW="" bash train_pgcw.sh (current defaults, crop on).
# CLF_EXTRA: appended to the classifier jobs only (e.g. CLF_EXTRA="--dense_adapter linear_max").
# CLFTAG: init the transformers from the clf_$CLFTAG classifiers (default: $TAG), e.g. a second transformer variant on existing classifiers.
# Current dataset (2026-11-12): SP=data/stratified_splits_paxos2025_patient_q5_v3 TAG=<tag> bash train_pgcw.sh
SPLITS=(${SPLITS:-1 2 3 4 5})
MAX_GPUS=${MAX_GPUS:-5}
ONLY=${ONLY:-clf dft cnx dfz cnxp}
SESSION=${SESSION:-dr_hiwi_stage2}
TAG=${TAG:-pgcw}
cd "$(dirname "$0")/.."
command -v gpu >/dev/null || { echo "needs the local 'gpu' wrapper + tmux (GPU workstation); on SLURM use the sbatch scripts"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
mkdir -p dino/logs convnext/logs results
SP=${SP:-data/stratified_splits_paxos2025_patient_q5}
EXTRA=${EXTRA:-}
CLF_EXTRA=${CLF_EXTRA:-}
CLFTAG=${CLFTAG:-$TAG}
BW=${BW:-}  # backbone weights for the classifier (e.g. dino/ssl_lora/<run>/merged_backbone.pth); transformers inherit them

new=${NEW-"--sampler_power 0.5 --class_weight_power 0.5 --crop_borders"}
common="--lr 1e-5 --min_lr 1e-7 --weight_decay 0.0 --img_size 512 --complex_augs --num_frames 32 --random_segment_sample \
 --aggregation_mode sparse_pool --num_query_tokens 4 --num_heads 8 --ffn_dim 2048 --dropout 0.1 --selection_metric combined"

JOBS=()  # name|split|log dir|python -m args
for s in "${SPLITS[@]}"; do
  JOBS+=("clf|$s|dino|dino.train_dino_classifier --split_path $SP/split_$s --lr 5e-6 --batch_size 2 --epochs 15 \
 --img_size 512 --complex_augs --lr_schedule cosine --min_lr 1e-7 --distance_scale_weight 0.2 --weight_decay 0.0 $new \
 ${BW:+--backbone_weights $BW} $CLF_EXTRA --run_tag clf_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("dft|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s $common $new \
 --lr_schedule cosine --num_blocks 2 --attn_entropy_weight 0.0 --distance_scale_weight 0.3 --backbone_lr_mult 0.1 --epochs 15 \
 --run_tag dino_ft_sparse_$TAG")
done
cnx_args="--adapter linear_attnpool --feature_stage 3 --feature_block -1 --epochs 25 --backbone_lr_mult 0.1 --num_blocks 1 \
 --distance_scale_weight 0.2"
for s in "${SPLITS[@]}"; do
  JOBS+=("cnx|$s|convnext|convnext.train_convnext_transformer --split_path $SP/split_$s $common $new \
 --train_json mil_train_mixed.json --fused_dataset $cnx_args --run_tag cnx_sparse_mixed_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("dfz|$s|dino|dino.train_dino_transformer --classifier_checkpoint CLF --split_path $SP/split_$s $common $new \
 --lr_schedule cosine --num_blocks 2 --attn_entropy_weight 0.0 --distance_scale_weight 0.3 --freeze_backbone --epochs 8 \
 --run_tag dino_sparse_$TAG")
done
for s in "${SPLITS[@]}"; do
  JOBS+=("cnxp|$s|convnext|convnext.train_convnext_transformer --split_path $SP/split_$s $common $new $cnx_args \
 --run_tag cnx_sparse_$TAG")
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
