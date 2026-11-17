#!/bin/bash
# Local queue for dino/train_cached_transformer.py (head-only runs on cached CLS features, dino/cache_cls_features.py):
# every line "<tag> <extra args>" of $CONFIGS x SPLITS x SEEDS goes into a job pool file; each card (`gpu run`) runs PAR workers
# that pop jobs from it, so more cards can join later. Waits until the cache holds EXPECT clips; resumable (runs whose log has "Done." are skipped).
# Usage (repo root, inside tmux, `| tee`): CONFIGS=results/queues/cached_stageA.txt bash scripts/train_cached_transformer.sh
# BASE="<args>": put before every line's args (lines override it); PREFIX: prepended to every tag (e.g. the base's name).
# Add a card to a running queue: JOIN=<pool file printed at start> bash scripts/train_cached_transformer.sh
# Logs: dino/logs/cached/<PREFIX><tag>_s<split>_seed<seed>.log; results: results/run_results.csv (run tag cached_<PREFIX><tag>).
cd "$(dirname "$0")/.."
CACHE=${CACHE:-data/cls_cache/lvd_q5v3}
SP=${SP:-data/stratified_splits_paxos2025_patient_q5_v3}
LOGDIR=dino/logs/cached
PAR=${PAR:-6}

if [ "$1" == "--one" ]; then  # one run: "<tag>|<split>|<seed>|<extra args>"
  IFS='|' read -r tag s seed extra <<< "$2"
  log=$LOGDIR/${tag}_s${s}_seed${seed}.log
  if grep -q "^Done" "$log" 2>/dev/null; then echo "skip $tag s$s seed$seed"; exit 0; fi
  if OMP_NUM_THREADS=4 python -u -m dino.train_cached_transformer --cache_dir "$CACHE" --split_path "$SP/split_$s" \
       --seed "$seed" $extra --run_tag "cached_$tag" > "$log" 2>&1; then
    echo "$(date +%H:%M) done $tag s$s seed$seed | $(grep '^Epoch' "$log" | tail -1)"
  else
    echo "$(date +%H:%M) FAILED $tag s$s seed$seed ($log)"
  fi
  exit 0
fi

if [ "$1" == "--worker" ]; then  # pop jobs from the pool file until it is empty
  while true; do
    job=$(flock "$2.lock" bash -c 'j=$(head -n 1 "$0"); [ -n "$j" ] && sed -i 1d "$0"; echo "$j"' "$2")
    [ -z "$job" ] && exit 0
    bash scripts/train_cached_transformer.sh --one "$job"
  done
fi

workers() {  # PAR workers on one card; CACHE and SP come from the pool's .env file
  IFS='|' read -r CACHE SP < "$1.env"; export CACHE SP
  gpu run bash -c "seq $PAR | xargs -P $PAR -I{} bash scripts/train_cached_transformer.sh --worker $1"
}

if [ -n "$JOIN" ]; then
  echo "$(date '+%m-%d %H:%M') joining $JOIN ($(wc -l < "$JOIN") jobs left), $PAR workers"
  workers "$JOIN"; echo "$(date '+%m-%d %H:%M') pool empty"; exit 0
fi

: "${CONFIGS:?set CONFIGS=<file with lines '<tag> <extra args>'>}"
SPLITS=${SPLITS:-1 2 3 4 5}
SEEDS=${SEEDS:-42 43 44}
EXPECT=${EXPECT:-302}
BASE=${BASE:-}
PREFIX=${PREFIX:-}
mkdir -p "$LOGDIR"
pool=results/queues/$(basename "$CONFIGS" .txt)_${PREFIX}s${SPLITS// /}_$(date +%m%d_%H%M).pool
: > "$pool"; echo "$CACHE|$SP" > "$pool.env"
while read -r tag extra; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  for s in $SPLITS; do for seed in $SEEDS; do
    grep -q "^Done" "$LOGDIR/$PREFIX${tag}_s${s}_seed${seed}.log" 2>/dev/null || echo "$PREFIX$tag|$s|$seed|$BASE $extra" >> "$pool"
  done; done
done < "$CONFIGS"

until [ "$(ls "$CACHE"/*.npy 2>/dev/null | wc -l)" -ge "$EXPECT" ]; do sleep 60; done
echo "$(date '+%m-%d %H:%M') $(wc -l < "$pool") runs from $CONFIGS, $PAR per card; pool $pool (add a card: JOIN=$pool bash $0)"
workers "$pool"
echo "$(date '+%m-%d %H:%M') pool empty"
