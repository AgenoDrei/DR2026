#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Full Paxos2025 pre-processing in one job: split creation, cropping training +
# evaluation, frame cropping, informative-frame classifier training + evaluation,
# and the cleaned-video extraction at the end. All visualizations are produced
# along the way (label plots, training-batch plots of fold 0, evaluation plots).
#
# Prerequisites: annotated frames in cropping/frames_to_annotate_larger/,
# raw informative frames in data/informative_frames/frames_raw_extract/,
# Paxos2025 videos in data/dr_videos/.
#
# Usage: sbatch scripts/preprocess_all.sh
#        EPOCHS_CROP=10 EPOCHS_CLS=5 sbatch scripts/preprocess_all.sh   (shorter runs)

set -e
source ~/.bashrc
conda activate job_new
cd "${SLURM_SUBMIT_DIR:-.}"

EPOCHS_CROP=${EPOCHS_CROP:-20}
EPOCHS_CLS=${EPOCHS_CLS:-15}
FOLDS=${FOLDS:-"0 1 2 3 4"}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "=== 1/8 Split creation (q5_v3-based, patient-level) ==="
python -u -m cropping.make_splits_from_q5v3

echo "=== 2/8 Label distribution plots ==="
python -u -m informative_frames.analyze_labels \
  informative_frames/split_info_5fold_paxos2025_from_q5v3.csv \
  --outdir informative_frames/label_plots

echo "=== 3/8 Cropping (segmentation) training, $EPOCHS_CROP epochs per fold ==="
for fold in $FOLDS; do
  python -u -m cropping.train_segmentation \
    --fold "$fold" --epochs "$EPOCHS_CROP" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --amp
done

echo "=== 4/8 Cropping evaluation (metrics + segmentation_eval.png) ==="
python -u -m cropping.evaluate_segmentation

echo "=== 5/8 Cropping of the labeled informative frames ==="
python -u -m informative_frames.crop_frames

echo "=== 6/8 Informative-frame classifier training, $EPOCHS_CLS epochs per fold ==="
for fold in $FOLDS; do
  python -u -m informative_frames.classify_frames \
    --fold "$fold" --epochs "$EPOCHS_CLS" \
    --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
done

echo "=== 7/8 Classifier evaluation (metrics + classification_eval.png) ==="
python -u -m informative_frames.evaluate_classification

echo "=== 8/8 Cleaned-video extraction (q5_v3 videos only) ==="
python -u -m informative_frames.extract_inf_frames \
  --video_dir data/dr_videos \
  --seg_model_dir cropping/models \
  --cls_model_dir informative_frames/models \
  --out_base data/ensemble_results_paxos2025_q5v3

echo "Pre-processing finished. Outputs:"
echo "  cropping/train_vis/                 training-batch plots (fold 0)"
echo "  cropping/eval/                      cropping metrics + plot"
echo "  informative_frames/label_plots/     label distribution plots"
echo "  informative_frames/eval/            classifier metrics + plot"
echo "  data/ensemble_results_paxos2025_q5v3/    cleaned videos + per-frame probs + plots"
