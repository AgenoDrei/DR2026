# DR detection from smartphone fundus videos

Diabetic-retinopathy grading (0–4) from smartphone-captured fundus videos (Paxos cohorts). Pipeline: raw video →
retina crop + informative-frame filter → DINOv3 features per frame → video-level aggregation (frame classifier
ensemble vs. transformer over a bag of 32 frames). Experiment log with results: `TODO_experiments.md`.

## Setup

- Conda env `job_new` (Python 3.12): `torch`/`torchvision` (CUDA wheels), `opencv-python-headless`,
  `segmentation-models-pytorch`, `albumentations`, `timm`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`,
  `tensorboard`, `colorama`, `click`, `openpyxl`, plus `dino/dinov3/requirements.txt`.
- DINOv3: clone https://github.com/facebookresearch/dinov3 to `dino/dinov3` and put the weights in `dino/`
  (gated download via Meta's license portal): `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L, all DINO models),
  `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` (ConvNeXt models).
- Not in git (shared separately): videos, cleaned clips, stills, model checkpoints (`*.pth`, `*/models/`), TensorBoard `runs/`.
- Run everything from the repo root as modules (`python -m dino.train_dino_transformer ...`).

## Data layout

| Path | Content |
|---|---|
| `data/dr_videos` → `~/Data/2025_Paxos_Videos/videos` | 417 raw Paxos2025 videos (symlink) |
| `data/ensemble_results_paxos2025/cleaned_videos/CLEAN_<video>.mp4` | cropped + filtered Paxos2025 clips (417), what all models read |
| `data/ensemble_results_paxos2020/cleaned_videos/` | cleaned Paxos2020 clips (302; raw videos in `~/Data/2020_Paxos_Videos/data`) |
| `data/2024_Paxos_Frames/cropped_matched_frames/` | cropped clinical stills matched to video frames (frame classifier input) |
| `data/clip_labels_2025.json`, `data/image_labels_2025.json` | Paxos2025 grades per clip (328) / still (610) |
| `data/clip_labels_2020.json` | Paxos2020 grades per video (341) |

Rebuilding from raw videos (needs the `cropping/models` + `informative_frames/models` ensembles):
`scripts/extract_inf_frames_Paxos2025.sh` (clips) → `python data/find_frames.py` + `python -m cropping.segment_frames` (stills)
→ `python data/match_clips.py` → `python data/match_grades.py` (label JSONs) → `data/split_data.py` (below).
Re-extraction is not byte-identical, so rebuilt splits are equivalent, not identical.

## Current datasets

Split dirs hold `split_<n>/{mil,frame}_{train,val}.json` (`{filename: grade}`); pass one with `--split_path <dir>/split_<n>`.

**Paxos2025 (in-domain, 5-fold CV): `data/stratified_splits_paxos2025_patient_q5_v3`** (since 2026-11-12)
- 302 videos / 196 patients; patient-level grade-stratified 5-fold (no patient in both train and val),
  ~242 train / ~60 val videos per fold. Built with
  `python data/split_data.py --datasets 2025 --group_by_patient --min_frames 139 --output_dir data/stratified_splits_paxos2025_patient_q5_v3`.
- Excludes clips with ≤ 139 informative frames (26, `removed_videos.txt`). Repeat recordings of one eye (e.g. `R008R`, `R008R2`)
  are separate videos with the eye's grade and stay in the same fold.
- Grades (all folds): 0: 108, 1: 22, 2: 71, 3: 35, 4: 66.
- `mil_train_mixed.json` = `mil_train.json` + 53 hand-picked Paxos2020 videos (use with `--train_json mil_train_mixed.json --fused_dataset`).
- Results before 2026-11-12 used `_patient_q5` (281 videos, 2026-10-27 – 11-01) or `_patient_q5_v2` (283 videos, repeats merged;
  2026-11-03 – 11-11) and are not directly comparable. Older video-level splits (`stratified_splits_paxos2025`) leak the
  other eye of a patient into val.

**Cross-domain (train Paxos2025, test Paxos2020): `data/stratified_splits_paxos2025_q5_full_xdom`** (since 2026-10-29)
- `split_1`: train = all 281 Paxos2025 `_patient_q5` videos (`mil_train.json`) / 584 stills (`frame_train.json`);
  test = 294 Paxos2020 videos (`mil_val.json` = `mil_val_2020.json`) with sheet sharpness 3 and > 139 frames
  (47 removed, `removed_videos.txt`). Test grades: 0: 212, 1: 23, 2: 48, 3: 7, 4: 4.
- `frame_val.json` is a dummy (inside train). Paxos2020 is only a monitor during training: evaluate `last_*.pth`.
- One pseudo-split per seed (`split_2..N` = symlinks of `split_1`), built by `python -m data.make_xdomain_split --num_seeds N`.
- Train set still comes from `_patient_q5` (pre-v3); not yet rebuilt on v3.

## Scripts (`scripts/`)

Two kinds: `#SBATCH` jobs (`cd scripts && sbatch <job>.sh`, or `bash <job>.sh` on a GPU machine) and local queues
(`train_pgcw.sh`, `train_xdomain.sh`, `train_retfound_transformer.sh`, `train_ssl_lora.sh`) that open one tmux window per run
and need the workstation's `gpu run`/`gpu status` wrapper. Partition names / time limits in `#SBATCH` lines are cluster-specific.
Settings are passed as environment variables (documented at the top of each script).

| Script | Purpose |
|---|---|
| `train_dino_classifier.sh` | frame classifier, one split (`SPLIT=`, `SP=` split dir) |
| `train_dino_transformer.sh` | DINO transformer from a classifier (`CLF=`), frozen or `FT=1` |
| `train_convnext_transformer.sh` | ConvNeXt-L transformer (`MIXED=1`: + Paxos2020 train videos) |
| `train_pgcw.sh` | current 5-split recipe, all models (classifier → DINO frozen/FT, ConvNeXt), local queue |
| `train_xdomain.sh` | cross-domain runs (`SEEDS=`), local queue |
| `vis_video_results.sh` | per-video predictions + metrics/ROC reports (`PGCW=1`, `XDOM=1` modes) |
| `vis_attention.sh`, `vis_classifier_attention.sh` | attention / occlusion plots |
| `extract_inf_frames_Paxos20{25,20}.sh`, `segment*.sh`, `classify_frames.sh`, `cropped_frames.sh`, `cluster_frames_paxos2025.sh` | preprocessing stages |
| `train_adafocus.sh` | Uni-AdaFocus (`adafocus/`): policy-selected frames/patches → frozen DINOv3 ViT-L (`GLOBAL=`, `FT=1`) |
| `train_ssl_lora.sh`, `train_retfound_transformer.sh` | SSL domain adaptation / RETFound backbone experiments |
| `train_dino_transformer_baseline.sh`, `_dense.sh` | historical baseline v2 (old video-level splits) |

Current recipe on Paxos2025 (5 splits) and its evaluation:

```bash
cd scripts
SP=data/stratified_splits_paxos2025_patient_q5_v3 TAG=v3 bash train_pgcw.sh             # local queue
PGCW=1 PTAG=v3 Q5_SPLITS=data/stratified_splits_paxos2025_patient_q5_v3 sbatch vis_video_results.sh
# SLURM, one split at a time:
SPLIT=1 TAG=v3 sbatch train_dino_classifier.sh
CLF=classifier/models/<run>/best_model.pth SPLIT=1 TAG=v3 sbatch train_dino_transformer.sh
```

Cross-domain: `XTAG=xdom SEEDS="42 43 44" bash train_xdomain.sh`, then `XDOM=1 XTAG=xdom XDOM_SEEDS="42 43 44" CROP=1 sbatch vis_video_results.sh`.

Outputs: checkpoints in `classifier/models/`, `transformer/models/`, `convnext/models/`, `adafocus/models/` (`best_*.pth` = reported, `last_*.pth`),
TensorBoard logs in `runs/`, one summary row per run in `results/run_results.csv`.
Metrics: QWK, accuracy, macro F1, referable-DR (grade ≥ 2) recall, ROC-AUC.
