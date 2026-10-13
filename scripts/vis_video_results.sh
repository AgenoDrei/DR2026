#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate job_new
cd ..

CLASSIFIER_PATHS_BINARY=(
"classifier/models/dino_binary_split_1_0906_2310_full/best_model.pth"
"classifier/models/dino_binary_split_2_0906_2340_full/best_model.pth"
"classifier/models/dino_binary_split_3_0907_0010_full/best_model.pth"
"classifier/models/dino_binary_split_4_0906_2208_full/best_model.pth"
"classifier/models/dino_binary_split_5_0906_2239_full/best_model.pth"
)

CLASSIFIER_PATHS_ALL=(
"classifier/models/dino_all_split_1_0906_1934_full/best_model.pth"
"classifier/models/dino_all_split_2_0906_2005_full/best_model.pth"
"classifier/models/dino_all_split_3_0906_2036_full/best_model.pth"
"classifier/models/dino_all_split_4_0906_2107_full/best_model.pth"
"classifier/models/dino_all_split_5_0906_2138_full/best_model.pth"
)

TRANSFORMER_PATHS_BINARY=(
"transformer/models/dino_sa_binary_split_1_0907_1613_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_2_0907_1524_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_3_0907_1500_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_4_0907_1442_full/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_5_0907_1352_full/best_sa_model.pth"
)

TRANSFORMER_PATHS_BINARY_2L=(
"transformer/models/dino_sa_binary_split_1_0925_1835_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_2_0925_1901_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_3_0925_2006_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_4_0925_2133_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_binary_split_5_0926_0135_full_2blocks/best_sa_model.pth"
)

TRANSFORMER_PATHS_ALL=(
"transformer/models/dino_sa_all_split_1_0907_1147_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_2_0907_1203_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_3_0907_1224_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_4_0907_1314_full/best_sa_model.pth"
"transformer/models/dino_sa_all_split_5_0907_1330_full/best_sa_model.pth"
)

TRANSFORMER_PATHS_ALL_2L=(
"transformer/models/dino_sa_all_split_1_0926_1618_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_2_0926_1856_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_3_0926_2123_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_4_0928_1849_full_2blocks/best_sa_model.pth"
"transformer/models/dino_sa_all_split_5_0928_2110_full_2blocks/best_sa_model.pth"
)

#python -u -m dino.vis_video_results \
#  --dataset_name "2025" \
#  --annotations_path "data/stratified_splits" \
#  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
#  --output_dir "data/multi_eval_paxos2025_binary_new" \
#  --model Clf classifier "${CLASSIFIER_PATHS_BINARY[@]}" \
#  --model Clf32  classifier_video "${CLASSIFIER_PATHS_BINARY[@]}" \
#  --model Trans_1L transformer "${TRANSFORMER_PATHS_BINARY[@]}" \
#  --model Trans_2L transformer "${TRANSFORMER_PATHS_BINARY_2L[@]}" \
#  --img_size 512 \
#  --complex_augs \
#  --binary_classification

python -u -m dino.vis_video_results \
  --dataset_name "2025" \
  --annotations_path "data/stratified_splits" \
  --videos_path "data/ensemble_results_paxos2025/cleaned_videos" \
  --output_dir "data/multi_eval_paxos2025_all_2" \
  --model Clf classifier "${CLASSIFIER_PATHS_ALL[@]}" \
  --model Clf32  classifier_video "${CLASSIFIER_PATHS_ALL[@]}" \
  --model Trans_1L transformer "${TRANSFORMER_PATHS_ALL[@]}" \
  --model Trans_2L transformer "${TRANSFORMER_PATHS_ALL_2L[@]}" \
  --img_size 512 \
  --complex_augs
