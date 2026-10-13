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

TRANSFORMER_PATHS=(
"transformer/models/dino_sa_binary_split_1_0925_1835_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_binary_split_2_0925_1901_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_binary_split_3_0925_2006_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_binary_split_4_0925_2133_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_binary_split_5_0926_0135_full_2blocks/best_sa_model.pth"
)

TRANSFORMER_PATHS_ALL=(
"transformer/models/dino_sa_all_split_1_0926_1618_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_all_split_2_0926_1856_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_all_split_3_0926_2123_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_all_split_4_0928_1849_full_2blocks/best_sa_model.pth"
#"transformer/models/dino_sa_all_split_5_0928_2110_full_2blocks/best_sa_model.pth"
)

python -u -m dino.vis_attention \
  --trans_paths "${TRANSFORMER_PATHS[@]}" \
  --splits 1 \
  --complex_augs \
  --output_dir 'binary_transformer_attention_analysis_new_L2'

python -u -m dino.vis_attention \
  --trans_paths "${TRANSFORMER_PATHS_ALL[@]}" \
  --splits 1 \
  --complex_augs  \
  --output_dir 'all_transformer_attention_analysis_new_L2'
