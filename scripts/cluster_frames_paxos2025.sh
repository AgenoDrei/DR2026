#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate job_new
cd ..
python -u -m clustering.extract_features
python -u -m clustering.train_contrastive
python -u -m clustering.cluster_frames
