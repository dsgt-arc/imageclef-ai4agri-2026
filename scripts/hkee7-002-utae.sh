#!/bin/bash
#SBATCH --job-name=002-utae
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --qos=embers
#SBATCH --output=artifacts/slurm-%j.out

module load pytorch/2.1.0

# Environment setup — use TMPDIR for fast venv
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/.venv
export XDG_CACHE_HOME=$HOME/scratch/.cache

# Navigate to project and install dependencies
cd ~/ps-clef2026_img_ai4agri-0/imageclef-ai4agri-2026 || exit
uv venv $HOME/scratch/.venv
uv sync --package 002-utae

# Run experiment
uv run user/hkee7/experiments/002-utae/train.py --batch-size 16 --lr 0.001 --num-workers 2
