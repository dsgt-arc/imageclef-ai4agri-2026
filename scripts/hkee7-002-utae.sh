#!/bin/bash
#SBATCH --job-name=002-utae
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-rtxpro-blackwell
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --qos=inferno
#SBATCH --output=artifacts/slurm-%j.out

module load pytorch/2.1.0

# Environment setup — use TMPDIR for fast venv
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/.venv
export XDG_CACHE_HOME=$HOME/scratch/.cache

# Navigate to project and install dependencies
cd ~/ps-clef2026_img_ai4agri-0/imageclef-ai4agri-2026 || exit
uv sync --package 002-utae

# Run experiment
uv run user/hkee7/experiments/002-utae/train.py --mode ordinal --batch-size 32 --lr 0.001 --num-workers 6 --epochs 100
