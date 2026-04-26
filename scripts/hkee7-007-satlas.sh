#!/bin/bash
#SBATCH --job-name=007-satlas-finetune
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --partition=gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --qos=embers
#SBATCH --output=artifacts/slurm-%j.out

set -e

module load pytorch/2.1.0

# Environment setup — use TMPDIR for fast venv
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/.venv
export XDG_CACHE_HOME=$HOME/scratch/.cache
export HF_HOME="$HOME/scratch/.cache/huggingface"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Navigate to project and install dependencies
cd ~/ps-clef2026_img_ai4agri-0/imageclef-ai4agri-2026 || exit
uv sync --package 007-satlas-finetune

# Run experiment
uv run user/hkee7/experiments/007-satlas-finetune/train.py --save-dir /storage/project/ps-clef2026_img_ai4agri-0/hkee7/imageclef-ai4agri-2026/runs/satlas-v2-znorm
