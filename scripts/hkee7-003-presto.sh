#!/bin/bash
#SBATCH --job-name=003-presto-finetune
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --qos=inferno
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
uv sync --package 003-presto-finetune

# Run experiment
uv run user/hkee7/experiments/003-presto-finetune/train.py --stage 2 --batch-size 64 --chunk-size 32768 --num-workers 6 --ckpt artifacts/best_stage1.pt