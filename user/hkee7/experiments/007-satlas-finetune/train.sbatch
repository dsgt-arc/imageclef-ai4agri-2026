#!/bin/bash
#SBATCH --job-name=satlas-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --partition=gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -e

echo "Starting job on $(hostname)"

# ---------------------------------------------------------------------------
# PACE Phoenix best practices
# ---------------------------------------------------------------------------
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Fast local SSD venv — must reinstall each allocation but imports are 5× faster
export UV_PROJECT_ENVIRONMENT="$TMPDIR/.venv"

echo "Syncing uv environment to local SSD ($UV_PROJECT_ENVIRONMENT)..."
cd ../../../
uv sync --package 007-satlas-finetune

cd user/hkee7/experiments/007-satlas-finetune

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
echo "Training Swin-V2-B + FPN (SatLas/ImageNet-22k)..."
srun uv run python train.py \
    --save-dir /storage/project/ps-clef2026_img_ai4agri-0/hkee7/imageclef-ai4agri-2026/runs/satlas-v3-stacked

echo "Training completed."
