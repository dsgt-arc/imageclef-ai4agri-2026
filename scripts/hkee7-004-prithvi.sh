#!/bin/bash
#SBATCH --job-name=004-prithvi
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --partition=gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --constraint=RTX6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --qos=embers
#SBATCH --output=artifacts/slurm-%j.out

set -e

# ---------------------------------------------------------------------------
# Environment — redirect caches and use fast local SSD for the venv
# ---------------------------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/.venv"
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Sync dependencies from repo root
# ---------------------------------------------------------------------------
cd ~/ps-clef2026_img_ai4agri-0/imageclef-ai4agri-2026 || exit
uv sync --package 004-prithvi-finetune

# ---------------------------------------------------------------------------
# Train  (single-frame strategy: T=1, batch=32)
# ---------------------------------------------------------------------------
RUN_DIR=/storage/project/ps-clef2026_img_ai4agri-0/hkee7/imageclef-ai4agri-2026/runs/prithvi-single-frame

uv run user/hkee7/experiments/004-prithvi-finetune/train.py \
    --save-dir "$RUN_DIR"

# ---------------------------------------------------------------------------
# Predict  (average probs over all 34 frames per patch)
# ---------------------------------------------------------------------------
BEST_CKPT=$(ls -t "$RUN_DIR"/best_*.ckpt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "No best checkpoint found in $RUN_DIR — skipping predict"
    exit 0
fi

echo "Best checkpoint: $BEST_CKPT"
uv run user/hkee7/experiments/004-prithvi-finetune/predict.py \
    --checkpoint "$BEST_CKPT" \
    --output-dir "$RUN_DIR/submission" \
    --split test \
    --batch-size 64
