# PACE Cluster Setup

How to set up and use the Georgia Tech PACE cluster for this project.

## Scratch Directory Structure

PACE has limited home directory quota. Use scratch storage for large files:

```bash
# Shared team scratch
/storage/ice-shared/dsgt/

# Your personal scratch (create if needed)
SCRATCH=/storage/ice-shared/dsgt/$USER
mkdir -p $SCRATCH
```

Organize your scratch directory:

```
$SCRATCH/
├── .venv/          # Python virtual environment
├── data/           # Datasets
├── checkpoints/    # Model checkpoints
├── hf_cache/       # HuggingFace cache
└── torch_cache/    # PyTorch cache
```

## Environment Variables

Add these to your `~/.bashrc` or job scripts:

```bash
# Point uv's venv to scratch (not home)
export UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv

# Cache directories on scratch
export HF_HOME=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache

# Optional: faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## uv Setup on PACE

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv on scratch (NOT home directory)
cd /path/to/project
UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv uv venv $SCRATCH/.venv

# Install dependencies
UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv uv sync
```

## sbatch Script Patterns

### Basic GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/slurm-%j.out

# Environment setup
SCRATCH=/storage/ice-shared/dsgt/$USER
export UV_PROJECT_ENVIRONMENT=$SCRATCH/.venv
export HF_HOME=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache

# Activate environment
source $SCRATCH/.venv/bin/activate

# Run experiment
cd /path/to/project
uv run python -m my_module.train --config config.yaml
```

### Using `uv run` in Jobs

`uv run` automatically uses the project's virtual environment:

```bash
# Run a script
uv run python train.py

# Run a module
uv run python -m my_module.experiment

# Run pytest
uv run pytest tests/
```

## Download Patterns

### HuggingFace Models

```bash
# Using huggingface_hub (respects HF_HOME)
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('model-org/model-name', cache_dir='$SCRATCH/hf_cache')
"
```

### Large Files with aria2c

```bash
# aria2c supports parallel downloads
module load aria2
aria2c -x 16 -s 16 -d $SCRATCH/data/ <url>
```

## Tips

- **Never install packages to home directory** - quota is limited
- **Use `tmux` or `screen`** for long-running interactive sessions
- **Check disk usage**: `du -sh $SCRATCH/*`
- **Monitor jobs**: `squeue -u $USER`
- **Cancel jobs**: `scancel <job-id>`
