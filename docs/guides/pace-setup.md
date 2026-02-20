# PACE Cluster Setup

How to set up and use the Georgia Tech PACE Phoenix cluster for this project.

## Login Node vs Compute Node

**The login node is only for requesting resources.** Do not run experiments, install packages, or do development work on login nodes. Instead:

1. SSH into PACE (lands you on a login node)
2. Request compute resources with `salloc` (see [Requesting Resources](#requesting-resources))
3. Once allocated, SSH into your assigned compute node from a new terminal
4. Do all work (development, training, package installs) on the compute node

## Scratch Directory

PACE has limited home directory quota. Use your personal scratch for large files:

```bash
# Your personal scratch directory
~/scratch    # resolves to /storage/home/hcoda1/<group>/<user>/scratch
```

Organize your scratch directory:

```
~/scratch/
├── .cache/         # Redirected XDG cache (pip, uv, huggingface, etc.)
├── .venv/          # Python virtual environment (persistent option)
├── data/           # Datasets
├── checkpoints/    # Model checkpoints
└── ollama_models/  # Ollama models (if applicable)
```

> **Note:** Files on scratch are deleted after 90 days of inactivity.

## Recommended `.bashrc` Additions

Add these to your `~/.bashrc` so they apply on every login and compute node session:

```bash
# Load Python module and set include path
module load python/3.10
PYTHON_ROOT=$(python -c 'import sys; print(sys.base_prefix)')
export CPATH="${PYTHON_ROOT}/include/python3.10:${CPATH:-}"

# Redirect all caches to scratch (keeps home directory clean)
export XDG_CACHE_HOME="$HOME/scratch/.cache"

# Optional: redirect Ollama models to scratch
# export OLLAMA_MODELS=$HOME/scratch/ollama_models

# Optional: faster HuggingFace downloads
# export HF_HUB_ENABLE_HF_TRANSFER=1
```

Setting `XDG_CACHE_HOME` redirects caches for uv, pip, HuggingFace, and other tools to scratch in one line — no need to set `HF_HOME`, `TORCH_HOME`, etc. individually.

## Requesting Resources

Use `salloc` from the login node to request an interactive compute allocation:

```bash
# Basic CPU allocation (1 hour, free tier)
salloc --account=<your-account> --time=01:00:00 --qos=embers

# GPU allocation (2 hours, RTX6000)
salloc --account=<your-account> --time=02:00:00 --qos=embers \
    --gres=gpu:1 --constraint=RTX6000

# Longer session on paid QoS (higher priority, no preemption)
salloc --account=<your-account-paid> --time=08:00:00 --qos=inferno \
    --gres=gpu:A40:1
```

**QoS options:**
- **embers** — Free backfill, lower priority, max 8 hours, may be preempted after 1 hour
- **inferno** — Paid, higher priority, no preemption, max 3 days (GPU) or 21 days (CPU)

Once allocated, note the hostname and SSH to it from a new terminal:

```bash
ssh <allocated-hostname>
```

## Virtual Environment Setup

There are two strategies for where to place your `.venv`. We recommend **TMPDIR** for faster performance, but scratch works too.

### Option A: TMPDIR (Recommended)

The compute node's local temporary storage (`$TMPDIR`) is fast SSD-backed local disk. Installing your `.venv` here gives significantly faster package imports and script startup.

**Trade-off:** The `.venv` is ephemeral — you must recreate it every time you get a new compute allocation.

```bash
# On a compute node — set venv to TMPDIR
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/.venv

# Navigate to your project
cd ~/clef/<your-project>

# Create venv and install dependencies (fast with uv)
uv venv ${TMPDIR}/.venv
uv sync --package <experiment-name>
```

Add this to your workflow: every time you SSH into a new compute node, run the above before starting work. With uv's speed and cached packages on scratch (`XDG_CACHE_HOME`), this typically takes seconds.

### Option B: Scratch (Persistent)

Placing the `.venv` on scratch means it persists across sessions (until 90 days of inactivity). This avoids reinstalling each time but is slower due to network filesystem latency.

```bash
# Set venv to scratch
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/.venv

# Navigate to your project
cd ~/clef/<your-project>

# Create venv and install dependencies (one-time)
uv venv $HOME/scratch/.venv
uv sync --package <experiment-name>
```

### Quick Comparison

| | TMPDIR | Scratch |
|---|---|---|
| **Speed** | Fast (local SSD) | Slower (network filesystem) |
| **Persistence** | Gone when allocation ends | Survives across sessions |
| **Setup** | Reinstall each allocation | One-time install |
| **Best for** | Training, development | Quick iterations between sessions |

## uv Setup on PACE

```bash
# Install uv (one-time, from login or compute node)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then set UV_PROJECT_ENVIRONMENT per your chosen strategy above
# and run uv sync from your project directory
```

## sbatch Script Patterns

### Basic GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=embers
#SBATCH --output=artifacts/slurm-%j.out

# Environment setup — use TMPDIR for fast venv
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/.venv
export XDG_CACHE_HOME=$HOME/scratch/.cache

# Navigate to project and install dependencies
cd ~/clef/<your-project>
uv venv ${TMPDIR}/.venv
uv sync --package <experiment-name>

# Run experiment
uv run python -m my_module.train --config config.yaml
```

### Using `uv run` in Jobs

`uv run` automatically uses the environment pointed to by `UV_PROJECT_ENVIRONMENT`:

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
# Downloads are cached via XDG_CACHE_HOME on scratch
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('model-org/model-name')
"
```

### Large Files with aria2c

```bash
# aria2c supports parallel downloads
module load aria2
aria2c -x 16 -s 16 -d ~/scratch/data/ <url>
```

## Tips

- **Never install packages to home directory** — quota is limited
- **Never run experiments on login nodes** — request compute resources first
- **Use `tmux` or `screen`** for long-running interactive sessions on compute nodes
- **Check disk usage**: `du -sh ~/scratch/*`
- **Check account balance**: `pace-quota`
- **Monitor jobs**: `squeue -u $USER`
- **Cancel jobs**: `scancel <job-id>`
