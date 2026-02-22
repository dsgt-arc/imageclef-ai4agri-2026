---
name: pace-cluster
description: Use this skill when the user needs help with the Georgia Tech PACE Phoenix cluster. This includes writing sbatch or salloc commands, requesting GPUs, choosing QoS policies (embers vs inferno), setting up virtual environments on compute nodes, writing job scripts, debugging pending or failed jobs, understanding cluster storage (scratch, TMPDIR, home), or any Slurm-related task.
metadata:
  author: dsgt-arc
  version: "1.0"
---

# PACE Phoenix Cluster

## Overview

This project runs on the Georgia Tech PACE Phoenix cluster using Slurm for job scheduling. There are two key reference files you should read when helping with PACE tasks:

1. **Project setup guide**: [docs/guides/pace-setup.md](../../docs/guides/pace-setup.md) — Project-specific configuration (venv strategies, .bashrc, salloc patterns)
2. **Slurm reference**: [docs/vendor/pace-phoenix.md](../../docs/vendor/pace-phoenix.md) — Comprehensive Slurm reference (GPU types, QoS policies, job types, troubleshooting)

Read the relevant file before answering PACE questions.

## Key Concepts

### Login Node vs Compute Node

- Login nodes are **only for requesting resources** via `salloc` or `sbatch`
- All development, training, and package installation happens on compute nodes
- After `salloc` grants an allocation, SSH to the compute node from a new terminal

### Storage Hierarchy

| Location | Speed | Persistence | Use For |
|----------|-------|-------------|---------|
| `$HOME` | Moderate | Permanent | Config files, small scripts (limited quota) |
| `~/scratch` | Moderate | 90 days inactivity | Data, checkpoints, caches |
| `$TMPDIR` | Fast (local SSD) | Gone when job ends | Virtual environments, temp files |

### Virtual Environment Strategy

Recommend `$TMPDIR` for `.venv` (fast but must reinstall each allocation):

```bash
export UV_PROJECT_ENVIRONMENT=${TMPDIR}/.venv
uv venv ${TMPDIR}/.venv
uv sync --package <experiment-name>
```

Packages are cached on scratch via `XDG_CACHE_HOME`, so reinstalls are fast.

### QoS Quick Reference

| QoS | Cost | Priority | Max Time | Preemption |
|-----|------|----------|----------|------------|
| **embers** | Free | Low | 8 hours | After 1 hour |
| **inferno** | Paid | High | 3 days GPU / 21 days CPU | No |

### Available GPUs

| GPU | VRAM | Cores/GPU | GRES Flag |
|-----|------|-----------|-----------|
| V100 | 16/32 GB | 12 | `--gres=gpu:V100:N` |
| RTX 6000 | 24 GB | 6 | `--gres=gpu:RTX_6000:N` |
| A100 | 40/80 GB | 8-32 | `--gres=gpu:A100:N` |
| H100 | 80 GB | 8 | `--gres=gpu:H100:N` |
| H200 | 142 GB | 8 | `--gres=gpu:H200:N` |
| L40S | 48 GB | 4 | `--gres=gpu:L40S:N` |

## Common Patterns

### Interactive GPU Session

```bash
salloc --account=<account> --time=02:00:00 --qos=embers \
    --gres=gpu:RTX_6000:1 --mem-per-gpu=20G
```

### Batch Job Template

```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --account=<account>
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=embers
#SBATCH --output=artifacts/slurm-%j.out

export UV_PROJECT_ENVIRONMENT=${TMPDIR}/.venv
export XDG_CACHE_HOME=$HOME/scratch/.cache

cd ~/clef/<project>
uv venv ${TMPDIR}/.venv
uv sync --package <experiment-name>

uv run python -m my_module.train --config config.yaml
```

## When Helping Users

1. Always check which QoS is appropriate (embers for dev, inferno for production)
2. Always include `--account` in salloc/sbatch commands
3. For GPU jobs, use `--mem-per-gpu` instead of `--mem-per-cpu`
4. Remind users to copy results off `$TMPDIR` before job ends
5. For detailed Slurm flags, read [docs/vendor/pace-phoenix.md](../../docs/vendor/pace-phoenix.md)
