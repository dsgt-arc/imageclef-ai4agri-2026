---
name: experiment
description: Use this skill when the user wants to create, run, validate, or document experiments. This includes proposing new experiments with a hypothesis, setting up experiment directories, tracking experiment tasks, recording results, or checking experiment completeness. Activate when the user mentions experiments, hypotheses, baselines, ablations, or evaluation metrics.
metadata:
  author: dsgt-arc
  version: "1.0"
---

# Experiment Workflow

## Overview

Experiments follow a structured lifecycle: propose, approve, run, document results. Each experiment lives in its own directory under the user's workspace with its own `pyproject.toml` for dependency isolation.

## Directory Structure

```
user/<name>/
├── src/                # Shared code across experiments (data loaders, models, utils)
└── experiments/
    └── NNN-descriptive-name/
        ├── pyproject.toml   # Experiment dependencies (required)
        ├── proposal.md      # Hypothesis and method (required)
        ├── tasks.md         # Implementation checklist (required)
        ├── design.md        # Technical decisions (optional)
        ├── results.md       # Observations (added after completion)
        └── artifacts/       # Logs, configs, figures, checkpoints
```

**Note:** `src/` lives at the user level, not the experiment level. Experiments share common code (data loaders, training utilities, model definitions) through the user's `src/` package. Experiment directories contain only the experiment-specific configuration, proposals, and artifacts.

## Status Lifecycle

```
proposed → approved → in-progress → completed
    ↓         ↓           ↓
 (abandoned) (abandoned) (abandoned)
```

## Creating an Experiment

### 1. Determine Location

- User workspaces: `user/<name>/experiments/`
- Ask the user for their name if not known
- List existing experiments to find the next ID
- Use zero-padded three-digit IDs: `001-baseline-model`, `002-data-augmentation`

### 2. Create Directory and Files

```bash
mkdir -p user/<name>/experiments/NNN-descriptive-name/artifacts
```

### 3. Create pyproject.toml

Copy from `user/example/experiments/000-template/pyproject.toml` and update:
- `name` — match the experiment directory name (e.g., `001-baseline-model`)
- `description` — brief description of the experiment
- `authors` — the user's name and email
- `dependencies` — experiment-specific packages

The root `pyproject.toml` auto-discovers experiments via workspace members glob.

### 4. Create proposal.md

Use template at [docs/_templates/experiment-proposal.md](../../docs/_templates/experiment-proposal.md). Must include:
- **Status**: `proposed`
- **Hypothesis**: Specific and falsifiable
- **Method**: Approach, setup, procedure, variables
- **Evaluation**: Metrics, baseline, success/rejection criteria
- **Limitations**: Assumptions and constraints

### 5. Create tasks.md

Use template at [docs/_templates/experiment-tasks.md](../../docs/_templates/experiment-tasks.md). Break into phases:
- Setup (environment, data, baseline)
- Implementation (core logic, logging)
- Execution (run, collect metrics)
- Analysis (evaluate, document)

Use checkbox format: `- [ ] 1.1 Task description`

### 6. Optional: Create design.md

Use template at [docs/_templates/experiment-design.md](../../docs/_templates/experiment-design.md) when:
- Multiple implementation approaches need comparison
- Architecture decisions should be captured
- Non-trivial technical setup is required

## Validating an Experiment

Check that an experiment has all required components:

1. **Structure**: `pyproject.toml`, `proposal.md`, `tasks.md`, `artifacts/` all exist
2. **Proposal completeness**: Status field, non-empty hypothesis, method with approach/setup/procedure, evaluation with metrics and success criteria
3. **Tasks format**: At least one group, checkbox format, numbered tasks
4. **Status-specific**:
   - `in-progress`: at least one task checked
   - `completed`: all tasks checked, `results.md` exists

## Recording Results

Use template at [docs/_templates/experiment-result.md](../../docs/_templates/experiment-result.md). Include:
- Summary of what happened
- Key metrics vs. success criteria
- Confirm or reject the hypothesis
- Lessons learned and next steps

Update the proposal status to `completed`.

## Installing Experiment Dependencies

```bash
# From anywhere in the repo
uv sync --package <experiment-name>

# Or cd into the experiment directory
cd user/<name>/experiments/NNN-name/
uv sync
```

## Guidelines

- Every experiment must have a falsifiable hypothesis
- Define success/rejection criteria before running
- Keep experiments focused — test one thing at a time
- Link back to the concept that motivated the experiment
- Track progress by checking off tasks in `tasks.md`
