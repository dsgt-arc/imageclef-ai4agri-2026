# Example User Workspace

This is a template user workspace. Copy this directory to create your own:

```bash
cp -r user/example user/<your-name>
```

## What User Workspaces Are For

Each team member gets their own workspace under `user/<name>/`. This is where you:
- Run experiments independently
- Keep notebooks and scratch code
- Store personal scripts and utilities

## Directory Layout

```
user/<your-name>/
├── README.md           # Workspace overview (this file)
├── experiments/        # Experiment proposals and results
│   └── NNN-name/       # Each experiment gets a numbered directory
│       ├── pyproject.toml  # Experiment dependencies
│       ├── proposal.md     # Hypothesis and method
│       ├── tasks.md        # Implementation checklist
│       ├── results.md      # Observations (after completion)
│       └── artifacts/      # Logs, configs, figures, checkpoints
├── src/                # Shared code across experiments (data loaders, models, utils)
├── notebooks/          # Jupyter notebooks (optional)
└── tests/              # Tests for personal code (optional)
```

## Per-Experiment Dependencies

Each experiment has its own `pyproject.toml` declaring its specific dependencies. This gives better isolation — different experiments can use different library versions without conflict.

Install an experiment's dependencies:

```bash
uv sync --package <experiment-name>
```

The root `pyproject.toml` defines a uv workspace that auto-discovers all experiments under `user/*/experiments/*`.

## Experiment Numbering

Experiments use zero-padded three-digit IDs with descriptive names:

```
000-template/           # This template
001-baseline-model/     # First real experiment
002-data-augmentation/  # Second experiment
003-ensemble-methods/   # Third experiment
```

Use `/experiment:proposal` to create new experiments with proper structure.
