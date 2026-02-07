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
├── pyproject.toml      # Your dependencies (required)
├── README.md           # Workspace overview (this file)
├── experiments/        # Experiment proposals and results
│   └── NNN-name/       # Each experiment gets a numbered directory
│       ├── proposal.md # Hypothesis and method
│       ├── tasks.md    # Implementation checklist
│       ├── results.md  # Observations (after completion)
│       └── artifacts/  # Logs, configs, figures, checkpoints
├── src/                # Personal code modules (optional)
├── notebooks/          # Jupyter notebooks (optional)
└── tests/              # Tests for personal code (optional)
```

## Your pyproject.toml

Each user workspace has its own `pyproject.toml` with experiment-specific dependencies. This example includes a starter one — edit it to add the packages you need.

After copying this workspace, install your dependencies:

```bash
uv sync --package user-<your-name>
```

## Experiment Numbering

Experiments use zero-padded three-digit IDs with descriptive names:

```
000-template/           # This template
001-baseline-model/     # First real experiment
002-data-augmentation/  # Second experiment
003-ensemble-methods/   # Third experiment
```

Use `/experiment:proposal` to create new experiments with proper structure.
