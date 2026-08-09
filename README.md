# Predicting Viticulture Potential through an Ensemble of U-Net and a Geospatial Foundation Model

Determining agricultural potential is fundamental to sustainable land management and agricultural planning. Remote sensing data is increasingly valuable as an avenue for agricultural potential due to the cost of traditional methods (surveys, in-situ measurements, soil testing, etc).

ImageCLEF AI4Agri 2026: Subtask 1 is concerned with the prediction of viticulture potential in Southern France. This repository contains the implementation and experiment code for Georgia Tech's DS@GT ARC submission, which introduces an ensemble of U-Net and a Geospatial Foundation Model (Prithvi-2.0). Full methodology and analysis are available in the [working notes](https://arxiv.org/abs/2607.08449).

Trained checkpoints for the submission are available on Hugging Face at https://huggingface.co/PerezIgnacio/dsgt-arc-imageclef-ai4agri-2026.

## Project structure

```text
.
├── ai4agri/               # Shared package and utilities
├── notebooks/             # Exploratory notebooks
├── scripts/               # Utility scripts for experiments
├── src/                   # Final submission code and reusable source files
│   ├── utils/             # Shared training/evaluation helpers
│   ├── unet.py            # U-Net model definition
│   ├── prithvi.py         # Prithvi-based training utilities
│   ├── test.py            # Ensemble evaluation and submission generation
│   ├── train-unet.py      # U-Net training
│   ├── train-prithvi.py   # Prithvi training
│   └── preprocessing.py   # Preprocessing
├── skills/                # Agent skills used for experiment workflows
├── user/                  # User-specific experiment workspaces
│   ├── hkee7/
│   ├── lrassbach3/
|   └── perezIgnacio/
├── pyproject.toml         # Workspace configuration
├── AGENTS.md              # Repository conventions
└── README.md              # Project overview
```

## Setup and usage

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a virtual environment

```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install dependencies for an experiment

```bash
uv sync --package <experiment-name>
```

### 4. Run an experiment

Navigate to the relevant experiment directory under [user](user) and follow the instructions in that workspace. Most experiments are organized as self-contained directories with their own configuration and scripts.

## Agent skills

The repository keeps the relevant agent skills for experiment-oriented workflows:

- `experiment` — create and document experiments
