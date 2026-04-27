"""
Hyperparameter configuration for the stacked-channel 2D UNet experiment.

Designed to match the AgriPotential organiser baseline (El Sakka et al., 2025):
  - All T timesteps × 10 bands stacked as (T*C, H, W) input channels
  - AdamW optimiser, lr=1e-5 (as in the supplement), no scheduler
  - Ordinal BCE loss (best label representation per the supplement)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # ---- Data ---------------------------------------------------------------
    data_path: str = "data/precomputed_tensors"
    metadata_path: str = "data/agripotential/metadata.csv"
    num_bands: int = 10

    # ---- Model --------------------------------------------------------------
    mode: str = "ordinal"         # "regression", "classification", or "ordinal"
    num_classes: int = 5          # classes 1–5 (label 0 = unlabelled/ignore)
    base_channels: int = 128      # 2× organiser baseline — more capacity
    depth: int = 4                # number of UNet levels (128→256→512→1024)

    # ---- Training -----------------------------------------------------------
    # Cosine annealing over 500 epochs with a higher LR — converges well and
    # generalises better than flat 1e-5 for longer runs.
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    cosine_t0: int = 500          # single cosine cycle over full training
    cosine_t_mult: int = 1
    step_gamma: float = 0.5
    step_size: int = 50
    num_workers: int = 4
    pin_memory: bool = True

    # ---- Loss ---------------------------------------------------------------
    loss_fn: str = "smooth_l1"    # for regression mode only
    smooth_l1_beta: float = 1.0
    ignore_index: int = 0         # unlabelled pixels

    # ---- Normalisation ------------------------------------------------------
    stats_path: str = "stats.pt"  # output of compute_stats.py (per-band mean/std)

    # ---- Augmentation -------------------------------------------------------
    augment: bool = True          # flips + 90° rotations

    # ---- Misc ---------------------------------------------------------------
    seed: int = 42
    save_dir: str = "artifacts"
    log_every: int = 20
    device: str = "cuda"
    use_amp: bool = True
