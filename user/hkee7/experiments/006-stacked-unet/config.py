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
    base_channels: int = 64       # first encoder level width
    depth: int = 4                # number of UNet levels

    # ---- Training -----------------------------------------------------------
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-5              # matches organiser supplement
    weight_decay: float = 1e-4
    scheduler: str = "none"       # "cosine", "step", or "none" (organiser used none)
    cosine_t0: int = 300
    cosine_t_mult: int = 1
    step_gamma: float = 0.5
    step_size: int = 50
    num_workers: int = 4
    pin_memory: bool = True

    # ---- Loss ---------------------------------------------------------------
    loss_fn: str = "smooth_l1"    # for regression mode only
    smooth_l1_beta: float = 1.0
    ignore_index: int = 0         # unlabelled pixels

    # ---- Augmentation -------------------------------------------------------
    augment: bool = False

    # ---- Misc ---------------------------------------------------------------
    seed: int = 42
    save_dir: str = "artifacts"
    log_every: int = 20
    device: str = "cuda"
    use_amp: bool = True
