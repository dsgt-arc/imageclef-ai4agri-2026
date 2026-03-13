"""
Hyperparameter configuration for the U-TAE viticulture experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Data ---------------------------------------------------------------
    data_path: str = "data/precomputed_tensors"
    metadata_path: str = "data/agripotential/metadata.csv"
    num_bands: int = 10

    # ---- Model --------------------------------------------------------------
    mode: str = "ordinal"  # "regression", "classification", or "ordinal"
    num_classes: int = 5  # classes 1–5 (label 0 = unlabelled/ignore)
    encoder_widths: list[int] = field(default_factory=lambda: [64, 64, 64, 128])
    decoder_widths: list[int] = field(default_factory=lambda: [32, 32, 64, 128])
    n_head: int = 16
    d_model: int = 256
    d_k: int = 4

    # ---- Training -----------------------------------------------------------
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" or "step"
    cosine_t0: int = 100  # CosineAnnealingWarmRestarts: first cycle length
    cosine_t_mult: int = 1  # cycle length multiplier after each restart
    step_gamma: float = 0.5
    step_size: int = 15
    num_workers: int = 4
    pin_memory: bool = True

    # ---- Loss ---------------------------------------------------------------
    # For regression mode:
    loss_fn: str = "smooth_l1"  # "smooth_l1" or "mse"
    smooth_l1_beta: float = 1.0
    # For classification mode:
    ignore_index: int = 0  # unlabelled pixels

    # ---- Augmentation -------------------------------------------------------
    augment: bool = False  # spatial augmentation (flips + rotations) during training

    # ---- Misc ---------------------------------------------------------------
    seed: int = 42
    save_dir: str = "artifacts"
    log_every: int = 20  # batches
    device: str = "cuda"
    use_amp: bool = True  # mixed precision
