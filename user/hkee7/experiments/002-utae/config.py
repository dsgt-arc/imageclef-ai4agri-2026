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
    mode: str | None = None
    num_bands: int = 10

    # ---- Model --------------------------------------------------------------
    encoder_widths: list[int] = field(default_factory=lambda: [64, 64, 64, 128])
    decoder_widths: list[int] = field(default_factory=lambda: [32, 32, 64, 128])
    n_head: int = 16
    d_model: int = 256
    d_k: int = 4

    # ---- Training -----------------------------------------------------------
    epochs: int = 50
    batch_size: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" or "step"
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

    # ---- Misc ---------------------------------------------------------------
    seed: int = 42
    save_dir: str = "artifacts"
    log_every: int = 20  # batches
    device: str = "cuda"
    use_amp: bool = True  # mixed precision
