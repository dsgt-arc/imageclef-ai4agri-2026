"""
Hyperparameter configuration for the Presto fine-tune viticulture experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # ---- Data ---------------------------------------------------------------
    data_path: str = "data/precomputed_tensors"
    metadata_path: str = "data/agripotential/metadata.csv"
    num_bands: int = 10  # full Sentinel-2 stack from precomputed tensors

    # ---- Model --------------------------------------------------------------
    num_classes: int = 5  # ordinal classes 1–5
    head_hidden_dim: int = 64  # Conv2d hidden channels in spatial head
    freeze_encoder: bool = True  # Stage 1: frozen; Stage 2: False

    # ---- Training -----------------------------------------------------------
    # Stage 1 (frozen encoder)
    stage1_epochs: int = 10
    stage1_lr: float = 1e-3

    # Stage 2 (full fine-tune)
    stage2_epochs: int = 40
    stage2_lr: float = 5e-5  # lower LR for encoder weights
    stage2_head_lr: float = 5e-4  # higher LR for head

    batch_size: int = 128  # Increased for massive VRAM (90GB Blackwell)
    chunk_size: int = 16384  # max safe value before CUDA grid limit errors
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    num_workers: int = 4
    pin_memory: bool = True

    # ---- Loss ---------------------------------------------------------------
    ignore_index: int = 0  # unlabelled pixels

    # ---- Augmentation -------------------------------------------------------
    augment: bool = False

    # ---- Misc ---------------------------------------------------------------
    seed: int = 42
    save_dir: str = "artifacts"
    log_every: int = 20
    device: str = "cuda"
    use_amp: bool = True

    # Presto pre-trained weights (downloaded automatically by Presto.load_pretrained())
    presto_path: str | None = None  # None = use default cached weights
