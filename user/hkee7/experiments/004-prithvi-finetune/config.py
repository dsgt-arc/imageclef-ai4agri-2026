import os
from dataclasses import dataclass

@dataclass
class Config:
    # Environment
    device: str = "cuda"
    seed: int = 42

    # Paths
    data_path: str = "data/precomputed_tensors"
    metadata_path: str = "data/agripotential/metadata.csv"
    save_dir: str = "artifacts"

    # Prithvi backbone name in terratorch registry
    backbone: str = "prithvi_eo_v2_300"

    # Number of input timesteps (must match dataset; 11 for AgriPotential)
    num_frames: int = 11

    # Optimization
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 50
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True

    # Dataloader — num_workers=0 avoids IPC socket crashes on PACE
    num_workers: int = 4
    pin_memory: bool = False
    augment: bool = True

    num_classes: int = 5
    ignore_index: int = 0
