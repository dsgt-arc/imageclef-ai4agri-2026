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

    # Total timesteps in the dataset tensors
    num_frames_data: int = 34

    # Timesteps fed to the backbone (uniformly subsampled from num_frames_data).
    # Attention memory scales as O(T²): 34 frames → 2177 tokens → ~14 GB just for
    # backbone intermediates across 24 layers.  12 frames → 769 tokens → ~2 GB.
    num_frames: int = 12

    # Spatial tile size in pixels (must match precomputed tensor patches)
    img_size: int = 128

    # Optimization
    batch_size: int = 4
    accumulate_grad_batches: int = 2   # effective batch = batch_size * accumulate_grad_batches
    lr: float = 1e-4
    epochs: int = 50
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True

    # Freeze the ViT backbone and only train the neck + decoder.
    # AdamW optimizer states for 923 M params consume ~13 GB in fp32 — more than
    # the 22 GiB GPU can accommodate alongside activations.  With the backbone
    # frozen, trainable params drop to ~tens of millions and the model fits.
    freeze_backbone: bool = True

    # Dataloader — num_workers=0 avoids IPC socket crashes on PACE
    num_workers: int = 4
    pin_memory: bool = False
    augment: bool = True

    num_classes: int = 5
    ignore_index: int = 0
