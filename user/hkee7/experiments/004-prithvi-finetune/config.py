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

    # Single-frame strategy: backbone always receives T=1.
    # Each of the 34 raw frames is treated as an independent training sample,
    # giving 34× more data points and exact alignment with pretrained weights.
    num_frames: int = 1

    # Spatial tile size in pixels (must match precomputed tensor patches)
    img_size: int = 128

    # Optimization
    # T=1 input → 64 tokens/sample (vs 2177 for T=34).
    # With grad checkpointing + bf16 on 96 GB GPU, batch=32 is comfortably
    # within memory; accumulate=1 is sufficient for stable gradients.
    batch_size: int = 32
    accumulate_grad_batches: int = 1   # effective batch = 32
    lr: float = 1e-4
    epochs: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    precision: str = "bf16-mixed"

    # Full backbone fine-tuning — set freeze_backbone=True for GPUs <40 GB.
    freeze_backbone: bool = False

    num_workers: int = 4
    pin_memory: bool = False
    augment: bool = True

    num_classes: int = 5
    ignore_index: int = 0
