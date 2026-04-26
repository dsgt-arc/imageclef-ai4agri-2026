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

    # Seasonal compositing: average the 34 raw frames within each of 4 meteorological
    # seasons (Winter/Spring/Summer/Autumn) before feeding to the backbone.
    # Reduces T from 34→4, collapsing redundant multi-year coverage into a compact
    # seasonal signature that matches the static viticulture suitability label.
    # Also shrinks LearnedInterpolateToPyramidal from 5.9B → ~50M params.
    seasonal_composite: bool = True
    num_frames: int = 4   # must equal 4 when seasonal_composite=True

    # Spatial tile size in pixels (must match precomputed tensor patches)
    img_size: int = 128

    # Optimization
    # Without working gradient checkpointing, all 24 ViT blocks store activations:
    # ~1.9 GB/block at batch=2 → 22 GB activations + 18 GB optimizer = 40 GB peak.
    # batch=4 → 44+18=62 GB which tips over 95 GB during validation overlap.
    # 4 seasonal frames → 401M params, ~10 GB peak on 96 GB GPU at batch=32.
    batch_size: int = 32
    accumulate_grad_batches: int = 1   # effective batch = 32
    lr: float = 1e-4
    epochs: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    precision: str = "bf16-mixed"

    # Full backbone fine-tuning — requires ≥40 GB GPU (96 GB RTX 6000 Blackwell).
    # Set True and reduce num_frames=12 / batch_size=4 for smaller GPUs.
    freeze_backbone: bool = False

    num_workers: int = 4
    pin_memory: bool = False
    augment: bool = True

    num_classes: int = 5
    ignore_index: int = 0
