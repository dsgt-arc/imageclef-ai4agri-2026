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
    # LearnedInterpolateToPyramidal params scale as O(T²): 34 frames → 5.9B params,
    # 71 GB optimizer states alone.  12 frames → ~430M neck params, fits on 96 GB.
    # Backbone still fine-tunes end-to-end; temporal coverage traded for feasibility.
    num_frames: int = 12

    # Spatial tile size in pixels (must match precomputed tensor patches)
    img_size: int = 128

    # Optimization
    # Without working gradient checkpointing, all 24 ViT blocks store activations:
    # ~1.9 GB/block at batch=2 → 22 GB activations + 18 GB optimizer = 40 GB peak.
    # batch=4 → 44+18=62 GB which tips over 95 GB during validation overlap.
    # With 12 frames and full backbone: ~1B params, ~12 GB optimizer, ~6 GB activations
    # at batch=4 → ~20 GB total, very comfortable on 96 GB.
    batch_size: int = 4
    accumulate_grad_batches: int = 4   # effective batch = 16
    lr: float = 1e-4
    epochs: int = 50
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
