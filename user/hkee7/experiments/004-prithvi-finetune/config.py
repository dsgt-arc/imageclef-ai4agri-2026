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

    # Total timesteps in the dataset tensors (all 34 frames used directly).
    # TemporalMeanPool neck collapses T after spatial encoding, so
    # LearnedInterpolateToPyramidal's O(T²) param explosion is avoided.
    num_frames_data: int = 34
    seasonal_composite: bool = False
    num_frames: int = 34

    # Spatial tile size in pixels (must match precomputed tensor patches)
    img_size: int = 128

    # Optimization
    # 34 frames: 2177 tokens/sample; backbone ~300M params; TemporalMeanPool has no
    # learned params; InterpolateToPyramidal is bilinear (no params).
    # With grad checkpointing + bf16 on 96 GB GPU:
    #   model weights ~600 MB, optimizer (Adam) ~3.6 GB, activations ~1–2 GB/sample
    # batch=16, accum=2 → effective batch=32; peak ~25–30 GB — well within 96 GB.
    batch_size: int = 16
    accumulate_grad_batches: int = 2   # effective batch = 32
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
