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
    # On RTX 6000 Blackwell (96 GB) all 34 frames fit with full fine-tuning.
    # On smaller GPUs (22 GB) use num_frames=12 with freeze_backbone=True.
    num_frames: int = 34

    # Spatial tile size in pixels (must match precomputed tensor patches)
    img_size: int = 128

    # Optimization
    # With full backbone fine-tuning on 96 GB: batch=8, accum=2 → effective=16
    batch_size: int = 8
    accumulate_grad_batches: int = 2   # effective batch = batch_size * accumulate_grad_batches
    lr: float = 1e-4
    epochs: int = 50
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    precision: str = "bf16-mixed"

    # Full backbone fine-tuning — requires ≥40 GB GPU (96 GB RTX 6000 Blackwell).
    # Set True and reduce num_frames=12 / batch_size=4 for smaller GPUs.
    freeze_backbone: bool = False

    # num_workers=0: each worker loads a full chunk (~GB) into RAM; with 4 workers
    # the combined footprint can silently exceed the cgroup mem limit → SIGKILL.
    # Single-process loading is slower but safe on networked PACE storage.
    num_workers: int = 0
    pin_memory: bool = False
    augment: bool = True

    num_classes: int = 5
    ignore_index: int = 0
