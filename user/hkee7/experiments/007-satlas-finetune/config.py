from dataclasses import dataclass


@dataclass
class Config:
    # ---- Environment --------------------------------------------------------
    device: str = "cuda"
    seed: int = 42

    # ---- Paths --------------------------------------------------------------
    data_path: str = "data/precomputed_tensors"
    metadata_path: str = "data/agripotential/metadata.csv"
    save_dir: str = "artifacts"

    # ---- Model --------------------------------------------------------------
    # timm model name — ms_in22k_ft_in1k = pretrained on ImageNet-22k then fine-tuned
    # on ImageNet-1K, giving the richest feature initialisation available.
    encoder: str = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"

    # Path to a SatLas pretrained checkpoint (.pth) to use instead of
    # ImageNet-22k weights.  Set to "" to use the default timm weights.
    # Download from: https://huggingface.co/allenai/satlas-pretrain
    satlas_checkpoint: str = ""

    # 34 timesteps × (10 S2 bands + 3 spectral indices) = 34 × 13 = 442 channels.
    # NDVI, EVI, NDWI appended per timestep — ratio-based indices are physically
    # meaningful for vegetation and invariant to absolute reflectance scale.
    in_channels: int = 442
    fpn_channels: int = 256

    # ---- Task ---------------------------------------------------------------
    num_classes: int = 5    # suitability classes 1–5  (label 0 = unlabelled)
    ignore_index: int = 0

    # ---- Optimisation -------------------------------------------------------
    # Swin-V2-B (87 M params, all in backbone) + FPN head (~5 M) = ~92 M total.
    # bf16-mixed on 96 GB Blackwell: batch=64 is well within budget.
    batch_size: int = 64
    accumulate_grad_batches: int = 1    # effective batch = 64
    lr: float = 1e-3                    # head-only LR — backbone is frozen
    backbone_lr_scale: float = 0.0      # 0.0 = freeze backbone entirely
    epochs: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    precision: str = "bf16-mixed"

    # ---- DataLoader ---------------------------------------------------------
    num_workers: int = 6
    pin_memory: bool = False
    augment: bool = True

    # ---- Misc ---------------------------------------------------------------
    img_size: int = 128
