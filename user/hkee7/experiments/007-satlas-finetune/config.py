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
    # timm model name — window8 variant generalises better to 128×128 input
    # (window_size=8 patches → 32×32 pixel windows at stride 4, no padding issues)
    encoder: str = "swinv2_base_window8_256.ms_in22k_ft_in1k"

    # Path to a SatLas pretrained checkpoint (.pth) to use instead of
    # ImageNet-22k weights.  Set to "" to use the default timm weights.
    # Download from: https://huggingface.co/allenai/satlas-pretrain
    satlas_checkpoint: str = ""

    # 10 S2 bands × 4 temporal statistics (mean, std, min, max) = 40 channels.
    # Treating multi-temporal frames as channel-wise statistics avoids the
    # temporal ordering assumption and matches how UNet currently sees the data.
    in_channels: int = 40
    fpn_channels: int = 256

    # ---- Task ---------------------------------------------------------------
    num_classes: int = 5    # suitability classes 1–5  (label 0 = unlabelled)
    ignore_index: int = 0

    # ---- Optimisation -------------------------------------------------------
    # Swin-V2-B (87 M params, all in backbone) + FPN head (~5 M) = ~92 M total.
    # bf16-mixed on 96 GB Blackwell: batch=64 is well within budget.
    batch_size: int = 64
    accumulate_grad_batches: int = 1    # effective batch = 64
    lr: float = 3e-4                    # head learning rate
    backbone_lr_scale: float = 0.1      # backbone uses lr × this factor
    epochs: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    precision: str = "bf16-mixed"

    # ---- DataLoader ---------------------------------------------------------
    num_workers: int = 8
    pin_memory: bool = False
    augment: bool = True

    # ---- Misc ---------------------------------------------------------------
    img_size: int = 128
