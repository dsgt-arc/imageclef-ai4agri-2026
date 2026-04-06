import os
from dataclasses import dataclass, field

@dataclass
class Config:
    # Environment
    device: str = "cuda"
    seed: int = 42

    # Paths 
    data_path: str = "data/precomputed_tensors"
    metadata_path: str = "data/agripotential/metadata.csv"
    save_dir: str = "artifacts"
    
    # Prithvi
    # The Prithvi-EO-2.0 300M model backbone name in terratorch
    backbone: str = "prithvi_eo_v2_300" 
    
    # Optimization
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 40
    weight_decay: float = 0.05
    use_amp: bool = True
    
    # Dataloader
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True
    
    # We will pool temporals in the model logic
    num_classes: int = 5
    ignore_index: int = 0
