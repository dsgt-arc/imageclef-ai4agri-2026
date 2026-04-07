from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("data/precomputed_tensors")
    artifacts_dir: Path = Path("user/hkee7/experiments/005-rf-cuml-baseline/artifacts")

    random_state: int = 42
    n_estimators: int = 300
    max_depth: int | None = 24
    max_features: str = "sqrt"

    max_train_pixels: int = 300_000
    max_val_pixels: int = 120_000
    chunks_per_split: int | None = None

    use_cuml: bool = True

