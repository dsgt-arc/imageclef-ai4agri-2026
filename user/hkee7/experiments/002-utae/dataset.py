"""
PyTorch Dataset adapter for AgriPotential → U-TAE training.

Wraps the shared PotentialDataset and applies normalization.
Also supports the precomputed StreamingChunkDataset for faster I/O.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset

from stream_chunk_dataset import StreamingChunkDataset

# Sentinel-2 L2A reflectance is stored as integers in [0, 10000]
REFLECTANCE_SCALE = 10_000.0


class UTAEDataset(IterableDataset):
    """
    Map-style dataset that wraps ``StreamChunkDataset`` for U-TAE training.

    Returns
    -------
    data : Tensor[float32]  — (T, C, H, W)
        Sentinel-2 time series, normalised to [0, 1].
    label : Tensor[long]    — (H, W)
        Pixel labels 0–5 (0 = unlabelled, 1–5 = potential classes).
    positions : Tensor[long] — (T,)
        Days since first acquisition (used by LTAE positional encoding).
        The 34 timesteps span 2017–2019, so day-of-year would wrap and
        collide across years. Absolute day offset preserves full ordering.
    """

    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        data_path: str | None = "data/precomputed_tensors",
        metadata_path: str | None = "data/agripotential/metadata.csv",
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.base = StreamingChunkDataset(
            mode=mode, chunk_dir=data_path, shuffle=shuffle, seed=seed
        )

        # ---- Temporal positions from metadata --------------------------------
        # metadata.csv has columns: filename, day, month, year
        # The 34 images span 2017–2019, so we use days since the first
        # acquisition to preserve full temporal ordering across years
        metadata_df = pl.read_csv(metadata_path)
        self.time_offsets = torch.from_numpy(_parse_positions(metadata_df))

    def __iter__(self):
        for data, label in self.base:
            data = (data / REFLECTANCE_SCALE).clamp(0.0, 1.0)
            yield data, label, self.time_offsets


def _parse_positions(meta: pl.DataFrame) -> np.ndarray:
    """Compute days since first acquisition from metadata.

    Uses the (year, month, day) columns in metadata.csv.
    Falls back to sequential indices.
    """
    dates = meta.select(
        pl.date(
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
            pl.col("day").cast(pl.Int32),
        ).alias("date")
    ).get_column("date")

    origin = dates.min()
    offsets = (dates - origin).dt.total_days()
    return offsets.to_numpy().astype(np.int64)
