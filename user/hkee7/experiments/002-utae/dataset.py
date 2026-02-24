"""
PyTorch Dataset for AgriPotential → U-TAE training.

Reads precomputed .pt chunk files directly — no external dependencies.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

REFLECTANCE_SCALE = 10_000.0


class UTAEDataset(Dataset):
    """
    Map-style dataset over precomputed AgriPotential chunks.

    Returns
    -------
    data : Tensor[float32]  — (T, C, H, W)
        Sentinel-2 time series, normalised to [0, 1].
    label : Tensor[long]    — (H, W)
        Pixel labels 0–5 (0 = unlabelled, 1–5 = potential classes).
    positions : Tensor[long] — (T,)
        Days since first acquisition (used by LTAE positional encoding).
    """

    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        chunk_dir: str = "data/precomputed_tensors",
        metadata_path: str = "data/agripotential/metadata.csv",
    ):
        self.chunk_dir = os.path.join(chunk_dir, mode)
        chunk_files = sorted(f for f in os.listdir(self.chunk_dir) if f.endswith(".pt"))

        self.index: list[tuple[str, int]] = []
        for f in chunk_files:
            path = os.path.join(self.chunk_dir, f)
            n = torch.load(path, weights_only=True, mmap=True)["data"].shape[0]
            self.index.extend((f, i) for i in range(n))

        meta = pl.read_csv(metadata_path)
        self.time_offsets = torch.from_numpy(_parse_positions(meta))

        self._cache_file: str | None = None
        self._cache_data = None
        self._cache_label = None
        self._cache_ids = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, patch_idx = self.index[idx]
        if f != self._cache_file:
            payload = torch.load(os.path.join(self.chunk_dir, f), weights_only=True)
            self._cache_file = f
            self._cache_data = payload["data"]
            self._cache_label = payload["label"]
            self._cache_ids = payload["patch_ids"]

        data = self._cache_data[patch_idx].float() / REFLECTANCE_SCALE
        data = data.clamp(0.0, 1.0)
        label = self._cache_label[patch_idx]
        patch_id = self._cache_ids[patch_idx]
        return data, label, self.time_offsets, patch_id


class ChunkAwareSampler(Sampler):
    """Yields indices grouped by chunk to keep the per-worker cache warm."""

    def __init__(self, dataset: UTAEDataset, shuffle: bool = True, seed: int = 42):
        self.shuffle = shuffle
        self.rng = torch.Generator().manual_seed(seed)

        chunks: dict[str, list[int]] = {}
        for i, (f, _) in enumerate(dataset.index):
            chunks.setdefault(f, []).append(i)
        self.chunks = list(chunks.values())

    def __iter__(self):
        chunk_order = list(range(len(self.chunks)))
        if self.shuffle:
            chunk_order = torch.randperm(len(self.chunks), generator=self.rng).tolist()

        for ci in chunk_order:
            indices = self.chunks[ci]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.rng).tolist()
                indices = [indices[j] for j in perm]
            yield from indices

    def __len__(self):
        return sum(len(c) for c in self.chunks)


def _parse_positions(meta: pl.DataFrame) -> np.ndarray:
    """Compute days since first acquisition from metadata."""
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
