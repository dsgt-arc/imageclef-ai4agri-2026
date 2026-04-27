"""
PyTorch Dataset for AgriPotential → U-TAE training.

Reads precomputed .pt chunk files directly — no external dependencies.
"""

from __future__ import annotations

import os
import random
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
        Sentinel-2 time series, z-score normalised per band (if stats provided)
        or scaled to [0, 1] by dividing by 10,000.
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
        augment: bool = False,
        band_mean: torch.Tensor | None = None,
        band_std: torch.Tensor | None = None,
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

        self.augment = augment

        # Per-band normalisation: (C,) tensors reshaped for broadcasting (1, C, 1, 1)
        if band_mean is not None and band_std is not None:
            self.band_mean = band_mean.view(1, -1, 1, 1)
            self.band_std  = band_std.view(1, -1, 1, 1)
        else:
            self.band_mean = None
            self.band_std  = None

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

        raw = self._cache_data[patch_idx].float() / REFLECTANCE_SCALE  # (T, C, H, W)
        raw = raw.clamp(0.0, 1.0)

        # Compute spectral indices from raw [0,1] reflectance before z-scoring.
        # Ratios are physically meaningful and more invariant to absolute reflectance
        # shifts between train and test geographic regions.
        indices = _spectral_indices(raw)   # (T, 4, H, W)

        if self.band_mean is not None:
            data = (raw - self.band_mean) / self.band_std
        else:
            data = raw

        # Concatenate original bands + spectral indices: (T, C+4, H, W)
        data = torch.cat([data, indices], dim=1)

        label = self._cache_label[patch_idx]
        patch_id = self._cache_ids[patch_idx]

        if self.augment:
            data, label = _random_spatial_augment(data, label)

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


def _spectral_indices(data: torch.Tensor) -> torch.Tensor:
    """Per-timestep spectral indices from 10-band Sentinel-2 reflectance in [0,1].

    Band layout: 0=B2(Blue) 1=B3(Green) 2=B4(Red) 3=B5(RE1) 4=B6(RE2) 5=B7(RE3)
                 6=B8(NIR-broad) 7=B8A(NIR-narrow) 8=B11(SWIR1) 9=B12(SWIR2)

    Returns (T, 4, H, W) clipped to [-1, 1]:
        NDVI  — vegetation health (NIR vs Red)
        EVI   — enhanced vegetation (atmospheric-corrected)
        NDWI  — water content (Green vs NIR)
        NDRE  — red-edge NDVI (NIR vs RedEdge1) — earlier stress detection
    """
    eps = 1e-6
    blue, green, red = data[:, 0], data[:, 1], data[:, 2]
    re1, nir = data[:, 3], data[:, 7]

    ndvi = (nir - red)   / (nir + red   + eps)
    evi  = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndre = (nir - re1)   / (nir + re1   + eps)

    return torch.stack([ndvi, evi, ndwi, ndre], dim=1).clamp(-1.0, 1.0)


def _random_spatial_augment(
    data: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random flips and 90° rotations applied identically to data and label."""
    if random.random() < 0.5:
        data = data.flip(-1)
        label = label.flip(-1)
    if random.random() < 0.5:
        data = data.flip(-2)
        label = label.flip(-2)
    k = random.randint(0, 3)
    if k > 0:
        data = torch.rot90(data, k, dims=(-2, -1))
        label = torch.rot90(label, k, dims=(-2, -1))
    return data, label


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
