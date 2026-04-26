import os
import random
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

REFLECTANCE_SCALE = 10_000.0


class PrithviDataset(Dataset):
    """
    Map-style dataset over precomputed AgriPotential chunks.
    Filters the 10 S2 bands to the 6 bands required by Prithvi-EO-1.0/2.0:
    Blue, Green, Red, Narrow NIR, SWIR1, SWIR2.

    When seasonal_composite=True (recommended), the 34 raw timesteps are
    averaged within each of 4 meteorological seasons (Winter/Spring/Summer/Autumn),
    returning T=4 composites instead of T=34 raw frames.  This collapses the
    redundant multi-year temporal dimension into a compact seasonal signature,
    matching the static nature of the viticulture suitability label.

    Returns
    -------
    data       : Tensor[float32] — (T, C=6, H, W)   T=4 (seasonal) or T=34 (raw)
    label      : Tensor[long]    — (H, W)
    doys       : Tensor[float32] — (T,)
    """

    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        chunk_dir: str = "data/precomputed_tensors",
        metadata_path: str = "data/agripotential/metadata.csv",
        augment: bool = False,
        seasonal_composite: bool = False,
    ):
        self.chunk_dir = os.path.join(chunk_dir, mode)
        chunk_files = sorted(f for f in os.listdir(self.chunk_dir) if f.endswith(".pt"))

        self.index: list[tuple[str, int]] = []
        for f in chunk_files:
            path = os.path.join(self.chunk_dir, f)
            n = torch.load(path, weights_only=True, mmap=True)["data"].shape[0]
            self.index.extend((f, i) for i in range(n))

        self.augment = augment
        self.seasonal_composite = seasonal_composite

        meta = pl.read_csv(metadata_path)
        if seasonal_composite:
            self.seasonal_indices, seasonal_doys = _parse_seasonal_indices(meta)
            self.doys = torch.from_numpy(seasonal_doys)
        else:
            self.doys = torch.from_numpy(_parse_doys(meta)).float()

        self._cache_file: str | None = None
        self._cache_data = None
        self._cache_label = None
        self._cache_ids = None
        self.mode = mode

        # Indices in the 10-band dataset corresponding to:
        # B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR), B11 (SWIR1), B12 (SWIR2)
        # Original: 0=B2, 1=B3, 2=B4, 3=B5, 4=B6, 5=B7, 6=B8, 7=B8A, 8=B11, 9=B12
        self.band_indices = torch.tensor([0, 1, 2, 7, 8, 9])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, patch_idx = self.index[idx]
        if f != self._cache_file:
            payload = torch.load(os.path.join(self.chunk_dir, f), weights_only=True)
            self._cache_file = f
            self._cache_data = payload["data"]
            self._cache_label = payload.get("label", torch.zeros(
                payload["data"].shape[0], payload["data"].shape[-2], payload["data"].shape[-1], dtype=torch.long
            ))
            self._cache_ids = payload.get("patch_ids", [])

        # Load all timesteps for this patch, select 6 HLS bands
        data = self._cache_data[patch_idx, :, self.band_indices, :, :].float() / REFLECTANCE_SCALE
        data = data.clamp(0.0, 1.0)  # (34, 6, H, W)

        if self.seasonal_composite:
            # Average within each season → (4, 6, H, W): [Winter, Spring, Summer, Autumn]
            data = torch.stack([
                data[torch.from_numpy(idx)].mean(dim=0)
                for idx in self.seasonal_indices
            ])
        label = self._cache_label[patch_idx]
        
        # We also support test mode which needs patch_ids
        patch_id = self._cache_ids[patch_idx] if len(self._cache_ids) > 0 else ""

        if self.augment:
            data, label = _random_spatial_augment(data, label)

        if self.mode == "test":
            return data, label, self.doys, patch_id
        return data, label, self.doys


class ChunkAwareSampler(Sampler):
    """Yields indices grouped by chunk to keep the per-worker cache warm."""

    def __init__(self, dataset: PrithviDataset, shuffle: bool = True, seed: int = 42):
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


def _random_spatial_augment(
    data: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random flips + 90° rotations applied identically to data and label."""
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


def _month_to_season(month: int) -> int:
    """Meteorological season index: 0=Winter, 1=Spring, 2=Summer, 3=Autumn."""
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3  # 9, 10, 11


def _parse_seasonal_indices(
    meta: pl.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Group the 34 raw timesteps into 4 meteorological seasons.

    Returns
    -------
    seasonal_indices : list of 4 int arrays — timestep indices per season
                       order: [Winter, Spring, Summer, Autumn]
    seasonal_doys    : (4,) float32 — mean day-of-year per season composite
    """
    months = meta.get_column("month").cast(pl.Int32).to_numpy()
    doys = _parse_doys(meta)
    season_ids = np.array([_month_to_season(int(m)) for m in months])

    seasonal_indices = [np.where(season_ids == s)[0] for s in range(4)]
    # Representative DOY: mean of frames in each season (fallback to mid-season)
    fallback_doys = np.array([15.0, 105.0, 196.0, 288.0], dtype=np.float32)
    seasonal_doys = np.array(
        [
            float(doys[idx].mean()) if len(idx) > 0 else fallback_doys[s]
            for s, idx in enumerate(seasonal_indices)
        ],
        dtype=np.float32,
    )
    return seasonal_indices, seasonal_doys


def _parse_doys(meta: pl.DataFrame) -> np.ndarray:
    """Day-of-year (1–365) for each acquisition timestep."""
    dates = meta.select(
        pl.date(
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
            pl.col("day").cast(pl.Int32),
        ).alias("date")
    ).get_column("date")

    doys = dates.dt.ordinal_day()
    return doys.to_numpy().astype(np.float32)
