"""
Dataset for the SatLas experiment.

Returns per-pixel temporal statistics over all 34 acquisitions across all
10 Sentinel-2 bands: mean, std, min, max → 40 input channels total.

This deliberately collapses the temporal axis into summary statistics rather
than modelling time explicitly. The Swin-V2-B backbone then treats the 40-channel
tensor like a multi-spectral image with richer spectral diversity.
"""

import os
import random
from typing import Literal

import torch
from torch.utils.data import Dataset, Sampler

REFLECTANCE_SCALE = 10_000.0


class SatlasDataset(Dataset):
    """
    Map-style dataset over precomputed AgriPotential chunks.

    Loads all 10 S2 bands across all 34 timesteps and computes four
    per-band temporal statistics (mean, std, min, max), yielding a
    (40, H, W) feature tensor ready for a standard image backbone.

    Returns
    -------
    features   : Tensor[float32] — (C=40, H, W)
    label      : Tensor[long]    — (H, W)
    patch_id   : str             — unique patch identifier (test mode only)
    """

    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        chunk_dir: str = "data/precomputed_tensors",
        augment: bool = False,
    ):
        self.mode = mode
        self.chunk_dir = os.path.join(chunk_dir, mode)
        self.augment = augment

        chunk_files = sorted(f for f in os.listdir(self.chunk_dir) if f.endswith(".pt"))
        self.index: list[tuple[str, int]] = []
        for f in chunk_files:
            path = os.path.join(self.chunk_dir, f)
            n = torch.load(path, weights_only=True, mmap=True)["data"].shape[0]
            self.index.extend((f, i) for i in range(n))

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
            self._cache_label = payload.get(
                "label",
                torch.zeros(
                    payload["data"].shape[0],
                    payload["data"].shape[-2],
                    payload["data"].shape[-1],
                    dtype=torch.long,
                ),
            )
            self._cache_ids = payload.get("patch_ids", [])

        # (T=34, C=10, H, W) → float32 in [0, 1]
        data = self._cache_data[patch_idx].float() / REFLECTANCE_SCALE
        data = data.clamp(0.0, 1.0)

        features = _temporal_stats(data)  # (40, H, W)

        # Per-patch z-score normalisation: remove absolute reflectance level so the
        # model sees relative spectral patterns.  Test patches come from different
        # regions with different absolute levels; this makes features region-invariant.
        mean = features.mean(dim=(-2, -1), keepdim=True)   # (40, 1, 1)
        std  = features.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        features = (features - mean) / std
        label = self._cache_label[patch_idx]
        patch_id = self._cache_ids[patch_idx] if len(self._cache_ids) > 0 else ""

        if self.augment:
            features, label = _random_spatial_augment(features, label)

        if self.mode == "test":
            return features, label, patch_id
        return features, label


def _temporal_stats(data: torch.Tensor) -> torch.Tensor:
    """Compute (mean, std, min, max) over T=34 for each of the 10 bands.

    Args:
        data: (T, C, H, W) float32

    Returns:
        (4*C, H, W) float32 — [mean_B1..B10, std_B1..B10, min_B1..B10, max_B1..B10]
    """
    mean = data.mean(dim=0)             # (C, H, W)
    std  = data.std(dim=0)              # (C, H, W)
    vmin = data.amin(dim=0)             # (C, H, W)
    vmax = data.amax(dim=0)             # (C, H, W)
    return torch.cat([mean, std, vmin, vmax], dim=0)  # (4C, H, W)


class ChunkAwareSampler(Sampler):
    """Yields indices grouped by chunk file to keep the per-worker LRU cache warm."""

    def __init__(self, dataset: SatlasDataset, shuffle: bool = True, seed: int = 42):
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
    features: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal/vertical flips and 90° rotations."""
    if random.random() < 0.5:
        features = features.flip(-1)
        label = label.flip(-1)
    if random.random() < 0.5:
        features = features.flip(-2)
        label = label.flip(-2)
    k = random.randint(0, 3)
    if k > 0:
        features = torch.rot90(features, k, dims=(-2, -1))
        label = torch.rot90(label, k, dims=(-2, -1))
    return features, label
