"""
Dataset for the SatLas stacked-channel experiment.

All 34 timesteps × 10 bands are stacked into a single (340, H, W) tensor,
matching the organiser's stacked-channel approach — but fed to a pretrained
Swin-V2-B backbone instead of a from-scratch UNet.

Normalisation: simple /10000 clamp to [0, 1], preserving absolute reflectance
levels which carry real signal for viticulture suitability (e.g. high summer
NIR = healthy vegetation = higher suitability). Matches the organiser baseline.
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

    Stacks all 34 timesteps × 10 bands → (340, H, W) input tensor,
    then z-score normalises per patch to remove region-level reflectance bias.

    Returns
    -------
    features   : Tensor[float32] — (C=340, H, W)
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

        # Append per-timestep spectral indices: NDVI, EVI, NDWI → (T, 13, H, W)
        data = torch.cat([data, _spectral_indices(data)], dim=1)

        # Stack all timesteps into channels: (T, C, H, W) → (T*C, H, W)
        T, C, H, W = data.shape
        features = data.reshape(T * C, H, W)   # (442, H, W) in [0, 1]

        label = self._cache_label[patch_idx]
        patch_id = self._cache_ids[patch_idx] if len(self._cache_ids) > 0 else ""

        if self.augment:
            features, label = _random_spatial_augment(features, label)

        if self.mode == "test":
            return features, label, patch_id
        return features, label


def _spectral_indices(data: torch.Tensor) -> torch.Tensor:
    """Compute NDVI, EVI, NDWI for each of the T timesteps.

    Band layout (10-band Sentinel-2):
      0=B2 (Blue), 1=B3 (Green), 2=B4 (Red), 3=B5, 4=B6, 5=B7,
      6=B8, 7=B8A (Narrow NIR), 8=B11 (SWIR1), 9=B12 (SWIR2)

    Args:
        data: (T, 10, H, W) float32 in [0, 1]

    Returns:
        (T, 3, H, W) — [NDVI, EVI, NDWI] clipped to [-1, 1]
    """
    eps = 1e-6
    blue = data[:, 0]   # B2
    green = data[:, 1]  # B3
    red = data[:, 2]    # B4
    nir = data[:, 7]    # B8A

    ndvi = (nir - red) / (nir + red + eps)
    evi  = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    ndwi = (green - nir) / (green + nir + eps)

    indices = torch.stack([ndvi, evi, ndwi], dim=1).clamp(-1, 1)  # (T, 3, H, W)
    return indices


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
