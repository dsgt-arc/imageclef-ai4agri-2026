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

    Each item is a **single timestep** of a single spatial patch, so the model
    always receives T=1 frames.  This lets the backbone run exactly as pretrained
    (no temporal-token interpolation) and multiplies the effective training set
    size by num_frames (34×).  Predictions for the same patch are averaged at
    inference time.

    Returns
    -------
    data   : Tensor[float32] — (1, C=6, H, W)
    label  : Tensor[long]    — (H, W)
    doy    : Tensor[float32] — (1,)  day-of-year for this specific frame
    """

    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        chunk_dir: str = "data/precomputed_tensors",
        metadata_path: str = "data/agripotential/metadata.csv",
        augment: bool = False,
    ):
        self.chunk_dir = os.path.join(chunk_dir, mode)
        chunk_files = sorted(f for f in os.listdir(self.chunk_dir) if f.endswith(".pt"))

        # index: (chunk_filename, patch_idx_in_chunk, frame_idx)
        self.index: list[tuple[str, int, int]] = []
        for f in chunk_files:
            path = os.path.join(self.chunk_dir, f)
            chunk = torch.load(path, weights_only=True, mmap=True)
            n_patches, n_frames = chunk["data"].shape[:2]
            for p in range(n_patches):
                for t in range(n_frames):
                    self.index.append((f, p, t))

        self.augment = augment
        self.mode = mode

        meta = pl.read_csv(metadata_path)
        # Per-frame DOY array, shape (num_frames,)
        self._all_doys = torch.from_numpy(_parse_doys(meta)).float()

        self._cache_file: str | None = None
        self._cache_data = None
        self._cache_label = None
        self._cache_ids = None

        # Indices in the 10-band dataset corresponding to:
        # B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR), B11 (SWIR1), B12 (SWIR2)
        # Original: 0=B2, 1=B3, 2=B4, 3=B5, 4=B6, 5=B7, 6=B8, 7=B8A, 8=B11, 9=B12
        self.band_indices = torch.tensor([0, 1, 2, 7, 8, 9])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, patch_idx, frame_idx = self.index[idx]

        if f != self._cache_file:
            payload = torch.load(os.path.join(self.chunk_dir, f), weights_only=True)
            self._cache_file = f
            self._cache_data = payload["data"]
            n = payload["data"].shape[0]
            h, w = payload["data"].shape[-2], payload["data"].shape[-1]
            self._cache_label = payload.get(
                "label", torch.zeros(n, h, w, dtype=torch.long)
            )
            self._cache_ids = payload.get("patch_ids", [])

        # Single frame: (1, C=6, H, W)
        data = (
            self._cache_data[patch_idx, frame_idx, self.band_indices, :, :]
            .float()
            .div_(REFLECTANCE_SCALE)
            .clamp_(0.0, 1.0)
            .unsqueeze(0)  # add T=1 dim
        )
        label = self._cache_label[patch_idx]  # (H, W)
        doy = self._all_doys[frame_idx : frame_idx + 1]  # (1,)

        if self.augment:
            data, label = _random_spatial_augment(data, label)

        if self.mode == "test":
            patch_id = self._cache_ids[patch_idx] if len(self._cache_ids) > 0 else ""
            return data, label, doy, patch_id
        return data, label, doy


class ChunkAwareSampler(Sampler):
    """Yields indices grouped by chunk to keep the per-worker cache warm.

    Groups items by chunk file so consecutive samples share the same mmap'd
    tensor, minimising redundant disk reads across workers.
    """

    def __init__(self, dataset: PrithviDataset, shuffle: bool = True, seed: int = 42):
        self.shuffle = shuffle
        self.rng = torch.Generator().manual_seed(seed)

        chunks: dict[str, list[int]] = {}
        for i, (f, _p, _t) in enumerate(dataset.index):
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
