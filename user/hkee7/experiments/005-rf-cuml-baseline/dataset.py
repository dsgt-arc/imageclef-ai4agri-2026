from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REFLECTANCE_SCALE = 10_000.0


def feature_names(num_timesteps: int = 34, num_bands: int = 10) -> list[str]:
    return [
        f"t{t:02d}_b{b:02d}"
        for t in range(num_timesteps)
        for b in range(num_bands)
    ]


def _chunk_paths(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.glob("*.pt"))


def sample_pixel_table(
    split: str,
    *,
    data_root: Path,
    max_pixels: int,
    random_state: int,
    max_chunks: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    split_dir = data_root / split
    paths = _chunk_paths(split_dir)
    if not paths:
        raise FileNotFoundError(f"No chunk files found in {split_dir}")

    if max_chunks is not None:
        paths = paths[:max_chunks]

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    per_chunk_budget = max(1, max_pixels // len(paths))

    for path in paths:
        payload = torch.load(path, weights_only=True)
        data = payload["data"].float().numpy() / REFLECTANCE_SCALE
        labels = payload["label"].numpy()

        # (N, T, C, H, W) -> (N*H*W, T*C)
        n, t, c, h, w = data.shape
        x = data.transpose(0, 3, 4, 1, 2).reshape(n * h * w, t * c)
        y = labels.reshape(n * h * w)

        mask = y > 0
        x = x[mask]
        y = y[mask]

        if x.shape[0] > per_chunk_budget:
            idx = rng.choice(x.shape[0], size=per_chunk_budget, replace=False)
            x = x[idx]
            y = y[idx]

        x_parts.append(x.astype(np.float32, copy=False))
        y_parts.append(y.astype(np.int32, copy=False))

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    if x_all.shape[0] > max_pixels:
        idx = rng.choice(x_all.shape[0], size=max_pixels, replace=False)
        x_all = x_all[idx]
        y_all = y_all[idx]

    return x_all, y_all

