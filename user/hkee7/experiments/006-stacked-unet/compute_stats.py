"""
Compute per-band mean and standard deviation over the training set.

Runs a single pass through every training chunk and accumulates Welford
online statistics (numerically stable, O(1) memory per band).

Usage:
    uv run python compute_stats.py
    uv run python compute_stats.py --data-path /path/to/precomputed_tensors --out stats.pt
"""

from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm


def compute_band_stats(chunk_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Welford online algorithm over all training patches.

    Returns
    -------
    mean : Tensor[float64]  — (C,)  per-band mean
    std  : Tensor[float64]  — (C,)  per-band std (unbiased)
    """
    chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No .pt chunks found in {chunk_dir}")

    # Peek at first chunk to get shape
    sample = torch.load(os.path.join(chunk_dir, chunk_files[0]), weights_only=True)
    # data shape: (N_patches, T, C, H, W)
    _, T, C, H, W = sample["data"].shape
    print(f"  Chunk shape: T={T}, C={C}, H={H}, W={W}")

    # Welford accumulators — one per band (treating T as extra samples)
    count = torch.zeros(C, dtype=torch.float64)
    mean  = torch.zeros(C, dtype=torch.float64)
    M2    = torch.zeros(C, dtype=torch.float64)

    for fname in tqdm(chunk_files, desc="Chunks"):
        chunk = torch.load(os.path.join(chunk_dir, fname), weights_only=True)
        # (N, T, C, H, W) → raw int16/uint16 values
        data = chunk["data"].float() / 10_000.0   # to reflectance [0, ~1]

        # Reshape to (C, N*T*H*W) for vectorised Welford
        data = data.permute(2, 0, 1, 3, 4).reshape(C, -1).double()

        for c in range(C):
            vals = data[c]
            n = vals.numel()
            new_count = count[c] + n
            delta = vals - mean[c]
            mean[c] += delta.sum() / new_count
            delta2 = vals - mean[c]
            M2[c] += (delta * delta2).sum()
            count[c] = new_count

    variance = M2 / (count - 1)   # unbiased
    std = variance.sqrt()

    return mean.float(), std.float()


def main():
    parser = argparse.ArgumentParser(description="Compute per-band normalisation stats")
    parser.add_argument("--data-path", type=str, default="data/precomputed_tensors")
    parser.add_argument("--out", type=str, default="stats.pt",
                        help="Output path for the stats tensor dict")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_path, "train")
    print(f"Computing stats from: {train_dir}")

    mean, std = compute_band_stats(train_dir)

    # Guard against zero-std bands (constant channels)
    std = std.clamp(min=1e-6)

    stats = {"mean": mean, "std": std}
    torch.save(stats, args.out)

    print(f"\nSaved stats → {args.out}")
    print(f"  mean (per band): {mean.tolist()}")
    print(f"  std  (per band): {std.tolist()}")


if __name__ == "__main__":
    main()
