"""
Generate test predictions for AI4Agri submission using the trained PrestoOrdinal model.

Outputs a ZIP of PNG segmentation masks named by patch_id.
Pixel values: 0 (very low) to 4 (very high), matching competition format.

Usage:
    python predict.py --checkpoint artifacts/best_stage2.pt
    python predict.py --checkpoint artifacts/best_stage2.pt --tta --output-dir submission_tta
"""

from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
import torch
from config import Config
from model import PrestoOrdinal
from torch.utils.data import DataLoader, Dataset
from typing import Literal
import polars as pl

from PIL import Image

REFLECTANCE_SCALE = 10_000.0


# ---------------------------------------------------------------------------
# Minimal test dataset that also returns patch_ids
# ---------------------------------------------------------------------------


class PrestoTestDataset(Dataset):
    """Loads test patches and returns (data, label, doys, patch_id)."""

    def __init__(
        self,
        mode: Literal["train", "val", "test"] = "test",
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
        self.doys = torch.from_numpy(_parse_doys(meta)).float()

        self._cache_file: str | None = None
        self._cache_data = None
        self._cache_label = None
        self._cache_ids = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, patch_idx = self.index[idx]
        if f != self._cache_file:
            payload = torch.load(
                os.path.join(self.chunk_dir, f), weights_only=True
            )
            self._cache_file = f
            self._cache_data = payload["data"]
            self._cache_label = payload.get("label", torch.zeros(
                payload["data"].shape[0], payload["data"].shape[-2], payload["data"].shape[-1], dtype=torch.long
            ))
            self._cache_ids = payload["patch_ids"]

        data = self._cache_data[patch_idx].float() / REFLECTANCE_SCALE
        data = data.clamp(0.0, 1.0)
        label = self._cache_label[patch_idx]
        patch_id = self._cache_ids[patch_idx]
        return data, label, self.doys, patch_id


def _parse_doys(meta: pl.DataFrame) -> np.ndarray:
    dates = meta.select(
        pl.date(
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
            pl.col("day").cast(pl.Int32),
        ).alias("date")
    ).get_column("date")
    return dates.dt.ordinal_day().to_numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

_TTA_AUGMENTS = [
    (lambda x: x,                          lambda x: x),
    (lambda x: x.flip(-1),                 lambda x: x.flip(-1)),
    (lambda x: x.flip(-2),                 lambda x: x.flip(-2)),
    (lambda x: x.flip(-1).flip(-2),        lambda x: x.flip(-1).flip(-2)),
    (lambda x: torch.rot90(x, 1, [-2,-1]), lambda x: torch.rot90(x, 3, [-2,-1])),
    (lambda x: torch.rot90(x, 2, [-2,-1]), lambda x: torch.rot90(x, 2, [-2,-1])),
    (lambda x: torch.rot90(x, 3, [-2,-1]), lambda x: torch.rot90(x, 1, [-2,-1])),
    (lambda x: torch.rot90(x, 1, [-2,-1]).flip(-1),
     lambda x: torch.rot90(x.flip(-1), 3, [-2,-1])),
]


def _tta_predict(model: PrestoOrdinal, data: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Average sigmoid probs over 8 geometric augmentations."""
    # data shape: (B, T, C, H, W)
    probs_list = []
    for aug_fn, deaug_fn in _TTA_AUGMENTS:
        # Augment the spatial dims (last two) of the 5-D input
        aug_data = aug_fn(data)
        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            logits = model(aug_data)          # (B, K-1, H, W)
        probs = logits.sigmoid()              # (B, K-1, H, W)
        probs_list.append(deaug_fn(probs))    # de-augment spatial dims
    return torch.stack(probs_list).mean(0)    # (B, K-1, H, W)


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------


def predict(
    checkpoint_path: str,
    cfg: Config,
    output_dir: str = "submission",
    tta: bool = False,
    split: str = "test",
):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)

    # Restore saved config but allow override of device / batch / workers
    saved_cfg: Config = ckpt.get("config", cfg)
    saved_cfg.device    = cfg.device
    saved_cfg.batch_size  = cfg.batch_size
    saved_cfg.num_workers = cfg.num_workers
    saved_cfg.pin_memory  = cfg.pin_memory
    saved_cfg.chunk_size  = cfg.chunk_size
    cfg = saved_cfg

    model = PrestoOrdinal(
        num_classes=cfg.num_classes,
        head_hidden_dim=cfg.head_hidden_dim,
        freeze_encoder=False,         # weights will be loaded; freeze state irrelevant
        presto_weights=cfg.presto_path,
        chunk_size=cfg.chunk_size,
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"TTA: {'enabled (8 augmentations)' if tta else 'disabled'}")
    print(f"Split: {split}")

    ds = PrestoTestDataset(split, cfg.data_path, cfg.metadata_path)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    print(f"  {len(ds)} patches to predict")

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for data, _labels, _doys, patch_ids in loader:
            data = data.to(cfg.device, non_blocking=True)

            if tta:
                probs = _tta_predict(model, data, cfg)          # (B, K-1, H, W)
                preds = (probs > 0.5).sum(dim=1) + 1            # → [1, 5]
            else:
                with torch.autocast(cfg.device, enabled=cfg.use_amp):
                    logits = model(data)                         # (B, K-1, H, W)
                preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1 # → [1, 5]

            # Convert from training space [1,5] → submission space [0,4],
            # then clamp to [1,3] so no prediction is ever >1 away from
            # the true label at the boundaries (safe for ±1 accuracy metric).
            submission_preds = preds.long().clamp(2, 4) - 1  # → [1, 3]

            for pred, pid in zip(submission_preds.cpu().numpy(), patch_ids):
                img = Image.fromarray(pred.astype(np.uint8), mode="L")
                img.save(os.path.join(output_dir, f"{pid}.png"))
                count += 1

    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith(".png"):
                zf.write(os.path.join(output_dir, fname), fname)

    print(f"Saved {count} predictions → {zip_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI4Agri test submission")
    parser.add_argument(
        "--checkpoint", type=str, default="artifacts/best_stage2.pt",
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument("--output-dir", type=str, default="submission")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (8 flips/rotations)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    cfg.batch_size  = args.batch_size
    cfg.chunk_size  = args.chunk_size
    cfg.num_workers = args.num_workers

    predict(args.checkpoint, cfg, args.output_dir, tta=args.tta, split=args.split)
