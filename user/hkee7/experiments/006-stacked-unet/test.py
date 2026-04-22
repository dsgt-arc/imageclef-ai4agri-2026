"""
Generate test predictions for AI4Agri submission (stacked-channel UNet).

Outputs a ZIP of PNG segmentation masks named by patch_id.
Pixel values: 0 (very low) to 4 (very high).

Supports optional test-time augmentation (--tta) — each aug is inverted
before averaging so all predictions are in the original spatial orientation.
"""

from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
import torch
from config import Config
from dataset import UTAEDataset
from PIL import Image
from torch.utils.data import DataLoader
from train import build_model, stack_input

# 8 geometric augmentations paired with their spatial inverses.
_TTA_AUGMENTS = [
    (lambda x: x,                                    lambda x: x),
    (lambda x: x.flip(-1),                           lambda x: x.flip(-1)),
    (lambda x: x.flip(-2),                           lambda x: x.flip(-2)),
    (lambda x: x.flip(-1).flip(-2),                  lambda x: x.flip(-1).flip(-2)),
    (lambda x: torch.rot90(x, 1, [-2, -1]),          lambda x: torch.rot90(x, 3, [-2, -1])),
    (lambda x: torch.rot90(x, 2, [-2, -1]),          lambda x: torch.rot90(x, 2, [-2, -1])),
    (lambda x: torch.rot90(x, 3, [-2, -1]),          lambda x: torch.rot90(x, 1, [-2, -1])),
    (lambda x: torch.rot90(x, 1, [-2, -1]).flip(-1), lambda x: torch.rot90(x.flip(-1), 3, [-2, -1])),
]


def _tta_predict(model: torch.nn.Module, x: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Average predictions over 8 geometric augmentations (with spatial inverse)."""
    preds = []
    for aug_fn, deaug_fn in _TTA_AUGMENTS:
        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            out = model(aug_fn(x))
        out = deaug_fn(out)
        if cfg.mode == "classification":
            preds.append(out.softmax(dim=1))
        elif cfg.mode == "ordinal":
            preds.append(out.sigmoid())
        else:
            preds.append(out)
    return torch.stack(preds).mean(0)


def predict(
    checkpoint_path: str,
    cfg: Config,
    output_dir: str = "submission",
    tta: bool = False,
):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=cfg.device)

    # Restore the saved config so architecture always matches the checkpoint.
    saved_cfg: Config = ckpt.get("config", cfg)
    saved_cfg.device = cfg.device
    saved_cfg.batch_size = cfg.batch_size
    saved_cfg.num_workers = cfg.num_workers
    saved_cfg.pin_memory = cfg.pin_memory
    cfg = saved_cfg

    in_channels: int = ckpt["in_channels"]
    model = build_model(cfg, in_channels).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Mode: {cfg.mode}  |  in_channels: {in_channels}")
    print(f"TTA: {'enabled (8 augmentations)' if tta else 'disabled'}")

    test_ds = UTAEDataset("test", cfg.data_path, cfg.metadata_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for data, _, _doys, patch_ids in test_loader:
            x = stack_input(data.to(cfg.device, non_blocking=True))

            if tta:
                out = _tta_predict(model, x, cfg)
            else:
                with torch.autocast(cfg.device, enabled=cfg.use_amp):
                    out = model(x)

            if cfg.mode == "classification":
                train_preds = out.argmax(dim=1) + 1
            elif cfg.mode == "ordinal":
                probs = out if tta else out.sigmoid()
                train_preds = (probs > 0.5).sum(dim=1) + 1
            else:
                train_preds = out.squeeze(1).round().long()

            # clamp(2,4) stays within ±1 of extremes; -1 converts to submission [0,4]
            submission_preds = train_preds.clamp(2, 4).long() - 1

            for pred, pid in zip(submission_preds.cpu().numpy(), patch_ids, strict=False):
                img = Image.fromarray(pred.astype(np.uint8), mode="L")
                img.save(os.path.join(output_dir, f"{pid}.png"))
                count += 1

    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith(".png"):
                zf.write(os.path.join(output_dir, fname), fname)

    print(f"Saved {count} predictions → {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI4Agri submission (stacked UNet)")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best.pt")
    parser.add_argument("--output-dir", type=str, default="submission")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if key not in ("checkpoint", "output_dir", "tta") and value is not None:
            setattr(cfg, key.replace("-", "_"), value)
    predict(args.checkpoint, cfg, args.output_dir, tta=args.tta)
