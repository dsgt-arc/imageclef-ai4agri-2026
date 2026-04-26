"""
Generate test-set predictions with a fine-tuned Prithvi-EO-2.0 model
and package them into a competition submission ZIP.

Usage:
    uv run python predict.py --checkpoint artifacts/best_prithvi_....ckpt
"""

import argparse
import os
import zipfile

import numpy as np
import torch
from config import Config
from dataset import PrithviDataset
from model import PrithviSegmentation
from PIL import Image
from torch.utils.data import DataLoader


def predict(checkpoint_path: str, cfg: Config, output_dir: str = "submission"):
    print(f"Loading checkpoint: {checkpoint_path}")

    # PyTorch 2.6 defaults weights_only=True, which blocks custom classes like Config.
    torch.serialization.add_safe_globals([Config])
    model = PrithviSegmentation.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    model.to(cfg.device)

    test_ds = PrithviDataset("test", cfg.data_path, cfg.metadata_path, augment=False)
    loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=False,
    )
    print(f"  {len(test_ds)} patches to predict")

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for data, _labels, _doys, patch_ids in loader:
            data = data.to(cfg.device, non_blocking=True)

            with torch.autocast(cfg.device, enabled=cfg.use_amp):
                logits = model(data)                          # (B, K-1, H, W)

            preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1  # → [1, 5] training space
            # Clamp to [2, 4] for ±1 safety at extremes, then subtract 1 for
            # submission space [0, 4] (competition expects 0=very_low … 4=very_high)
            preds = preds.long().clamp(2, 4) - 1             # → [1, 3] submission space

            for pred, pid in zip(preds.cpu().numpy(), patch_ids):
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
    parser = argparse.ArgumentParser("Generate test predictions with Prithvi")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="submission")
    args = parser.parse_args()

    cfg = Config()
    cfg.batch_size = args.batch_size

    predict(args.checkpoint, cfg, args.output_dir)
