"""
Generate test predictions for AI4Agri submission.

Outputs a ZIP of PNG segmentation masks named by patch_id.
Pixel values: 0 (very low) to 4 (very high).
"""

import argparse
import os
import zipfile

import numpy as np
import torch
from config import Config
from dataset import UTAEDataset
from model import UTAERegression
from PIL import Image
from torch.utils.data import DataLoader


def predict(checkpoint_path: str, cfg: Config, output_dir: str = "submission"):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=cfg.device)
    model = UTAERegression(
        input_dim=cfg.num_bands,
        encoder_widths=cfg.encoder_widths,
        decoder_widths=cfg.decoder_widths,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        d_k=cfg.d_k,
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
        for data, _, doys, patch_ids in test_loader:
            data = data.to(cfg.device, non_blocking=True)
            doys = doys.to(cfg.device, non_blocking=True).float()

            with torch.autocast(cfg.device, enabled=cfg.use_amp):
                raw = model(data, batch_positions=doys)
                preds = raw.round().clamp(2, 4).long()  # train [2,4] → submission [2,4]

            for pred, pid in zip(preds.cpu().numpy(), patch_ids, strict=False):
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
    parser = argparse.ArgumentParser(description="Generate AI4Agri test submission")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best.pt")
    parser.add_argument("--output-dir", type=str, default="submission")
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if key not in ("checkpoint", "output_dir") and value is not None:
            setattr(cfg, key.replace("-", "_"), value)
    predict(args.checkpoint, cfg, args.output_dir)
