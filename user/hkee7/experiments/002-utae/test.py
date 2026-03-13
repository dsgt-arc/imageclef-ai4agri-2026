"""
Generate test predictions for AI4Agri submission.

Outputs a ZIP of PNG segmentation masks named by patch_id.
Pixel values: 0 (very low) to 4 (very high).

Supports optional test-time augmentation (TTA) via --tta flag.
"""

import argparse
import os
import zipfile

import numpy as np
import torch
from config import Config
from dataset import UTAEDataset
from model import UTAEClassification, UTAEOrdinal, UTAERegression
from PIL import Image
from torch.utils.data import DataLoader

# 8 geometric augmentations (original + 7 flips/rotations)
_TTA_AUGMENTS = [
    lambda x: x,
    lambda x: x.flip(-1),
    lambda x: x.flip(-2),
    lambda x: x.flip(-1).flip(-2),
    lambda x: torch.rot90(x, 1, [-2, -1]),
    lambda x: torch.rot90(x, 2, [-2, -1]),
    lambda x: torch.rot90(x, 3, [-2, -1]),
    lambda x: torch.rot90(x, 1, [-2, -1]).flip(-1),
]


def _tta_predict(model, data, doys, cfg):
    """Average predictions over 8 geometric augmentations.

    - classification: averages softmax probabilities, then argmax.
    - ordinal: averages sigmoid probabilities, then threshold count.
    - regression: averages continuous outputs directly.
    Returns the averaged output in a form ready for the prediction step.
    """
    preds = []
    for aug in _TTA_AUGMENTS:
        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            out = model(aug(data), batch_positions=doys)
        if cfg.mode == "classification":
            preds.append(out.softmax(dim=1))
        elif cfg.mode == "ordinal":
            preds.append(out.sigmoid())
        else:
            preds.append(out)
    return torch.stack(preds).mean(0)


def _build_model(cfg: Config) -> torch.nn.Module:
    shared = dict(
        input_dim=cfg.num_bands,
        encoder_widths=cfg.encoder_widths,
        decoder_widths=cfg.decoder_widths,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        d_k=cfg.d_k,
    )
    if cfg.mode == "classification":
        return UTAEClassification(num_classes=cfg.num_classes, **shared)
    if cfg.mode == "ordinal":
        return UTAEOrdinal(num_classes=cfg.num_classes, **shared)
    return UTAERegression(**shared)


def predict(
    checkpoint_path: str,
    cfg: Config,
    output_dir: str = "submission",
    tta: bool = False,
):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=cfg.device)

    # Use the config saved in the checkpoint so architecture always matches.
    saved_cfg: Config = ckpt.get("config", cfg)
    # Allow caller to override device/batch-size for the current machine.
    saved_cfg.device = cfg.device
    saved_cfg.batch_size = cfg.batch_size
    saved_cfg.num_workers = cfg.num_workers
    saved_cfg.pin_memory = cfg.pin_memory
    cfg = saved_cfg

    model = _build_model(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Mode: {cfg.mode}")
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
        for data, _, doys, patch_ids in test_loader:
            data = data.to(cfg.device, non_blocking=True)
            doys = doys.to(cfg.device, non_blocking=True).float()

            if tta:
                out = _tta_predict(model, data, doys, cfg)
            else:
                with torch.autocast(cfg.device, enabled=cfg.use_amp):
                    out = model(data, batch_positions=doys)

            if cfg.mode == "classification":
                # (B, K, H, W) logits or averaged probs after TTA
                train_preds = out.argmax(dim=1) + 1  # → [1, 5]
            elif cfg.mode == "ordinal":
                # (B, K-1, H, W) logits or averaged sigmoid probs after TTA
                probs = out if tta else out.sigmoid()
                train_preds = (probs > 0.5).sum(dim=1) + 1  # → [1, 5]
            else:
                # (B, H, W) continuous
                train_preds = out.round().long()

            # Clamp to [2, 4] in training label space — safe for ±1 metric:
            # clamp(1→2, 5→4) stays within ±1 of true label at the extremes.
            # Subtract 1 to convert to submission space [0, 4].
            submission_preds = train_preds.clamp(2, 4).long() - 1  # → [1, 3]

            for pred, pid in zip(
                submission_preds.cpu().numpy(), patch_ids, strict=False
            ):
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
    parser.add_argument(
        "--tta", action="store_true", help="Enable test-time augmentation"
    )
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if key not in ("checkpoint", "output_dir", "tta") and value is not None:
            setattr(cfg, key.replace("-", "_"), value)
    predict(args.checkpoint, cfg, args.output_dir, tta=args.tta)
