"""
Generate test-set predictions with the single-frame Prithvi-EO-2.0 model
and package them into a competition submission ZIP.

Single-frame strategy:
  Each of the 34 timesteps of every patch is run through the model independently.
  The sigmoid probabilities are averaged across all frames before thresholding,
  giving a single ordinal prediction per patch.

Usage:
    uv run python predict.py --checkpoint artifacts/best_....ckpt
    uv run python predict.py --checkpoint artifacts/best_....ckpt --split val
    uv run python predict.py --checkpoint artifacts/best_....ckpt --tta
"""

import argparse
import os
import zipfile
from collections import defaultdict

import numpy as np
import torch
from config import Config
from dataset import PrithviDataset
from model import PrithviSegmentation
from PIL import Image
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# TTA helpers (operate on 5-D input: B, T=1, C, H, W)
# ---------------------------------------------------------------------------

_TTA_AUGMENTS = [
    (lambda x: x,                           lambda x: x),
    (lambda x: x.flip(-1),                  lambda x: x.flip(-1)),
    (lambda x: x.flip(-2),                  lambda x: x.flip(-2)),
    (lambda x: x.flip(-1).flip(-2),         lambda x: x.flip(-1).flip(-2)),
    (lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, 3, [-2, -1])),
    (lambda x: torch.rot90(x, 2, [-2, -1]), lambda x: torch.rot90(x, 2, [-2, -1])),
    (lambda x: torch.rot90(x, 3, [-2, -1]), lambda x: torch.rot90(x, 1, [-2, -1])),
    (
        lambda x: torch.rot90(x, 1, [-2, -1]).flip(-1),
        lambda x: torch.rot90(x.flip(-1),  3, [-2, -1]),
    ),
]


def _frame_probs(model: PrithviSegmentation, data: torch.Tensor, cfg: Config, tta: bool) -> torch.Tensor:
    """Return averaged sigmoid probs for one batch of single frames.

    Args:
        data: (B, 1, C, H, W)
    Returns:
        probs: (B, K-1, H, W)  in [0, 1]
    """
    amp_dtype = torch.bfloat16 if "bf16" in cfg.precision else torch.float16

    if not tta:
        with torch.autocast(cfg.device, dtype=amp_dtype, enabled=cfg.use_amp):
            logits = model(data)  # (B, K-1, H, W)
        return logits.sigmoid()

    # 8-fold geometric TTA
    probs_list = []
    for aug_fn, deaug_fn in _TTA_AUGMENTS:
        aug_data = aug_fn(data)
        with torch.autocast(cfg.device, dtype=amp_dtype, enabled=cfg.use_amp):
            logits = model(aug_data)
        probs_list.append(deaug_fn(logits.sigmoid()))
    return torch.stack(probs_list).mean(0)  # (B, K-1, H, W)


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
    torch.serialization.add_safe_globals([Config])
    model = PrithviSegmentation.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    model.to(cfg.device)

    print(f"Split  : {split}")
    print(f"TTA    : {'enabled (8 augmentations)' if tta else 'disabled'}")

    # Dataset yields (data, label, doy, patch_id) for test mode —
    # one item per (patch, frame) pair.
    ds = PrithviDataset(split, cfg.data_path, cfg.metadata_path, augment=False)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=False,
    )
    print(f"  {len(ds)} frame-patch samples to process ({len(ds)} / ~34 ≈ {len(ds)//34} patches)")

    os.makedirs(output_dir, exist_ok=True)

    # Accumulate per-patch probs across all frames.
    # Keys are patch_ids; values are running sum tensors (K-1, H, W) on CPU.
    prob_sum: dict[str, torch.Tensor] = defaultdict(lambda: None)
    frame_count: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            # test mode returns 4-tuple; train/val returns 3-tuple
            if len(batch) == 4:
                data, _labels, _doy, patch_ids = batch
            else:
                data, _labels, _doy = batch
                patch_ids = [f"patch_{i}" for i in range(len(data))]

            data = data.to(cfg.device, non_blocking=True)
            probs = _frame_probs(model, data, cfg, tta)  # (B, K-1, H, W) float32

            for prob, pid in zip(probs.cpu().float(), patch_ids):
                pid = str(pid)
                if prob_sum[pid] is None:
                    prob_sum[pid] = prob.clone()
                else:
                    prob_sum[pid] += prob
                frame_count[pid] += 1

    # Write one PNG per patch
    count = 0
    for pid, total_probs in prob_sum.items():
        n = frame_count[pid]
        avg_probs = total_probs / n                            # (K-1, H, W)
        pred = (avg_probs > 0.5).sum(dim=0).long() + 1        # (H, W) in [1, 5]
        # Map to submission space [0, 4]; clamp to [1, 3] for ±1 boundary safety
        submission_pred = pred.clamp(2, 4) - 1                # → [1, 3]

        img = Image.fromarray(submission_pred.numpy().astype(np.uint8), mode="L")
        img.save(os.path.join(output_dir, f"{pid}.png"))
        count += 1

    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith(".png"):
                zf.write(os.path.join(output_dir, fname), fname)

    print(f"Averaged over {sum(frame_count.values()) // max(count, 1):.0f} frames/patch on average")
    print(f"Saved {count} predictions → {zip_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate test predictions with Prithvi (single-frame)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Lightning .ckpt file")
    parser.add_argument("--output-dir", type=str, default="submission")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (8 flips/rotations per frame)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Frames per batch (default 64; safe for T=1 on most GPUs)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.device:
        cfg.device = args.device

    predict(args.checkpoint, cfg, args.output_dir, tta=args.tta, split=args.split)
