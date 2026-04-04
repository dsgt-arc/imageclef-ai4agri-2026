"""
Smoke-test for 003-presto-finetune.

Verifies:
  1. PrestoDataset reads precomputed chunks correctly
  2. PrestoOrdinal forward pass produces correct output shape
  3. Ordinal loss is finite and backprop works
  4. Ordinal predictions are in [1, 5]

Run from the experiment directory:
    python test.py
    python test.py --device cpu   # if no GPU
"""

from __future__ import annotations

import argparse
import sys

import torch
from config import Config
from model import PrestoOrdinal


def test_model_shapes(device: str = "cpu"):
    """Forward pass shape and prediction range checks using random tensors."""
    print("=== test_model_shapes ===")
    cfg = Config()

    B, T, C, H, W = 2, 34, 10, 32, 32  # small patch for speed
    K = cfg.num_classes  # 5

    x = torch.rand(B, T, C, H, W, device=device)
    labels = torch.randint(0, 6, (B, H, W), device=device)  # 0=unlabelled, 1-5

    model = PrestoOrdinal(
        num_classes=K,
        head_hidden_dim=cfg.head_hidden_dim,
        freeze_encoder=True,
    ).to(device)

    # --- forward ---
    logits = model(x)
    assert logits.shape == (
        B,
        K - 1,
        H,
        W,
    ), f"Expected ({B}, {K-1}, {H}, {W}), got {logits.shape}"
    print(f"  logits shape: {logits.shape}  ✓")

    # --- loss ---
    from train import ordinal_loss_step

    loss = ordinal_loss_step(logits, labels, ignore_index=0)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"  loss: {loss.item():.4f}  ✓")

    # --- backprop (head only) ---
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.head.parameters())
    assert grad_ok, "Head parameters have no gradient"
    print("  head gradients present  ✓")

    # --- predictions in [1, 5] ---
    model.eval()
    with torch.no_grad():
        preds = model.predict(x)
    assert preds.shape == (B, H, W), f"Pred shape wrong: {preds.shape}"
    assert (
        preds.min() >= 1 and preds.max() <= 5
    ), f"Preds out of [1,5]: {preds.min()}, {preds.max()}"
    print(f"  preds range: [{preds.min().item()}, {preds.max().item()}]  ✓")

    print()


def test_dataset(data_path: str = "data/precomputed_tensors"):
    """Dataset smoke-test — only runs if precomputed tensors exist."""
    import os

    train_dir = os.path.join(data_path, "train")
    if not os.path.isdir(train_dir):
        print(f"=== test_dataset SKIPPED (no tensors at {data_path}) ===\n")
        return

    print("=== test_dataset ===")
    from dataset import PrestoDataset

    ds = PrestoDataset("train", chunk_dir=data_path)
    print(f"  dataset size: {len(ds)} patches")

    data, label, doys = ds[0]
    print(f"  data shape:  {data.shape}")  # (T, C, H, W)
    print(f"  label shape: {label.shape}")  # (H, W)
    print(f"  doys shape:  {doys.shape}")  # (T,)
    assert data.shape[1] == 10, f"Expected 10 bands, got {data.shape[1]}"
    assert data.min() >= 0 and data.max() <= 1.01, "Normalisation out of range"
    assert label.min() >= 0 and label.max() <= 5
    print("  all checks passed  ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-path", default="data/precomputed_tensors")
    args = parser.parse_args()

    try:
        test_model_shapes(device=args.device)
        test_dataset(data_path=args.data_path)
        print("All tests passed.")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
