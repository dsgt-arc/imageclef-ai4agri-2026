"""
Training script for U-TAE on AgriPotential (viticulture, ±1 accuracy).

Usage:
    uv run python -m src.train                        # defaults
    uv run python -m src.train --epochs 30 --lr 5e-4  # override
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from config import Config
from dataset import ChunkAwareSampler, UTAEDataset
from metrics import exact_accuracy, mae, pm1_accuracy
from model import UTAEClassification, UTAEOrdinal, UTAERegression
from torch.amp import GradScaler
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_model(cfg: Config) -> nn.Module:
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


def build_criterion(cfg: Config):
    if cfg.mode == "classification":
        # Labels are 1–5; we shift by -1 so unlabelled (0) becomes -1 = ignore.
        return nn.CrossEntropyLoss(ignore_index=-1)
    if cfg.mode == "ordinal":
        return None  # loss is computed inline in ordinal_loss_step
    if cfg.loss_fn == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none", beta=cfg.smooth_l1_beta)
    return nn.MSELoss(reduction="none")


def regression_loss_step(criterion, pred, target, ignore_index=0):
    """Compute masked regression loss (ignore unlabelled pixels)."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return pred.sum() * 0.0  # no labelled pixels → zero grad
    loss_map = criterion(pred, target.float())  # (B, H, W)
    return loss_map[mask].mean()


def classification_loss_step(criterion, logits, target):
    """Compute cross-entropy loss, remapping label 0 (unlabelled) to -1 (ignore)."""
    # Shift: 0→-1 (ignore), 1-5→0-4 (class indices)
    shifted = target.long() - 1  # (B, H, W)
    return criterion(logits, shifted)  # CrossEntropyLoss handles (B,C,H,W) input


def ordinal_loss_step(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0
) -> torch.Tensor:
    """
    Binary cross-entropy loss over K-1 cumulative thresholds.

    Args:
        logits:  (B, K-1, H, W) raw threshold logits from UTAEOrdinal.
        target:  (B, H, W) integer labels in [0, K]; 0 = unlabelled (ignored).

    Threshold k (0-indexed) should fire (target=1) when true class > k+1,
    i.e. true class >= k+2.  For k in [0, 3] and classes [1, 5]:
        k=0 fires for classes 2,3,4,5  (target >= 2)
        k=1 fires for classes 3,4,5    (target >= 3)
        k=2 fires for classes 4,5      (target >= 4)
        k=3 fires for class 5 only     (target >= 5)
    """
    mask = target != ignore_index  # (B, H, W)
    if mask.sum() == 0:
        return logits.sum() * 0.0

    n_thresholds = logits.shape[1]
    # thresholds[k] = k+2; threshold k fires when target >= k+2
    thresholds = torch.arange(2, n_thresholds + 2, device=target.device)  # [2,3,4,5]
    # Expand for broadcasting: (B, K-1, H, W)
    ordinal_targets = (target.unsqueeze(1) >= thresholds[None, :, None, None]).float()

    # Expand mask to all thresholds, select only labelled pixels
    mask_expanded = mask.unsqueeze(1).expand_as(logits)  # (B, K-1, H, W)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask_expanded], ordinal_targets[mask_expanded]
    )


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    cfg: Config,
    scaler: GradScaler | None,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, (data, labels, doys, _) in enumerate(loader):
        data = data.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        doys = doys.to(cfg.device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            out = model(data, batch_positions=doys)
            if cfg.mode == "classification":
                loss = classification_loss_step(criterion, out, labels)
            elif cfg.mode == "ordinal":
                loss = ordinal_loss_step(out, labels, cfg.ignore_index)
            else:
                loss = regression_loss_step(criterion, out, labels, cfg.ignore_index)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % cfg.log_every == 0:
            avg = total_loss / n_batches
            print(
                f"  [epoch {epoch+1}] batch {i+1}  loss={avg:.4f}",
                flush=True,
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: Config):
    model.eval()
    total_pm1, total_exact, total_mae, total_pixels = 0.0, 0.0, 0.0, 0

    for data, labels, doys, _ in loader:
        data = data.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        doys = doys.to(cfg.device, non_blocking=True).float()

        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            out = model(data, batch_positions=doys)

        if cfg.mode == "classification":
            # (B, K, H, W) logits; argmax + 1 → training label space [1, 5]
            preds = out.argmax(dim=1) + 1
        elif cfg.mode == "ordinal":
            # (B, K-1, H, W) threshold logits; count firing thresholds + 1
            preds = (out.sigmoid() > 0.5).sum(dim=1) + 1  # → [1, 5]
        else:
            preds = out.round().clamp(2, 4).long()

        n_valid = (labels != cfg.ignore_index).sum().item()
        if n_valid == 0:
            continue
        total_pm1 += pm1_accuracy(preds, labels, cfg.ignore_index) * n_valid
        total_exact += exact_accuracy(preds, labels, cfg.ignore_index) * n_valid
        total_mae += mae(preds, labels, cfg.ignore_index) * n_valid
        total_pixels += n_valid

    return {
        "pm1_acc": total_pm1 / max(total_pixels, 1),
        "exact_acc": total_exact / max(total_pixels, 1),
        "mae": total_mae / max(total_pixels, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    print("Loading datasets …")
    train_ds = UTAEDataset(
        "train", cfg.data_path, cfg.metadata_path, augment=cfg.augment
    )
    val_ds = UTAEDataset("val", cfg.data_path, cfg.metadata_path, augment=False)

    print(f"  train: {len(train_ds)} patches, val: {len(val_ds)} patches")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=ChunkAwareSampler(train_ds, shuffle=True, seed=cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # ---- Model / optim / loss -----------------------------------------------
    model = build_model(cfg).to(cfg.device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {param_count/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_t_mult
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma
        )

    criterion = build_criterion(cfg)
    scaler = GradScaler() if cfg.use_amp and cfg.device == "cuda" else None

    # ---- Training loop ------------------------------------------------------
    best_pm1 = 0.0
    print(f"Training for {cfg.epochs} epochs …\n")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg, scaler, epoch
        )
        scheduler.step()

        metrics = validate(model, val_loader, cfg)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  "
            f"±1={metrics['pm1_acc']:.4f}  "
            f"exact={metrics['exact_acc']:.4f}  "
            f"MAE={metrics['mae']:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({elapsed:.0f}s)"
        )

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }

        # Always save last checkpoint
        torch.save(ckpt, os.path.join(cfg.save_dir, "last.pt"))

        # Save best by val ±1 accuracy
        if metrics["pm1_acc"] > best_pm1:
            best_pm1 = metrics["pm1_acc"]
            path = os.path.join(cfg.save_dir, "best.pt")
            torch.save(ckpt, path)
            print(f"  ↑ new best ±1 accuracy — saved to {path}")

    print(f"\nDone. Best ±1 accuracy: {best_pm1:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-TAE on AgriPotential")
    parser.add_argument("--mode", choices=["regression", "classification", "ordinal"])
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--loss-fn", choices=["smooth_l1", "mse"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)
    main(cfg)
