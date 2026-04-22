"""
Training script for stacked-channel 2D UNet on AgriPotential (viticulture, ±1 accuracy).

Matches the organiser's baseline (El Sakka et al., 2025 supplement):
  - All T timesteps × 10 bands stacked → (B, T*C, H, W) input
  - AdamW, lr=1e-5, no scheduler, ordinal BCE loss

Usage:
    uv run python train.py                                # defaults
    uv run python train.py --epochs 500 --lr 1e-4        # override
    uv run python train.py --mode classification          # cross-entropy
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
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from unet2d import UNet2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_model(cfg: Config, in_channels: int) -> nn.Module:
    if cfg.mode == "ordinal":
        out_ch = cfg.num_classes - 1  # 4 threshold logits for K=5
    elif cfg.mode == "classification":
        out_ch = cfg.num_classes       # 5 class logits
    else:
        out_ch = 1                     # regression
    return UNet2D(
        in_channels=in_channels,
        out_channels=out_ch,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
    )


def build_criterion(cfg: Config):
    if cfg.mode == "classification":
        return nn.CrossEntropyLoss(ignore_index=-1)
    if cfg.mode == "ordinal":
        return None  # computed inline
    if cfg.loss_fn == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none", beta=cfg.smooth_l1_beta)
    return nn.MSELoss(reduction="none")


def stack_input(data: torch.Tensor) -> torch.Tensor:
    """Flatten time and band dims: (B, T, C, H, W) → (B, T*C, H, W)."""
    B, T, C, H, W = data.shape
    return data.reshape(B, T * C, H, W).contiguous()


def regression_loss_step(criterion, pred, target, ignore_index=0):
    mask = target != ignore_index
    if mask.sum() == 0:
        return pred.sum() * 0.0
    loss_map = criterion(pred.squeeze(1), target.float())
    return loss_map[mask].mean()


def classification_loss_step(criterion, logits, target):
    shifted = target.long() - 1  # 0→-1 (ignore), 1-5→0-4
    return criterion(logits, shifted)


def ordinal_loss_step(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:
    """
    Masked BCE loss over K-1 cumulative thresholds.
    Threshold k fires when true class >= k+2.
    """
    mask = target != ignore_index
    if mask.sum() == 0:
        return logits.sum() * 0.0

    n_thresholds = logits.shape[1]
    thresholds = torch.arange(2, n_thresholds + 2, device=target.device)
    ordinal_targets = (target.unsqueeze(1) >= thresholds[None, :, None, None]).float()
    mask_expanded = mask.unsqueeze(1).expand_as(logits)
    return nn.functional.binary_cross_entropy_with_logits(
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
    total_loss, n_batches = 0.0, 0

    for i, (data, labels, _doys, _ids) in enumerate(loader):
        x = stack_input(data.to(cfg.device, non_blocking=True))
        labels = labels.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            out = model(x)
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
            print(f"  [epoch {epoch+1}] batch {i+1}  loss={total_loss/n_batches:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: Config):
    model.eval()
    total_pm1, total_exact, total_mae, total_pixels = 0.0, 0.0, 0.0, 0

    for data, labels, _doys, _ids in loader:
        x = stack_input(data.to(cfg.device, non_blocking=True))
        labels = labels.to(cfg.device, non_blocking=True)

        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            out = model(x)

        if cfg.mode == "classification":
            preds = out.argmax(dim=1) + 1
        elif cfg.mode == "ordinal":
            preds = (out.sigmoid() > 0.5).sum(dim=1) + 1
        else:
            preds = out.squeeze(1).round().clamp(2, 4).long()

        n_valid = (labels != cfg.ignore_index).sum().item()
        if n_valid == 0:
            continue
        total_pm1   += pm1_accuracy(preds, labels, cfg.ignore_index) * n_valid
        total_exact += exact_accuracy(preds, labels, cfg.ignore_index) * n_valid
        total_mae   += mae(preds, labels, cfg.ignore_index) * n_valid
        total_pixels += n_valid

    return {
        "pm1_acc":   total_pm1   / max(total_pixels, 1),
        "exact_acc": total_exact / max(total_pixels, 1),
        "mae":       total_mae   / max(total_pixels, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading datasets …")
    train_ds = UTAEDataset("train", cfg.data_path, cfg.metadata_path, augment=cfg.augment)
    val_ds   = UTAEDataset("val",   cfg.data_path, cfg.metadata_path, augment=False)
    print(f"  train: {len(train_ds)} patches, val: {len(val_ds)} patches")

    # Infer in_channels from first sample (T * num_bands)
    sample_data, *_ = train_ds[0]
    T, C, H, W = sample_data.shape
    in_channels = T * C
    print(f"  Input: {T} timesteps × {C} bands = {in_channels} channels")

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

    model = build_model(cfg, in_channels).to(cfg.device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: UNet2D  {param_count/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_t_mult
        )
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma
        )
    else:
        scheduler = None  # constant LR — matches organiser supplement

    criterion = build_criterion(cfg)
    scaler = GradScaler() if cfg.use_amp and cfg.device == "cuda" else None

    best_pm1 = 0.0
    print(f"Training for {cfg.epochs} epochs (mode={cfg.mode}, lr={cfg.lr:.0e}, scheduler={cfg.scheduler}) …\n")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler, epoch)

        if scheduler is not None:
            scheduler.step()

        metrics = validate(model, val_loader, cfg)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:4d}/{cfg.epochs}  "
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
            "in_channels": in_channels,
        }
        torch.save(ckpt, os.path.join(cfg.save_dir, "last.pt"))

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
    parser = argparse.ArgumentParser(description="Train stacked UNet on AgriPotential")
    parser.add_argument("--mode", choices=["regression", "classification", "ordinal"])
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"])
    parser.add_argument("--cosine-t0", type=int)
    parser.add_argument("--cosine-t-mult", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--step-gamma", type=float)
    parser.add_argument("--loss-fn", choices=["smooth_l1", "mse"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--depth", type=int)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)
    main(cfg)
