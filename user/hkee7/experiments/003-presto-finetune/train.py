"""
Training script for Presto fine-tune on AgriPotential (viticulture, ±1 accuracy).

Two-stage training:
  Stage 1 (--stage 1): frozen encoder, head only,  stage1_epochs epochs
  Stage 2 (--stage 2): full fine-tune, lower LR,   stage2_epochs epochs
  Both    (--stage 0): run Stage 1 then Stage 2 sequentially (default)

Usage:
    python train.py                          # full two-stage run
    python train.py --stage 1               # head-only
    python train.py --stage 2 --ckpt <path> # fine-tune from Stage 1 checkpoint
    python train.py --device cpu            # local smoke-test
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from config import Config
from dataset import ChunkAwareSampler, PrestoDataset
from metrics import exact_accuracy, mae, pm1_accuracy
from model import PrestoOrdinal
from torch.amp import GradScaler
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def ordinal_loss_step(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Binary cross-entropy over K-1 cumulative ordinal thresholds.

    Threshold k fires when true class >= k+2:
        k=0 → class >= 2
        k=1 → class >= 3
        k=2 → class >= 4
        k=3 → class >= 5
    """
    mask = target != ignore_index  # (B, H, W)
    if mask.sum() == 0:
        return logits.sum() * 0.0

    n_thresholds = logits.shape[1]
    thresholds = torch.arange(2, n_thresholds + 2, device=target.device)
    # (B, K-1, H, W) binary targets
    ordinal_targets = (target.unsqueeze(1) >= thresholds[None, :, None, None]).float()

    mask_expanded = mask.unsqueeze(1).expand_as(logits)
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask_expanded], ordinal_targets[mask_expanded]
    )


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    scaler: GradScaler | None,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, (data, labels, _doys) in enumerate(loader):
        data = data.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            logits = model(data)  # (B, K-1, H, W)
            loss = ordinal_loss_step(logits, labels, cfg.ignore_index)

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
            print(f"  [epoch {epoch+1}] batch {i+1}  loss={avg:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: Config) -> dict:
    model.eval()
    total_pm1, total_exact, total_mae, total_pixels = 0.0, 0.0, 0.0, 0

    for data, labels, _doys in loader:
        data = data.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)

        with torch.autocast(cfg.device, enabled=cfg.use_amp):
            logits = model(data)  # (B, K-1, H, W)

        preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1  # → [1, 5]

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
# Stage runners
# ---------------------------------------------------------------------------


def _build_loaders(cfg: Config):
    train_ds = PrestoDataset(
        "train", cfg.data_path, cfg.metadata_path, augment=cfg.augment
    )
    val_ds = PrestoDataset("val", cfg.data_path, cfg.metadata_path, augment=False)
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
    return train_loader, val_loader


def _run_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    n_epochs: int,
    lr: float,
    encoder_lr: float | None,
    tag: str,
) -> float:
    """
    Run one training stage. Returns best ±1 accuracy achieved.

    encoder_lr=None → single param group (head-only or same LR everywhere).
    encoder_lr=<float> → separate LR for encoder vs head params.
    """
    if encoder_lr is not None:
        head_params = list(model.head.parameters())
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": encoder_lr},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler() if cfg.use_amp and cfg.device == "cuda" else None

    best_pm1 = 0.0
    best_path = os.path.join(cfg.save_dir, f"best_{tag}.pt")

    print(f"\n{'='*60}")
    print(
        f"  {tag}: {n_epochs} epochs, lr={lr}"
        + (f", enc_lr={encoder_lr}" if encoder_lr else "")
    )
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, scaler, epoch)
        scheduler.step()

        metrics = validate(model, val_loader, cfg)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{n_epochs}  "
            f"loss={train_loss:.4f}  "
            f"±1={metrics['pm1_acc']:.4f}  "
            f"exact={metrics['exact_acc']:.4f}  "
            f"MAE={metrics['mae']:.3f}  "
            f"({elapsed:.0f}s)"
        )

        ckpt = {
            "epoch": epoch + 1,
            "stage": tag,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(cfg.save_dir, f"last_{tag}.pt"))

        if metrics["pm1_acc"] > best_pm1:
            best_pm1 = metrics["pm1_acc"]
            torch.save(ckpt, best_path)
            print(f"  ↑ new best ±1 = {best_pm1:.4f} — saved to {best_path}")

    return best_pm1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config, stage: int, ckpt_path: str | None):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading datasets …")
    train_loader, val_loader = _build_loaders(cfg)

    print("Building model …")
    model = PrestoOrdinal(
        num_classes=cfg.num_classes,
        head_hidden_dim=cfg.head_hidden_dim,
        freeze_encoder=True,  # always start frozen; unfreeze in Stage 2
        presto_weights=cfg.presto_path,
    ).to(cfg.device)

    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable/1e6:.2f}M (encoder frozen)")

    if stage in (0, 1):
        best_s1 = _run_stage(
            model,
            train_loader,
            val_loader,
            cfg,
            n_epochs=cfg.stage1_epochs,
            lr=cfg.stage1_lr,
            encoder_lr=None,
            tag="stage1",
        )
        print(f"\nStage 1 best ±1: {best_s1:.4f}")

    if stage in (0, 2):
        # Unfreeze encoder for Stage 2
        model.unfreeze_encoder()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params (encoder unfrozen): {n_trainable/1e6:.2f}M")

        best_s2 = _run_stage(
            model,
            train_loader,
            val_loader,
            cfg,
            n_epochs=cfg.stage2_epochs,
            lr=cfg.stage2_head_lr,
            encoder_lr=cfg.stage2_lr,
            tag="stage2",
        )
        print(f"\nStage 2 best ±1: {best_s2:.4f}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Presto on AgriPotential")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0=full two-stage, 1=head only, 2=fine-tune only",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint to resume from"
    )
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--stage1-epochs", type=int)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, help="number of dataloader workers")
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if key in ("stage", "ckpt") or value is None:
            continue
        setattr(cfg, key.replace("-", "_"), value)

    main(cfg, stage=args.stage, ckpt_path=args.ckpt)
