"""
Fine-tune Prithvi-EO-2.0-300M on AgriPotential viticulture suitability.

Usage:
    uv run python train.py                              # defaults
    uv run python train.py --epochs 50 --batch-size 4  # override
    uv run python train.py --resume artifacts/last.ckpt # resume from checkpoint
"""

import argparse
import os

import torch
from config import Config
from dataset import ChunkAwareSampler, PrithviDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from model import PrithviSegmentation
from torch.utils.data import DataLoader


def train(cfg: Config, resume: str | None = None):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading datasets …")
    train_ds = PrithviDataset("train", cfg.data_path, cfg.metadata_path, augment=cfg.augment)
    val_ds   = PrithviDataset("val",   cfg.data_path, cfg.metadata_path, augment=False)
    print(f"  train: {len(train_ds)} frame-patch samples, val: {len(val_ds)} frame-patch samples")

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

    print(f"Building PrithviSegmentation (backbone={cfg.backbone}, T=1, single-frame strategy) …")
    model = PrithviSegmentation(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="best_{epoch:03d}_{val_pm1:.4f}",
            save_top_k=1,
            monitor="val_pm1",
            mode="max",
            verbose=True,
        ),
        # Also keep a rolling last checkpoint for resuming
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="last",
            every_n_epochs=1,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        precision=cfg.precision if cfg.use_amp else "32",
        gradient_clip_val=cfg.grad_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callbacks,
        logger=CSVLogger(cfg.save_dir, name="logs"),
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume,   # None = fresh start; path = resume
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tune Prithvi-EO-2.0 on AgriPotential")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .ckpt file to resume training from")
    args = parser.parse_args()

    cfg = Config()
    resume_path = args.resume
    for key, value in vars(args).items():
        if key == "resume":
            continue
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)

    train(cfg, resume=resume_path)
