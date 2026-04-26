"""
Fine-tune Swin-V2-B + FPN on AgriPotential viticulture suitability.

Usage:
    uv run python train.py                                   # defaults
    uv run python train.py --epochs 50 --batch-size 32      # override
    uv run python train.py --save-dir /path/to/run          # custom output
    uv run python train.py --resume artifacts/last.ckpt     # resume
"""

import argparse
import os

import torch
from config import Config
from dataset import ChunkAwareSampler, SatlasDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from model import SatlasSegmentation
from torch.utils.data import DataLoader


def train(cfg: Config, resume: str | None = None):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading datasets …")
    train_ds = SatlasDataset("train", cfg.data_path, augment=cfg.augment)
    val_ds = SatlasDataset("val", cfg.data_path, augment=False)
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

    print(f"Building SatlasSegmentation (encoder={cfg.encoder}) …")
    model = SatlasSegmentation(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="best_{epoch:03d}_{val_pm1:.4f}",
            save_top_k=1,
            monitor="val_pm1",
            mode="max",
            verbose=True,
        ),
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
        ckpt_path=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SatLas Swin-V2-B on AgriPotential")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--satlas-checkpoint", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    resume_path = args.resume
    for key, value in vars(args).items():
        if key == "resume":
            continue
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)

    train(cfg, resume=resume_path)
