import argparse
import os

import torch
from config import Config
from dataset import ChunkAwareSampler, PrithviDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from model import PrithviLightning
from torch.utils.data import DataLoader

def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Loading datasets …")
    train_ds = PrithviDataset("train", cfg.data_path, cfg.metadata_path, augment=cfg.augment)
    val_ds = PrithviDataset("val", cfg.data_path, cfg.metadata_path, augment=False)
    
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

    print("Building PrithviLightning model …")
    model = PrithviLightning(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="best_prithvi_{epoch:02d}_{val_pm1:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_pm1",
        mode="max",
    )
    
    logger = CSVLogger(cfg.save_dir, name="logs")

    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        precision="16-mixed" if cfg.use_amp else "32",
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Prithvi-EO-2.0 via PyTorch Lightning")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    cfg = Config()
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs

    train(cfg)
