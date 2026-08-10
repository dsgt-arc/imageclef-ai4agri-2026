import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from prithvi import (
    ChunkedDataset,
    FreezeBackboneCallback,
    MetricsCallback,
    OrdinalSegmentationTask,
    ShuffleDatasetCallback,
    build_prithvi_model,
)
from unet import ResUNetOrdinal

CHECKPOINTS_DIR = os.path.expandvars('$HOME/scratch/checkpoints/')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_dataset = ChunkedDataset(mode='train', cache_size=4)
    val_dataset = ChunkedDataset(mode='val', cache_size=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model_unet = ResUNetOrdinal(
        in_channels=15,
        num_timesteps=34,
        num_classes=4,
        base_dim=128,
        dropout=0.2,
    ).to(device)
    teacher_path = 'best_unet.pt'
    if os.path.exists(teacher_path):
        model_unet.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        print(f'No teacher checkpoint found at {teacher_path}; using randomly initialized teacher weights.')

    model_prithvi = build_prithvi_model()
    task = OrdinalSegmentationTask(
        epochs=40,
        model=model_prithvi,
        model_args=dict(num_classes=4),
        ignore_index=-1,
        lr=2e-4,
        optimizer='AdamW',
        optimizer_hparams=dict(weight_decay=0.01),
        freeze_backbone=False,
        freeze_decoder=False,
        plot_on_val=False,
    )
    task.set_teacher(model_unet)
    task.raw_bias.data = model_unet.raw_bias.data.clone()
    task.to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        dirpath=os.path.join(CHECKPOINTS_DIR, 'prithvi_final'),
        filename='best',
        save_top_k=1,
        save_last=False,
    )
    metrics_callback = MetricsCallback(save_path='prithvi_loss_curve.png')
    frozen_backbone_callback = FreezeBackboneCallback(unfreeze_epoch=7)
    early_stopping_callback = EarlyStopping(monitor='val/loss', patience=10, mode='min')
    shuffle_dataset_callback = ShuffleDatasetCallback()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        max_epochs=40,
        callbacks=[
            shuffle_dataset_callback,
            frozen_backbone_callback,
            early_stopping_callback,
            checkpoint_callback,
            metrics_callback,
        ],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(task, train_loader, val_loader)


if __name__ == '__main__':
    main()
