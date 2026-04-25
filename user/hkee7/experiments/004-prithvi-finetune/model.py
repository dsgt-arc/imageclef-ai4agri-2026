"""
Prithvi-EO-2.0 fine-tuning for AgriPotential viticulture suitability segmentation.

Architecture:
  - Backbone: Prithvi-EO-2.0-300M (ViT, pretrained on Sentinel-2 / HLS time series)
  - Head:     UperNet segmentation head via TerraTorch
  - Loss:     Ordinal BCE over K-1 = 4 cumulative thresholds

Input: (B, T, C=6, H, W) — 6 HLS bands selected from the 10 S2 bands.
TerraTorch expects (B, C, T, H, W), so we permute before the forward pass.
"""

import logging

import lightning.pytorch as pl
import torch
import torch.nn as nn
from terratorch.datasets import HLSBands
from terratorch.models import EncoderDecoderFactory
from terratorch.registry import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

model_factory = EncoderDecoderFactory()

# 6 HLS bands Prithvi-EO-2.0 was pretrained on
_HLS_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]

# Embed dim for prithvi_eo_v2_300
_PRITHVI_300M_EMBED_DIM = 1024


def _probe_effective_time_dim(backbone_name: str, bands, num_frames: int, img_size: int) -> int:
    """Build a lightweight (no-pretrained) backbone to read its actual effective_time_dim.

    PrithviViT sets out_channels[i] = embed_dim * T_patches, where
    T_patches = input_size[0] / patch_size[0].  Dividing by embed_dim gives us
    the exact value ReshapeTokensToImage must use, regardless of any internal
    default that overrides our backbone_num_frames request.
    """
    probe = BACKBONE_REGISTRY.build(
        backbone_name,
        pretrained=False,
        bands=bands,
        num_frames=num_frames,
        img_size=img_size,
    )
    eff_t = probe.out_channels[0] // _PRITHVI_300M_EMBED_DIM
    del probe
    logger.info(
        "[prithvi] probe backbone: out_channels[0]=%d  → effective_time_dim=%d "
        "(requested num_frames=%d)",
        eff_t * _PRITHVI_300M_EMBED_DIM,
        eff_t,
        num_frames,
    )
    if eff_t != num_frames:
        logger.warning(
            "[prithvi] effective_time_dim (%d) differs from cfg.num_frames (%d). "
            "Using %d for ReshapeTokensToImage.",
            eff_t,
            num_frames,
            eff_t,
        )
    return eff_t


def pm1_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 0) -> float:
    mask = target != ignore_index
    if not torch.any(mask):
        return 0.0
    err = (pred[mask] - target[mask]).abs()
    return (err <= 1).float().mean().item()


def ordinal_loss_step(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0
) -> torch.Tensor:
    """Binary cross-entropy over K-1 cumulative ordinal thresholds."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return logits.sum() * 0.0

    n_thresholds = logits.shape[1]
    # threshold k fires when true class >= k+2  (classes are 1–5)
    thresholds = torch.arange(2, n_thresholds + 2, device=target.device)
    ordinal_targets = (target.unsqueeze(1) >= thresholds[None, :, None, None]).float()
    mask_expanded = mask.unsqueeze(1).expand_as(logits)

    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask_expanded], ordinal_targets[mask_expanded]
    )


class PrithviSegmentation(pl.LightningModule):
    """
    Wraps Prithvi-EO-2.0 + UperNet head in a Lightning module.

    The TerraTorch PixelWiseModel expects input in (B, C, T, H, W) order.
    We accept (B, T, C, H, W) from the dataloader and permute internally.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Probe the backbone (no pretrained weights) to discover the actual
        # effective_time_dim.  PrithviViT sets
        #   out_channels[i] = embed_dim * T_patches
        # so dividing by embed_dim is always self-consistent with what
        # ReshapeTokensToImage will actually receive.
        effective_t = _probe_effective_time_dim(
            cfg.backbone, _HLS_BANDS, cfg.num_frames, cfg.img_size
        )

        # TerraTorch builds the backbone + UperNet head in one call.
        # SelectIndices picks 4 evenly-spaced transformer layers for UperNet's FPN.
        # ReshapeTokensToImage converts flat token sequences to spatial feature maps.
        # LearnedInterpolateToPyramidal creates the multi-scale pyramid for UperNet.
        self.model = model_factory.build_model(
            task="segmentation",
            backbone=cfg.backbone,
            backbone_pretrained=True,
            backbone_bands=_HLS_BANDS,
            backbone_num_frames=cfg.num_frames,
            backbone_img_size=cfg.img_size,
            necks=[
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {
                    "name": "ReshapeTokensToImage",
                    "effective_time_dim": effective_t,
                    "remove_cls_token": True,
                },
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            decoder="UperNetDecoder",
            decoder_channels=256,
            head_dropout=0.1,
            num_classes=cfg.num_classes - 1,  # K-1 ordinal thresholds
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)  — dataloader format
        Returns:
            logits: (B, K-1, H, W)
        """
        # TerraTorch expects (B, C, T, H, W)
        return self.model(x.permute(0, 2, 1, 3, 4))

    def training_step(self, batch, batch_idx):
        x, y, _doys = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, self.cfg.ignore_index)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _doys = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, self.cfg.ignore_index)
        preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1
        acc = pm1_accuracy(preds, y, self.cfg.ignore_index)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_pm1", acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Differential LR: backbone gets 10× smaller LR than head
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.lr * 0.1},
                {"params": head_params, "lr": self.cfg.lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
