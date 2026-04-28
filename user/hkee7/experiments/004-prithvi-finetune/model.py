"""
Prithvi-EO-2.0 fine-tuning for AgriPotential viticulture suitability segmentation.

Architecture:
  - Backbone: Prithvi-EO-2.0-300M (ViT, pretrained on Sentinel-2 / HLS)
  - Head:     UperNet segmentation head via TerraTorch
  - Loss:     Ordinal BCE over K-1 = 4 cumulative thresholds

Input strategy: each timestep is treated as an independent data point.
  - Dataset yields (B, T=1, C=6, H, W) — one frame per sample.
  - Backbone runs with num_frames=1, matching its pretrained configuration
    exactly (no positional-encoding interpolation needed).
  - At inference, predictions for all T frames of the same patch are averaged.

TerraTorch expects (B, C, T, H, W), so we permute before the forward pass.
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from terratorch.datasets import HLSBands
from terratorch.models import EncoderDecoderFactory

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
    Wraps Prithvi-EO-2.0 (T=1) + UperNet head in a Lightning module.

    Input convention:
      - Training/val:  batch = (x, y, doy)  where x is (B, 1, C, H, W)
      - The TerraTorch PixelWiseModel expects (B, C, T, H, W); we permute inside forward().

    Because T=1, the neck is simply:
      SelectIndices → ReshapeTokensToImage → InterpolateToPyramidal
    No TemporalMeanPool is needed (T*embed_dim == embed_dim when T=1).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # TerraTorch builds backbone + UperNet head in one call.
        # Neck pipeline (T=1):
        #   SelectIndices          → 4 × (B, H_p*W_p+1, embed_dim) token lists
        #   ReshapeTokensToImage   → 4 × (B, embed_dim, H_p, W_p) spatial maps
        #   InterpolateToPyramidal → 4-scale pyramid  [bilinear, no params]
        self.model = model_factory.build_model(
            task="segmentation",
            backbone=cfg.backbone,
            backbone_pretrained=True,
            backbone_bands=_HLS_BANDS,
            backbone_num_frames=1,
            backbone_img_size=cfg.img_size,
            necks=[
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {
                    "name": "ReshapeTokensToImage",
                    "effective_time_dim": 1,
                    "remove_cls_token": True,
                },
                {"name": "InterpolateToPyramidal", "scale_factor": 2, "mode": "nearest"},
            ],
            decoder="UperNetDecoder",
            decoder_channels=256,
            head_dropout=0.1,
            num_classes=cfg.num_classes - 1,  # K-1 ordinal thresholds
        )

        if cfg.freeze_backbone:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            frozen = sum(p.numel() for p in self.model.encoder.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"[init] backbone frozen ({frozen/1e6:.1f} M params); "
                f"trainable (neck+decoder): {trainable/1e6:.1f} M params",
                flush=True,
            )
        else:
            if hasattr(self.model.encoder, "set_grad_checkpointing"):
                self.model.encoder.set_grad_checkpointing(True)
                print("[init] full fine-tuning with gradient checkpointing enabled", flush=True)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"[init] full fine-tuning: {total/1e6:.1f} M trainable params", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T=1, C, H, W)  — dataloader format
        Returns:
            logits: (B, K-1, H, W)
        """
        x_in = x.permute(0, 2, 1, 3, 4)  # → (B, C, T=1, H, W)
        result = self.model(x_in)
        return result.output if hasattr(result, "output") else result

    def training_step(self, batch, batch_idx):
        x, y, _doy = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, self.cfg.ignore_index)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _doy = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, self.cfg.ignore_index)
        preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1
        acc = pm1_accuracy(preds, y, self.cfg.ignore_index)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_pm1", acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [{"params": head_params, "lr": self.cfg.lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.cfg.lr * 0.1})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
