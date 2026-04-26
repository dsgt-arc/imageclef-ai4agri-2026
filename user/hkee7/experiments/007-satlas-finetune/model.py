"""
Swin-V2-B + FPN segmentation model for AgriPotential viticulture suitability.

Architecture
------------
  Encoder : Swin-V2-B via timm, pretrained on ImageNet-22k (or SatLas weights).
             First patch-embed conv inflated from 3 → 40 input channels by
             repeating + rescaling pretrained weights, preserving feature scale.
  Neck    : 4-level FPN projecting [128, 256, 512, 1024] → fpn_channels each.
             Top-down upsampling merges coarse semantic with fine spatial detail.
  Head    : Conv → BN → ReLU → Conv outputting K-1 = 4 ordinal logits per pixel.
             Upsampled back to the original input resolution via bilinear interp.

Loss    : Ordinal binary cross-entropy over 4 cumulative thresholds.
Metric  : ±1 accuracy (val_pm1) — predictions within 1 class of ground truth.
"""

import math

import lightning.pytorch as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss / metric helpers
# ---------------------------------------------------------------------------

def pm1_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 0) -> float:
    mask = target != ignore_index
    if not torch.any(mask):
        return 0.0
    return ((pred[mask] - target[mask]).abs() <= 1).float().mean().item()


def ordinal_loss_step(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0
) -> torch.Tensor:
    """Binary cross-entropy over K-1 cumulative ordinal thresholds.

    Threshold k fires when true class >= k+2 (classes are labelled 1–5).
    """
    mask = target != ignore_index
    if mask.sum() == 0:
        return logits.sum() * 0.0

    n_thresh = logits.shape[1]
    thresholds = torch.arange(2, n_thresh + 2, device=target.device)
    ordinal_targets = (target.unsqueeze(1) >= thresholds[None, :, None, None]).float()
    mask_exp = mask.unsqueeze(1).expand_as(logits)

    return F.binary_cross_entropy_with_logits(logits[mask_exp], ordinal_targets[mask_exp])


# ---------------------------------------------------------------------------
# FPN neck
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """Lightweight Feature Pyramid Network neck.

    Projects multi-scale backbone features to a uniform channel width, then
    merges them top-down.  Outputs the finest-scale merged feature map at
    stride 4 (i.e., H/4 × W/4 for input H × W).

    Args:
        in_channels : list of channel counts from backbone (coarse→fine).
        out_channels: unified channel width for all FPN levels.
    """

    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        # Lateral 1×1 projections — applied at each scale
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        # 3×3 output convs to smooth after adding neighbours
        self.outputs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [p0, p1, p2, p3] from fine to coarse (stride 4 … 32)
        Returns:
            Finest-scale merged feature map (stride-4 resolution).
        """
        # Project all levels
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down merge: start from coarsest, upsample + add
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # Apply output convs; return finest level only (other levels unused for dense pred)
        return self.outputs[0](laterals[0])


# ---------------------------------------------------------------------------
# Main Lightning module
# ---------------------------------------------------------------------------

class SatlasSegmentation(pl.LightningModule):
    """Swin-V2-B + FPN for pixel-wise ordinal viticulture suitability prediction."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # ------------------------------------------------------------------
        # Build Swin-V2-B backbone with in_chans=40.
        # timm automatically inflates the 3-channel patch-embed weight to
        # 40 channels via adapt_input_conv (repeat-and-rescale).
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(
            cfg.encoder,
            pretrained=(cfg.satlas_checkpoint == ""),
            features_only=True,
            in_chans=cfg.in_channels,
            out_indices=(0, 1, 2, 3),  # strides: 4, 8, 16, 32
        )

        if cfg.satlas_checkpoint:
            self._load_satlas_weights(cfg.satlas_checkpoint, cfg.in_channels)

        # Feature channel counts at each stride level
        feat_channels = self.backbone.feature_info.channels()   # [128, 256, 512, 1024]
        print(f"[init] backbone feature channels: {feat_channels}", flush=True)

        # ------------------------------------------------------------------
        # FPN neck + segmentation head
        # ------------------------------------------------------------------
        self.fpn = FPNDecoder(feat_channels, cfg.fpn_channels)

        self.head = nn.Sequential(
            nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.fpn_channels, cfg.num_classes - 1, 1),  # K-1 ordinal thresholds
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[init] total params: {total / 1e6:.1f} M", flush=True)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_satlas_weights(self, ckpt_path: str, in_channels: int):
        """Load SatLas pretrained backbone weights, adapting the first conv."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # SatLas checkpoints may be wrapped under "state_dict" or "backbone"
        if "state_dict" in state:
            state = state["state_dict"]
        if "backbone" in state:
            state = state["backbone"]

        # Strip any "backbone." prefix if present
        state = {k.removeprefix("backbone."): v for k, v in state.items()}

        # Adapt the patch-embed conv for our in_channels
        embed_key = next(
            (k for k in state if "patch_embed" in k and "proj.weight" in k), None
        )
        if embed_key is not None:
            w = state[embed_key]          # (out_ch, orig_in_ch, kH, kW)
            orig_in = w.shape[1]
            if orig_in != in_channels:
                w = _inflate_conv_weight(w, in_channels)
                state[embed_key] = w

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f"[init] SatLas weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}", flush=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 40, H, W) temporal statistics tensor
        Returns:
            logits: (B, K-1, H, W) at original input resolution
        """
        features = self.backbone(x)         # list of 4 tensors, fine→coarse
        fused = self.fpn(features)          # (B, fpn_ch, H/4, W/4)
        logits = self.head(fused)           # (B, K-1, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, self.cfg.ignore_index)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, self.cfg.ignore_index)
        preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1
        acc = pm1_accuracy(preds, y, self.cfg.ignore_index)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_pm1", acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.fpn.parameters()) + list(self.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.cfg.lr},
                {"params": backbone_params, "lr": self.cfg.lr * self.cfg.backbone_lr_scale},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _inflate_conv_weight(weight: torch.Tensor, new_in_channels: int) -> torch.Tensor:
    """Inflate a pretrained conv weight from orig_in_ch → new_in_channels.

    Repeats the pretrained weights along the input-channel axis and rescales
    so that the expected activation magnitude is preserved.
    """
    orig_in = weight.shape[1]
    repeat = math.ceil(new_in_channels / orig_in)
    w = weight.repeat(1, repeat, 1, 1)[:, :new_in_channels]
    return w * (orig_in / new_in_channels)
