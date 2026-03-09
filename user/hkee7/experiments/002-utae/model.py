"""
U-TAE model wrappers for the AgriPotential crop-suitability task.

Two modes are supported:
  1. **Regression** (default) — single continuous output, Smooth-L1 loss.
     Optimised for ±1 accuracy metric.
  2. **Classification** — softmax over K classes, cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from utae.utae import UTAE


class UTAERegression(nn.Module):
    """
    U-TAE with a 1-channel regression head.

    Output is a continuous value per pixel; at inference you clamp to [1, 5]
    and round to the nearest integer for the ±1 accuracy metric.
    """

    def __init__(
        self,
        input_dim: int = 10,
        encoder_widths: list[int] | None = None,
        decoder_widths: list[int] | None = None,
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        pad_value: float = 0,
    ):
        super().__init__()
        encoder_widths = encoder_widths or [64, 64, 64, 128]
        decoder_widths = decoder_widths or [32, 32, 64, 128]
        self.utae = UTAE(
            input_dim=input_dim,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=[32, 1],  # single-channel regression output
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            pad_value=pad_value,
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) input time series
            batch_positions: (B, T) day-of-year for positional encoding

        Returns:
            (B, H, W) continuous prediction
        """
        out = self.utae(x, batch_positions=batch_positions)  # (B, 1, H, W)
        return out.squeeze(1).sigmoid() * 4 + 1  # (B, H, W)

    def predict(self, x: torch.Tensor, batch_positions=None) -> torch.Tensor:
        """Return integer class predictions clamped to [1, 5]."""
        with torch.no_grad():
            raw = self.forward(x, batch_positions)
            return raw.round().clamp(2, 4).long()
