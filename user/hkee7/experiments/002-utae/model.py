"""
U-TAE model wrappers for the AgriPotential crop-suitability task.

Three modes are supported:
  1. **Regression** — single continuous output, Smooth-L1 loss.
  2. **Classification** — K-channel logits, cross-entropy loss (paper default).
  3. **Ordinal** — K-1 cumulative threshold logits, BCE loss per threshold.
     Recommended: the AgriPotential paper found ordinal outperforms both
     scalar regression and one-hot classification on this task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from utae.utae import UTAE


class UTAERegression(nn.Module):
    """
    U-TAE with a 1-channel regression head.

    Output is a raw continuous value per pixel (unbounded).
    At inference, clamp to [2, 4] and round for ±1 accuracy.
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
            out_conv=[32, 1],
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
            (B, H, W) continuous prediction (unbounded)
        """
        out = self.utae(x, batch_positions=batch_positions)  # (B, 1, H, W)
        return out.squeeze(1)  # (B, H, W)

    def predict(self, x: torch.Tensor, batch_positions=None) -> torch.Tensor:
        """Return integer class predictions in training label space [1, 5]."""
        with torch.no_grad():
            raw = self.forward(x, batch_positions)
            return raw.round().clamp(2, 4).long()


class UTAEClassification(nn.Module):
    """
    U-TAE with a K-channel classification head — the paper's original design.

    The decoder's highest-resolution feature map d1 has K channels (one per
    class), interpreted as logits and supervised with cross-entropy loss.

    Training labels are 1–5 (0 = unlabelled/ignore).
    At inference, argmax + 1 maps logits back to the [1, 5] label space.
    """

    def __init__(
        self,
        input_dim: int = 10,
        num_classes: int = 5,
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
        self.num_classes = num_classes
        self.utae = UTAE(
            input_dim=input_dim,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=[32, num_classes],  # K-channel logits — paper default
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
            (B, K, H, W) raw logits (no softmax — use CrossEntropyLoss directly)
        """
        return self.utae(x, batch_positions=batch_positions)  # (B, K, H, W)

    def predict(self, x: torch.Tensor, batch_positions=None) -> torch.Tensor:
        """Return integer class predictions in training label space [1, 5]."""
        with torch.no_grad():
            logits = self.forward(x, batch_positions)  # (B, K, H, W)
            return logits.argmax(dim=1) + 1  # 0-indexed → [1, 5]


class UTAEOrdinal(nn.Module):
    """
    U-TAE with a cumulative ordinal regression head.

    For K=5 classes the model outputs K-1=4 threshold logits per pixel.
    Threshold k (0-indexed) fires when the true class > k+1, i.e. the pixel
    belongs to class k+2 or higher.  The full ordinal encoding is:

        class 1 → [0, 0, 0, 0]
        class 2 → [1, 0, 0, 0]
        class 3 → [1, 1, 0, 0]
        class 4 → [1, 1, 1, 0]
        class 5 → [1, 1, 1, 1]

    Loss: mean binary cross-entropy across all 4 thresholds and labelled pixels.

    Inference: pred = 1 + number of thresholds that exceed 0.5.

    The AgriPotential dataset paper (El Sakka et al., 2025) found this
    representation consistently outperforms both scalar regression and one-hot
    classification for ±1 accuracy on this task.
    """

    def __init__(
        self,
        input_dim: int = 10,
        num_classes: int = 5,
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
        self.num_classes = num_classes
        n_thresholds = num_classes - 1  # 4 thresholds for 5 classes
        self.utae = UTAE(
            input_dim=input_dim,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=[32, n_thresholds],
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
            (B, K-1, H, W) raw threshold logits (apply sigmoid for probabilities)
        """
        return self.utae(x, batch_positions=batch_positions)  # (B, K-1, H, W)

    def predict(self, x: torch.Tensor, batch_positions=None) -> torch.Tensor:
        """Return integer class predictions in training label space [1, 5]."""
        with torch.no_grad():
            logits = self.forward(x, batch_positions)  # (B, K-1, H, W)
            # Count how many thresholds fire; add 1 for base class.
            return (logits.sigmoid() > 0.5).sum(dim=1) + 1  # → [1, 5]
