"""
Classic 2D U-Net for stacked multispectral time series.

Matches the organiser's AgriPotential baseline (El Sakka et al., 2025):
all T timesteps × C bands are concatenated along the channel dimension and
fed as a single (B, T*C, H, W) image to a 2D UNet.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3×3 convolutions each followed by BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """
    4-level U-Net with configurable input / output channels.

    Channel progression (default, depth=4):
        Encoder: in_ch → 64 → 128 → 256 → 512
        Decoder: 512 → 256 → 128 → 64 → out_ch

    Args:
        in_channels:   Number of input channels (T * num_bands for stacked input).
        out_channels:  Number of output channels:
                         1  for regression
                         5  for classification (K logits)
                         4  for ordinal (K-1 threshold logits, recommended)
        base_channels: Width of the first encoder level (default 64).
        depth:         Number of encoder/decoder levels (default 4).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2**i) for i in range(depth)]

        # Encoder
        self.inc = DoubleConv(in_channels, chs[0])
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            self.encoders.append(
                nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[i], chs[i + 1]))
            )

        # Decoder — upsample then concat skip, then DoubleConv
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_convs.append(
                nn.ConvTranspose2d(chs[i], chs[i - 1], kernel_size=2, stride=2)
            )
            self.dec_blocks.append(DoubleConv(chs[i - 1] * 2, chs[i - 1]))

        # Final 1×1 projection
        self.outc = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, out_channels, H, W)
        """
        skips = []
        x = self.inc(x)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Discard the deepest skip (it's the bottleneck itself)
        x = skips.pop()
        for up, dec in zip(self.up_convs, self.dec_blocks):
            skip = skips.pop()
            x = up(x)
            # Handle odd spatial sizes
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.outc(x)
