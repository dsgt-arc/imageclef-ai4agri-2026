"""
Presto-based model for AgriPotential pixel-level ordinal regression.

Architecture:
    Input  [B, T, C, H, W]
           ↓ reshape to pixels
    Pixels [B·H·W, T, C]
           ↓ Presto encoder
    Embeds [B·H·W, D=128]
           ↓ reshape back to spatial
    Map    [B, D, H, W]
           ↓ lightweight Conv2d head
    Logits [B, K-1, H, W]  (K-1 ordinal thresholds)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PrestoOrdinal(nn.Module):
    """
    Presto encoder + ordinal regression head for dense prediction.

    Parameters
    ----------
    num_classes : int
        Number of ordinal classes (K=5 → K-1=4 threshold logits).
    head_hidden_dim : int
        Channels in the intermediate Conv2d of the spatial head.
    freeze_encoder : bool
        If True, Presto encoder weights are frozen (Stage 1).
        Call `unfreeze_encoder()` to switch to Stage 2.
    presto_path : str | None
        Path to local Presto weights file, or None to use the default
        pretrained checkpoint downloaded by `Presto.load_pretrained()`.
    """

    def __init__(
        self,
        num_classes: int = 5,
        head_hidden_dim: int = 64,
        freeze_encoder: bool = True,
        presto_path: str | None = None,
    ):
        super().__init__()

        from presto import Presto  # package name: presto (via [tool.uv.sources])

        self.encoder = Presto.load_pretrained(model_path=presto_path)
        self.embed_dim = 128  # Presto's fixed output embedding size

        n_thresholds = num_classes - 1  # 4 for K=5

        # Small spatial head — 3×3 conv to blend neighbouring pixels, then 1×1
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, head_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_hidden_dim, n_thresholds, kernel_size=1),
        )

        if freeze_encoder:
            self._freeze_encoder()

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        """Switch to Stage 2: allow encoder gradients."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        self.encoder.train()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, T, C, H, W)  normalised Sentinel-2 timeseries [0, 1]

        Returns
        -------
        logits : (B, K-1, H, W)  raw ordinal threshold logits
        """
        B, T, C, H, W = x.shape

        # Reshape to pixel timeseries: (B·H·W, T, C)
        x_pix = x.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        x_pix = x_pix.reshape(B * H * W, T, C)  # (N, T, C)

        # Presto forward — no latlons or dynamic_world; mask=None means no masking
        # Presto.forward returns (N, D) mean-pooled embedding
        embeds = self._encode(x_pix)  # (N, D)

        # Reshape back to spatial map
        spatial = embeds.reshape(B, H, W, self.embed_dim)  # (B, H, W, D)
        spatial = spatial.permute(0, 3, 1, 2)  # (B, D, H, W)

        return self.head(spatial)  # (B, K-1, H, W)

    def _encode(self, x_pix: torch.Tensor) -> torch.Tensor:
        """
        Run Presto encoder on pixel timeseries.

        Presto.forward signature (from nasaharvest/presto):
            forward(x, mask=None, dynamic_world=None, latlons=None, month=None,
                    eval_task=True)
            → returns (embedding, tokens) when eval_task=True

        We take the CLS / mean-pooled embedding (first element of the tuple).
        """
        # Presto expects x in its own normalised space. Since we normalise to
        # [0,1] by /10000 and Presto's pretrained normalisation is also
        # reflectance-based, this is compatible.  If results are poor, revisit
        # the per-band mean/std normalisation from Presto's data_utils.
        result = self.encoder(x_pix, mask=None, eval_task=True)
        # result is a tuple (global_embedding, token_embeddings)
        # global_embedding: (N, D)
        if isinstance(result, tuple):
            return result[0]
        return result  # fallback if API differs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer predictions in [1, 5]."""
        with torch.no_grad():
            logits = self.forward(x)  # (B, K-1, H, W)
            return (logits.sigmoid() > 0.5).sum(dim=1) + 1  # → [1, 5]
