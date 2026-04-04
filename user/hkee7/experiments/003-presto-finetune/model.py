"""
Presto-based model for AgriPotential pixel-level ordinal regression.

Uses single_file_presto.py (vendored from nasaharvest/presto) to avoid
installing the full presto package and its heavy research-pipeline deps
(openmapflow, earthengine-api, cropharvest, …).

Architecture:
    Input  [B, T, C, H, W]
           ↓ select S2 bands → map to Presto's 17-band layout
    Pixels [B·H·W, T, 17]
           ↓ Presto Encoder (eval_task=True)
    Embeds [B·H·W, D=128]
           ↓ reshape back to spatial
    Map    [B, D, H, W]
           ↓ lightweight Conv2d head
    Logits [B, K-1, H, W]  (K-1 ordinal thresholds)

Band mapping
------------
Presto's BANDS_GROUPS_IDX expects up to 17 channels:
    [0,1]      S1 (SAR) — we zero-fill (not available)
    [2,3,4]    S2 RGB (B2, B3, B4)
    [5,6,7]    S2 Red Edge (B5, B6, B7)
    [8]        S2 NIR 10m (B8)
    [9]        S2 NIR 20m (B8A)
    [10,11]    S2 SWIR (B11, B12)
    [12,13]    ERA5 — zero-fill
    [14,15]    SRTM — zero-fill
    [16]       NDVI — derived from B8, B4

Our 10-band stack order (from precomputed tensors, standard ESA ordering):
    Band index  ESA name  Presto slot
    0           B2        2
    1           B3        3
    2           B4        4
    3           B5        5
    4           B6        6
    5           B7        7
    6           B8        8
    7           B8A       9
    8           B11       10
    9           B12       11
"""

from __future__ import annotations

import os
import urllib.request

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Pretrained weights
# ---------------------------------------------------------------------------

PRESTO_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/nasaharvest/presto/main/data/default_model.pt"
)
PRESTO_WEIGHTS_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "presto", "default_model.pt"
)



def _ensure_weights(path: str | None) -> str:
    """Download pretrained weights if not cached; return local path."""
    target = path or PRESTO_WEIGHTS_CACHE
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"Downloading Presto weights → {target}")
        urllib.request.urlretrieve(PRESTO_WEIGHTS_URL, target)
        print("  done.")
    return target


# ---------------------------------------------------------------------------
# Band-mapping helper
# ---------------------------------------------------------------------------

NUM_PRESTO_BANDS = 17  # Presto's full expected input width


def _to_presto_bands(x: torch.Tensor) -> torch.Tensor:
    """
    Map our 10-band Sentinel-2 tensor to Presto's 17-band layout.

    Args:
        x: (N, T, 10) — our S2 bands [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]

    Returns:
        (N, T, 17) with zeros for S1, ERA5, SRTM; NDVI derived from B8, B4.
    """
    N, T, _ = x.shape
    out = torch.zeros(N, T, NUM_PRESTO_BANDS, dtype=x.dtype, device=x.device)

    # S2 optical bands
    out[:, :, 2]  = x[:, :, 0]   # B2  → S2_RGB[0]
    out[:, :, 3]  = x[:, :, 1]   # B3  → S2_RGB[1]
    out[:, :, 4]  = x[:, :, 2]   # B4  → S2_RGB[2]
    out[:, :, 5]  = x[:, :, 3]   # B5  → S2_Red_Edge[0]
    out[:, :, 6]  = x[:, :, 4]   # B6  → S2_Red_Edge[1]
    out[:, :, 7]  = x[:, :, 5]   # B7  → S2_Red_Edge[2]
    out[:, :, 8]  = x[:, :, 6]   # B8  → S2_NIR_10m
    out[:, :, 9]  = x[:, :, 7]   # B8A → S2_NIR_20m
    out[:, :, 10] = x[:, :, 8]   # B11 → S2_SWIR[0]
    out[:, :, 11] = x[:, :, 9]   # B12 → S2_SWIR[1]

    # NDVI = (B8 - B4) / (B8 + B4 + 1e-6)  → slot 16
    nir, red = x[:, :, 6], x[:, :, 2]
    out[:, :, 16] = (nir - red) / (nir + red + 1e-6)

    # S1 [0,1], ERA5 [12,13], SRTM [14,15] remain zero (masked by Presto)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PrestoOrdinal(nn.Module):
    """
    Presto Encoder + ordinal regression head for dense pixel prediction.

    Parameters
    ----------
    num_classes : int
        Ordinal classes (K=5 → K-1=4 threshold logits per pixel).
    head_hidden_dim : int
        Channels in the Conv2d spatial head.
    freeze_encoder : bool
        Stage 1: True (head-only). Call `unfreeze_encoder()` for Stage 2.
    presto_weights : str | None
        Path to pretrained .pt file. None = download to ~/.cache/presto/.
    """

    def __init__(
        self,
        num_classes: int = 5,
        head_hidden_dim: int = 64,
        freeze_encoder: bool = True,
        presto_weights: str | None = None,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.freeze_encoder = freeze_encoder

        # Import from the vendored single-file copy — zero package deps
        from single_file_presto import Presto

        # Presto defaults to max 24 timesteps. Our data has 34.
        # The positional embeddings are frozen sinusoids, so we can just increase
        # max_sequence_length and skip loading the pos_embed from the state dict.
        presto = Presto.construct(max_sequence_length=34)
        weights_path = _ensure_weights(presto_weights)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Drop the frozen positional embeddings so they don't size-mismatch
        state_dict.pop("encoder.pos_embed", None)
        state_dict.pop("decoder.pos_embed", None)
        presto.load_state_dict(state_dict, strict=False)

        self.encoder = presto.encoder
        self.embed_dim = self.encoder.embedding_size  # 128

        n_thresholds = num_classes - 1  # 4 for K=5
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, head_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_hidden_dim, n_thresholds, kernel_size=1),
        )

        if freeze_encoder:
            self._freeze_encoder()

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        """Switch to Stage 2: allow encoder gradients."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        # keep positional / month embeddings frozen (they are not learned)
        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)
        self.encoder.train()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, T, C, H, W)  normalised Sentinel-2 [0, 1], C=10

        Returns
        -------
        logits : (B, K-1, H, W)  raw ordinal threshold logits
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Pixels: (B·H·W, T, C)
        x_pix = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        N = B * H * W

        # Presto processes every pixel as an independent sequence.
        # N = B*H*W can be massive.
        # We perform band-mapping and dummy allocations strictly per-chunk 
        # to ensure peak VRAM stays extremely low permanently.
        embeds_list = []
        for i in range(0, N, self.chunk_size):
            # 1. Take raw 10-band chunk
            chunk_raw = x_pix[i : i + self.chunk_size]
            
            # 2. Map only this chunk to 17-band Presto layout
            chunk_x = _to_presto_bands(chunk_raw)
            
            # 3. Create dummy DynamicWorld and LatLons just for this chunk
            chunk_size_real = chunk_x.shape[0]
            chunk_dw = torch.full((chunk_size_real, T), 9, dtype=torch.long, device=device)
            chunk_ll = torch.zeros(chunk_size_real, 2, dtype=chunk_x.dtype, device=device)

            # 4. Pass through Transformer
            if not self.freeze_encoder:
                # Stage 2: Encoder is unfrozen. If we don't checkpoint, PyTorch will keep  
                # enormous attention activation graphs across all chunks alive in VRAM 
                # until loss.backward() completes. This explodes VRAM linearly with batch size.
                # Checkpointing drops the graphs and perfectly flatlines VRAM to just 1 chunk!
                chunk_x.requires_grad_(True)  # Force graph connection hook
                chunk_embeds = torch.utils.checkpoint.checkpoint(
                    self.encoder,
                    chunk_x,
                    chunk_dw,
                    chunk_ll,
                    None,  # mask
                    0,     # month
                    True,  # eval_task
                    use_reentrant=False,
                )
            else:
                chunk_embeds = self.encoder(
                    x=chunk_x,
                    dynamic_world=chunk_dw,
                    latlons=chunk_ll,
                    mask=None,
                    month=0,
                    eval_task=True,
                )
            embeds_list.append(chunk_embeds)
            
        embeds = torch.cat(embeds_list, dim=0)     # (N, D)

        # Reshape to spatial map
        spatial = embeds.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
        return self.head(spatial)                  # (B, K-1, H, W)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer predictions in [1, 5]."""
        with torch.no_grad():
            logits = self.forward(x)
            return (logits.sigmoid() > 0.5).sum(dim=1) + 1  # → [1, 5]
