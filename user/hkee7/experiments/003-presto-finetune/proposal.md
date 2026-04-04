# Proposal: Presto Fine-tune for Viticulture Crop Suitability

**Status**: proposed
**Created**: 2026-04-04
**Author**: hkee7

## Hypothesis

Fine-tuning a pre-trained Presto encoder on the AgriPotential viticulture task
will outperform U-TAE trained from scratch on ±1 accuracy, because Presto has
already learned rich spectrotemporal representations from global Sentinel-2
time series and requires far less labelled data to specialise.

## Background

The AI4Agri ImageCLEF 2026 Subtask 1 requires pixel-level prediction of crop
suitability on 5 ordinal levels (1=very low … 5=very high) from 34 Sentinel-2
timesteps across 2017–2019. The evaluation metric is **±1 accuracy**.

**Presto** (Pretrained Remote Sensing Transformer, Tseng et al. NeurIPS 2023)
is a lightweight pixel-timeseries transformer (~3M params) pre-trained on
global Sentinel-2 + Sentinel-1 + ERA5 data using masked autoencoding. It was
the backbone of ESA's WorldCereal global crop-type mapping system, making it
highly domain-relevant.

Key advantages over U-TAE from scratch:
- Pre-trained temporal representations capture seasonal vegetation cycles
- Flexible sequence length — our 34-timestep series drops in without modification
- Masking heads allow robust handling of any subset of S2 bands
- Pixel-level design maps naturally to our precomputed `[T, C, H, W]` chunks
- Two operating modes: frozen encoder (fast, few-shot) vs full fine-tune

## Method

### Approach

Reshape precomputed `[T, C, H, W]` tensors to `[B·H·W, T, C]` pixel-timeseries,
pass through Presto encoder (`Presto.load_pretrained()`), reshape embeddings back
to `[B, H, W, D]`, attach a lightweight spatial head, and train with ordinal BCE
loss. Two stages:

1. **Stage 1 — Feature extraction**: freeze encoder, train only head (5–10 epochs)
2. **Stage 2 — Full fine-tune**: unfreeze encoder, end-to-end with lower LR

### Setup

| Component    | Details |
|--------------|---------|
| Data         | AgriPotential viticulture, train/val splits, 128×128 patches |
| Model        | Presto encoder (frozen → unfrozen) + Conv2d ordinal head |
| Compute      | 1× GPU (A100 or V100), ~10–20 min/epoch estimated |
| Dependencies | torch, presto-ssast (nasaharvest/presto), rasterio, pandas, polars |
| Code         | `user/hkee7/experiments/003-presto-finetune/` |

### Architecture

```
Input [B, T, C, H, W]
       ↓ reshape
Pixels [B·H·W, T, C]  (mask=None, latlons=None)
       ↓ Presto encoder
Embeds [B·H·W, D=128]
       ↓ reshape
Map    [B, H, W, D]  → permute → [B, D, H, W]
       ↓ Head: Conv2d(D→64, 3×3) → ReLU → Conv2d(64→K-1, 1×1)
Logits [B, K-1, H, W]  (ordinal thresholds)
```

Loss: ordinal BCE (same as 002-utae), ignore_index=0.

### Variables

- **Independent**: frozen vs fine-tuned encoder; LR schedule
- **Dependent**: ±1 accuracy, exact accuracy, MAE
- **Controlled**: Loss (ordinal BCE), head architecture, data splits

## Evaluation

### Metrics

| Metric      | Description |
|-------------|-------------|
| ±1 accuracy | Primary: fraction where \|pred − target\| ≤ 1 |
| Exact acc   | Secondary: exact pixel accuracy |
| MAE         | Mean absolute error on labelled pixels |

### Baseline

U-TAE ordinal from experiment 002.  Random baseline ≈ 52% ±1 accuracy.

### Success Criteria

- **Confirm if**: ±1 accuracy > 75% on validation set (better than U-TAE from scratch)
- **Marginal if**: 70%–75% (comparable to U-TAE, still validates transfer learning)
- **Reject if**: < 65% (no benefit vs training from scratch)

## Limitations

- Presto is pixel-only — the head has a small receptive field (3×3 Conv); larger
  spatial context requires a deeper head or skip connections from the patch
- Presto was pre-trained at monthly cadence; our 34-timestep series uses
  day-of-year offsets which differ slightly from its training distribution
- lat/lon metadata not currently available in our precomputed tensors (omitted;
  Presto can run without it via masking)
