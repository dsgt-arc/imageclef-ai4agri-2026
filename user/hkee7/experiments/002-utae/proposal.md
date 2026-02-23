# Proposal: U-TAE Regression for Viticulture Crop Suitability

**Status**: proposed
**Created**: 2026-02-22
**Author**: kang

## Hypothesis

A U-TAE model with a regression head and Smooth-L1 loss will achieve higher ±1 accuracy on the AgriPotential viticulture task than a standard classification approach, because the ordinal nature of the labels (very low → very high) and the ±1 tolerance metric reward predictions that are close to the ground truth even when not exact.

## Background

The AI4Agri ImageCLEF 2026 Subtask 1 requires pixel-level prediction of crop suitability on 5 ordinal levels (1=very low … 5=very high) from multi-temporal Sentinel-2 imagery. The evaluation metric is **±1 accuracy** — a prediction is correct if `|pred − target| ≤ 1`.

U-TAE (U-Net with Temporal Attention Encoder) was designed specifically for Satellite Image Time Series (SITS) segmentation and natively handles variable-length temporal sequences through a lightweight attention mechanism. This makes it well-suited for the 34-timestep AgriPotential data.

Using a regression head (single continuous output + round-and-clamp) instead of classification is expected to better optimise for ±1 accuracy because:
- Regression losses penalise predictions proportionally to their distance from the target
- Near-misses (off by 1) incur small loss, encouraging the model to at least get close
- Classification cross-entropy treats all misclassifications equally

## Method

### Approach

Train U-TAE with a 1-channel regression output on all 34 timesteps and 10 Sentinel-2 bands. Use Smooth-L1 loss with unlabelled-pixel masking. Day-of-year positional encoding from metadata timestamps.

### Setup

| Component | Details |
|-----------|---------|
| Data | AgriPotential viticulture, train/val splits, 128×128 patches |
| Model | U-TAE (encoder [64,64,64,128], decoder [32,32,64,128], 16-head LTAE) |
| Compute | 1× GPU (A100 or V100), ~30 min/epoch estimated |
| Dependencies | torch, rasterio, pandas, numpy |
| Code | `user/kang/experiments/002-utae-viticulture/src/` |

### Procedure

1. Load AgriPotential viticulture data, normalise to [0, 1], extract day-of-year
2. Train U-TAE regression for 50 epochs with AdamW + cosine LR
3. Evaluate ±1 accuracy, exact accuracy, and MAE on validation set
4. Compare regression vs. classification head on same architecture

### Variables

- **Independent**: Loss function (smooth L1 vs. cross-entropy), head type (regression vs. classification)
- **Dependent**: ±1 accuracy, exact accuracy, MAE
- **Controlled**: Architecture (U-TAE), data (viticulture), all hyperparameters except loss/head

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ±1 accuracy | Primary: fraction of pixels where \|pred − target\| ≤ 1 |
| Exact accuracy | Secondary: fraction of exactly correct pixels |
| MAE | Mean absolute error on labelled pixels |

### Baseline

Random prediction baseline: ±1 accuracy ≈ 52% (uniform over 5 classes).

### Success Criteria

- **Confirm if**: ±1 accuracy > 70% on validation set
- **Reject if**: ±1 accuracy < 55% (not meaningfully better than random)

## Limitations

- Day-of-year parsing depends on filename format in metadata.csv
- All 34 timesteps are used, which may be memory-intensive (128×128×34×10)
- No data augmentation in initial run
