# Proposal: Prithvi-EO-2.0 via PyTorch Lightning

**Status**: proposed
**Created**: 2026-04-06
**Author**: hkee7

## Hypothesis

Fine-tuning `Prithvi-EO-2.0` (300M parameters) using `terratorch` and PyTorch Lightning will significantly improve the ±1 accuracy leaderboard test score by mitigating the severe domain shifts and local validation overfitting observed in the Presto (49%) and U-TAE (57%) experiments. Prithvi-EO-v2's HLS (Sentinel-2 + Landsat) pre-trained spatial-spectral features will generalize better.

## Background

The previous Presto experiment achieved excellent validation performance (~78% ±1 accuracy) but degraded catastrophically on the unseen test Leaderboard (49%), demonstrating extreme overfitting to the validation geographical patches. `Prithvi-EO-2.0` is natively designed for massive-scale geospatial multi-temporal data and supports state-of-the-art segmentation adapters.

## Method

### Approach

We are implementing a pure PyTorch Lightning pipeline for `004-prithvi-finetune`, totally independent of the previous native PyTorch pipelines.

1.  **Architecture**: `terratorch.models.SemanticSegmentationTask` (or equivalent Lightning wrapper around custom module).
2.  **Dataset Band Mapping**: Filter the 10 precomputed S2 bands to the 6 standard HLS bands Prithvi expects: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2.
3.  **Loss**: Integration of our custom `ordinal_loss_step` into the `training_step`.
4.  **Submission Space**: Submissions will still be clamped `[1, 3]` effectively bounding prediction variance.

### Setup

| Component | Details |
|-----------|---------|
| Data | AgriPotential tensors, filtered to 6 bands |
| Compute | PACE Phoenix (Blackwell), large VRAM required for 300M params |
| Dependencies | `terratorch`, `lightning`, `timm` |
| Code | `user/hkee7/experiments/004-prithvi-finetune/` |

### Variables

- **Independent**: Use of `Prithvi-EO-2.0` architecture
- **Dependent**: `±1 accuracy` on the official Leaderboard
- **Controlled**: `[1, 3]` clamping out-bound, identical precomputed `.pt` patches

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ±1 accuracy | Fraction where \|pred − target\| <= 1 |

### Baseline

U-TAE baseline (57% ±1 accuracy test), Presto baseline (49% ±1 accuracy test).

### Success Criteria

- **Confirm if**: Test score > 60% (finally breaking the domain shift).
- **Reject if**: The model still suffers massive regression on test indicating possible leaderboard processing divergence.
