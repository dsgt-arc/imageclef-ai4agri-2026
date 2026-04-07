# Proposal: RandomForest Pixel Baseline (cuML)

**Status**: proposed
**Created**: 2026-04-07
**Author**: hkee7

## Hypothesis

A pixel-wise RandomForest baseline on flattened Sentinel-2 time-series features can
reach a strong ±1 accuracy quickly, and feature importance rankings will expose
whether spectral-temporal signals or a small subset of bands dominate this task.

## Background

This task predicts 5 ordinal crop-suitability classes per pixel from 34x10
Sentinel-2 sequences. Before changing model architecture, a tabular baseline helps
answer two immediate questions:

1. How hard is the task with no spatial context?
2. Which features matter most?

## Method

### Approach

- Flatten each pixel from `(T=34, C=10)` to `340` features.
- Train `RandomForestClassifier` with cuML when available.
- Evaluate on validation pixels using ±1 accuracy, exact accuracy, and MAE.
- Export top feature importances for quick interpretation.

### Setup

| Component | Details |
|-----------|---------|
| Data | `data/precomputed_tensors/{train,val}` |
| Model | `cuml.ensemble.RandomForestClassifier` (fallback: sklearn RF) |
| Features | Pixel-wise flattened temporal-spectral vector (`T*C`) |
| Metrics | ±1 accuracy (primary), exact accuracy, MAE |
| Code | `user/hkee7/experiments/005-rf-cuml-baseline/` |

## Evaluation

### Baseline Questions

- Can a non-spatial RF exceed random-level ±1 accuracy quickly?
- Do feature importances cluster around specific timesteps/bands?

### Success Criteria

- **Confirm if**: ±1 accuracy is meaningfully above random baseline and feature
  importance output is stable across reruns.
- **Reject if**: ±1 accuracy is close to random and importances are noisy/uninformative.

## Limitations

- Pixel-wise RF ignores spatial context (no neighborhood information).
- Flattened features may over-emphasize noisy timesteps.
- cuML availability depends on compatible CUDA Linux environment.

