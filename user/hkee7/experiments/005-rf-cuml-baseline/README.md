# 005-rf-cuml-baseline

Pixel-wise RandomForest baseline for AI4Agri using flattened `(T, C)` features.

## What this baseline answers quickly

- How strong a non-spatial model can be on this dataset
- Which timesteps/bands contribute most via feature importances

## Files

- `train.py`: train baseline and write artifacts
- `dataset.py`: flatten pixel features from chunked tensors
- `model.py`: cuML-first RF backend with sklearn fallback
- `metrics.py`: ±1 accuracy, exact accuracy, MAE
- `test.py`: smoke test (backend + metrics + fit/predict)

## Quick run

```bash
uv sync --package 005-rf-cuml-baseline
uv run --package 005-rf-cuml-baseline python user/hkee7/experiments/005-rf-cuml-baseline/test.py
uv run --package 005-rf-cuml-baseline python user/hkee7/experiments/005-rf-cuml-baseline/train.py --max-train-pixels 200000 --max-val-pixels 80000
```

Artifacts are written to `user/hkee7/experiments/005-rf-cuml-baseline/artifacts/`.

