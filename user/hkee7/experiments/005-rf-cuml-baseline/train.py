from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from config import Config
from dataset import feature_names, sample_pixel_table
from metrics import exact_accuracy, mae, pm1_accuracy
from model import build_rf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF pixel baseline")
    parser.add_argument("--max-train-pixels", type=int, default=300000)
    parser.add_argument("--max-val-pixels", type=int, default=120000)
    parser.add_argument("--chunks-per-split", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def _to_numpy(pred: object) -> np.ndarray:
    if hasattr(pred, "to_numpy"):
        return pred.to_numpy()
    return np.asarray(pred)


def _save_feature_importance(
    out_path: Path,
    names: list[str],
    importances: np.ndarray,
    top_k: int = 30,
) -> None:
    order = np.argsort(importances)[::-1]
    rows = [
        {"rank": i + 1, "feature": names[idx], "importance": float(importances[idx])}
        for i, idx in enumerate(order[:top_k])
    ]
    out_path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    args = parse_args()
    cfg = Config(
        max_train_pixels=args.max_train_pixels,
        max_val_pixels=args.max_val_pixels,
        chunks_per_split=args.chunks_per_split,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        random_state=args.seed,
        use_cuml=not args.cpu_only,
    )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pixel tables...")
    x_train, y_train = sample_pixel_table(
        "train",
        data_root=cfg.data_root,
        max_pixels=cfg.max_train_pixels,
        random_state=cfg.random_state,
        max_chunks=cfg.chunks_per_split,
    )
    x_val, y_val = sample_pixel_table(
        "val",
        data_root=cfg.data_root,
        max_pixels=cfg.max_val_pixels,
        random_state=cfg.random_state + 1,
        max_chunks=cfg.chunks_per_split,
    )

    backend = build_rf(
        use_cuml=cfg.use_cuml,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
    )
    print(f"Training backend: {backend.name}")
    print(f"Train samples: {x_train.shape[0]:,}, features: {x_train.shape[1]}")

    start = perf_counter()
    backend.model.fit(x_train, y_train)
    fit_seconds = perf_counter() - start

    pred_val = _to_numpy(backend.model.predict(x_val)).astype(np.int32)
    y_val = y_val.astype(np.int32)

    metrics = {
        "backend": backend.name,
        "fit_seconds": fit_seconds,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "num_features": int(x_train.shape[1]),
        "pm1_accuracy": pm1_accuracy(pred_val, y_val),
        "exact_accuracy": exact_accuracy(pred_val, y_val),
        "mae": mae(pred_val, y_val),
        "oob_score": float(getattr(backend.model, "oob_score_", np.nan)),
    }

    model_path = cfg.artifacts_dir / "rf_model.joblib"
    metrics_path = cfg.artifacts_dir / "metrics.json"
    importances_path = cfg.artifacts_dir / "feature_importance_top30.json"

    joblib.dump(backend.model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if hasattr(backend.model, "feature_importances_"):
        names = feature_names()
        importances = _to_numpy(backend.model.feature_importances_)
        _save_feature_importance(importances_path, names, importances)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

