from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from config import Config
from dataset import REFLECTANCE_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RF submission masks")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    return parser.parse_args()


def _to_numpy(pred: object) -> np.ndarray:
    if hasattr(pred, "to_numpy"):
        return pred.to_numpy()
    return np.asarray(pred)


def _chunk_paths(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*.pt"))


def _flatten_pixels(data: torch.Tensor) -> np.ndarray:
    # (N, T, C, H, W) -> (N*H*W, T*C)
    arr = data.float().numpy() / REFLECTANCE_SCALE
    n, t, c, h, w = arr.shape
    return arr.transpose(0, 3, 4, 1, 2).reshape(n * h * w, t * c)


def _predict_patch_batch(model: object, data_batch: torch.Tensor) -> np.ndarray:
    x = _flatten_pixels(data_batch)
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and x.shape[1] != int(expected_features):
        raise ValueError(
            f"Feature mismatch: got {x.shape[1]}, expected {expected_features}"
        )

    pred = _to_numpy(model.predict(x)).astype(np.int32)
    n, _t, _c, h, w = data_batch.shape
    return pred.reshape(n, h, w)


def _to_submission_space(pred: np.ndarray) -> np.ndarray:
    # Match other experiments: training label space [1,5] -> submission [0,4]
    # with edge-safe clamp used across this repo.
    return pred.astype(np.int32).clip(2, 4) - 1


def _save_pngs(
    pred_maps: np.ndarray,
    patch_ids: list[str],
    output_dir: Path,
    *,
    max_patches: int | None,
    offset: int,
) -> int:
    saved = 0
    for i, (pred_map, patch_id) in enumerate(zip(pred_maps, patch_ids, strict=True)):
        if max_patches is not None and offset + i >= max_patches:
            break
        img = Image.fromarray(pred_map.astype(np.uint8), mode="L")
        img.save(output_dir / f"{patch_id}.png")
        saved += 1
    return saved


def _zip_pngs(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for png in sorted(output_dir.glob("*.png")):
            zf.write(png, arcname=png.name)
    return zip_path


def main() -> None:
    args = parse_args()
    cfg = Config()

    model_path = args.model_path or (cfg.artifacts_dir / "rf_model.joblib")
    output_dir = args.output_dir or (cfg.artifacts_dir / f"submission_{args.split}")
    split_dir = cfg.data_root / args.split

    if args.patch_batch_size <= 0:
        raise ValueError("--patch-batch-size must be a positive integer")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    chunk_paths = _chunk_paths(split_dir)
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {split_dir}")
    if args.max_chunks is not None:
        chunk_paths = chunk_paths[: args.max_chunks]

    model = joblib.load(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for chunk_path in chunk_paths:
        payload = torch.load(chunk_path, weights_only=True)
        data = payload["data"]
        patch_ids = [str(pid) for pid in payload.get("patch_ids", [])]

        if not patch_ids:
            raise KeyError(f"Missing patch_ids in {chunk_path}")
        if data.shape[0] != len(patch_ids):
            raise ValueError(
                f"Mismatch in {chunk_path}: data has {data.shape[0]} patches but patch_ids has {len(patch_ids)}"
            )

        batch_saved = 0
        for start in range(0, data.shape[0], args.patch_batch_size):
            end = min(start + args.patch_batch_size, data.shape[0])
            pred = _predict_patch_batch(model, data[start:end])
            submission_pred = _to_submission_space(pred)

            saved = _save_pngs(
                submission_pred,
                patch_ids[start:end],
                output_dir,
                max_patches=args.max_patches,
                offset=total_saved,
            )
            total_saved += saved
            batch_saved += saved

            if args.max_patches is not None and total_saved >= args.max_patches:
                break

        print(f"Processed {chunk_path.name}: saved {batch_saved} masks")

        if args.max_patches is not None and total_saved >= args.max_patches:
            break

    zip_path = _zip_pngs(output_dir)
    print(f"Saved {total_saved} masks to {output_dir}")
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()

