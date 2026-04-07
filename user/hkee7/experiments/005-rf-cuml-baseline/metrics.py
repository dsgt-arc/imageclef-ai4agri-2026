from __future__ import annotations

import numpy as np


def pm1_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target) <= 1)) if target.size else 0.0


def exact_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred == target)) if target.size else 0.0


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target))) if target.size else 0.0

