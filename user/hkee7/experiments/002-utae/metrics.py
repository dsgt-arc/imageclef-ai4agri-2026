"""
Evaluation metrics for the AI4Agri crop-suitability task.

Primary metric: ±1 accuracy (prediction correct if |pred − target| ≤ 1).
"""

from __future__ import annotations

import torch


def pm1_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 0,
) -> float:
    """
    ±1 accuracy: fraction of labeled pixels where |pred − target| ≤ 1.

    Args:
        pred: (B, H, W) integer predictions in [1, 5].
        target: (B, H, W) ground-truth labels (0 = unlabelled, 1–5 = classes).
        ignore_index: Label value to ignore (default 0 = unlabelled).

    Returns:
        Scalar accuracy in [0, 1].
    """
    mask = target != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (pred[mask] - target[mask]).abs() <= 1
    return correct.float().mean().item()


def exact_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 0,
) -> float:
    """Standard (exact-match) accuracy on labelled pixels."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return 0.0
    return (pred[mask] == target[mask]).float().mean().item()


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 0,
) -> float:
    """Mean absolute error on labelled pixels."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return 0.0
    return (pred[mask].float() - target[mask].float()).abs().mean().item()
