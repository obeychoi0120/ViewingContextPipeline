from __future__ import annotations

import numpy as np


def mask_history(scores: np.ndarray, history_rows: list[int], target_row: int) -> np.ndarray:
    masked = scores.copy()
    for row in set(history_rows):
        if row != target_row:
            masked[row] = -np.inf
    return masked


def rank_of_target(scores: np.ndarray, target_row: int) -> int:
    if scores.ndim != 1 or not 0 <= target_row < len(scores):
        raise ValueError("ranking requires a one-dimensional score vector and valid target row")
    if not np.isfinite(scores[target_row]) or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("ranking requires finite candidate and target scores")
    target = scores[target_row]
    return int(np.count_nonzero(scores > target) + np.count_nonzero((scores == target) & (np.arange(len(scores)) < target_row)) + 1)


def top_k_rows(scores: np.ndarray, k: int) -> list[int]:
    """Return score-descending rows with the same row-index tie-break as ranking."""
    if scores.ndim != 1 or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("top-k ranking requires a one-dimensional score vector")
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k <= len(scores):
        raise ValueError("top-k ranking requires 0 < k <= candidate count")
    rows = np.arange(len(scores))
    return np.lexsort((rows, -scores))[:k].tolist()
