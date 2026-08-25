from __future__ import annotations

import numpy as np


def mask_history(scores: np.ndarray, history_rows: list[int], target_row: int) -> np.ndarray:
    masked = scores.copy()
    for row in set(history_rows):
        if row != target_row:
            masked[row] = -np.inf
    return masked


def rank_of_target(scores: np.ndarray, target_row: int) -> int:
    target = scores[target_row]
    return int(np.count_nonzero(scores > target) + np.count_nonzero((scores == target) & (np.arange(len(scores)) < target_row)) + 1)
