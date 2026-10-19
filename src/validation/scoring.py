from __future__ import annotations

import numpy as np
from validation.model import require_torch, torch


def mask_history(scores: np.ndarray, history_rows: list[int]) -> np.ndarray:
    if not np.isfinite(scores).all():
        raise ValueError("masking requires finite catalog scores")
    masked = scores.copy()
    for row in set(history_rows):
        masked[row] = -np.inf
    return masked


def rank_of_target(scores: np.ndarray, target_row: int) -> int:
    if scores.ndim != 1 or not 0 <= target_row < len(scores):
        raise ValueError("ranking requires a one-dimensional score vector and valid target row")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("ranking requires finite or history-masked scores")
    require_torch()
    order = torch.argsort(torch.as_tensor(scores), descending=True)
    return int((order == target_row).long().argmax()) + 1
