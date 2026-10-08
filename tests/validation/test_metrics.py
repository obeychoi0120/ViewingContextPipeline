import numpy as np
import pytest

from validation.metrics import (
    metrics_from_rank,
)
from validation.scoring import mask_history, rank_of_target


def test_metrics_and_repeated_target_history_mask() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    masked = mask_history(scores, [0, 2])
    assert np.isneginf(masked[0])
    rank = rank_of_target(masked, 2)
    assert np.isneginf(masked[2])
    assert rank == 4
    assert metrics_from_rank(rank)["NDCG@4"] > 0


def test_ranker_rejects_non_finite_scores_instead_of_reporting_rank_one() -> None:
    with pytest.raises(ValueError, match="finite"):
        rank_of_target(np.array([0.5, np.nan, 0.1]), target_row=1)
