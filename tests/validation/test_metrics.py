import numpy as np
import pytest

from validation.metrics import (
    metrics_from_rank,
)
from validation.scoring import mask_history, rank_of_target, top_k_rows


def test_metrics_and_repeated_target_history_mask() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    masked = mask_history(scores, [0, 2], target_row=2)
    assert np.isneginf(masked[0])
    rank = rank_of_target(masked, 2)
    assert rank == 2
    assert metrics_from_rank(rank)["NDCG@4"] > 0


def test_ranker_rejects_non_finite_scores_instead_of_reporting_rank_one() -> None:
    with pytest.raises(ValueError, match="finite"):
        rank_of_target(np.array([0.5, np.nan, 0.1]), target_row=1)


def test_top_k_uses_the_same_index_tie_break_as_target_rank() -> None:
    scores = np.zeros(30)

    assert rank_of_target(scores, target_row=0) == 1
    assert top_k_rows(scores, 20) == list(range(20))
