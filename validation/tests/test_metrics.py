import numpy as np

from vc_validation.metrics import metrics_from_rank, paired_bootstrap_ci, paired_relative_bootstrap_ci
from vc_validation.scoring import mask_history, rank_of_target


def test_metrics_and_repeated_target_history_mask() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    masked = mask_history(scores, [0, 2], target_row=2)
    assert np.isneginf(masked[0])
    rank = rank_of_target(masked, 2)
    assert rank == 2
    assert metrics_from_rank(rank)["NDCG@4"] > 0


def test_paired_bootstrap_positive_ci() -> None:
    result = paired_bootstrap_ci(np.ones(20), samples=200, seed=42)
    assert result["mean"] == result["ci_low"] == result["ci_high"] == 1.0


def test_relative_noninferiority_boundary() -> None:
    desc = np.full(20, 0.5)
    exactly_margin = paired_relative_bootstrap_ci(np.full(20, 0.475), desc, samples=200)
    above_margin = paired_relative_bootstrap_ci(np.full(20, 0.48), desc, samples=200)
    assert np.isclose(exactly_margin["ci_low"], -0.05)
    assert above_margin["ci_low"] > -0.05
