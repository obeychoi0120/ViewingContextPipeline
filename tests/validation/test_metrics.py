import numpy as np
import pytest

from validation.metrics import (
    bonferroni_alpha,
    metrics_from_rank,
    paired_bootstrap_ci,
    paired_relative_bootstrap_ci,
)
from validation.recommendation import _metric
from validation.scoring import mask_history, rank_of_target, top_k_rows


def test_metrics_and_repeated_target_history_mask() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    masked = mask_history(scores, [0, 2], target_row=2)
    assert np.isneginf(masked[0])
    rank = rank_of_target(masked, 2)
    assert rank == 2
    assert metrics_from_rank(rank)["NDCG@4"] > 0


def test_recommendation_metric_passes_target_row_to_ranker() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    metrics = _metric(scores, history=[0, 2], target=2, cutoffs=[10])

    assert metrics["NDCG@10"] == metrics_from_rank(2, [10])["NDCG@10"]


def test_ranker_rejects_non_finite_scores_instead_of_reporting_rank_one() -> None:
    with pytest.raises(ValueError, match="finite"):
        rank_of_target(np.array([0.5, np.nan, 0.1]), target_row=1)


def test_top_k_uses_the_same_index_tie_break_as_target_rank() -> None:
    scores = np.zeros(30)

    assert rank_of_target(scores, target_row=0) == 1
    assert top_k_rows(scores, 20) == list(range(20))


def test_paired_bootstrap_positive_ci() -> None:
    result = paired_bootstrap_ci(np.ones(20), samples=200, seed=42, alpha=0.01)
    assert result["mean"] == result["ci_low"] == result["ci_high"] == 1.0
    assert result["alpha"] == 0.01
    assert result["confidence_level"] == 0.99


def test_bonferroni_alpha_and_invalid_alpha() -> None:
    assert bonferroni_alpha(0.05, 3) == pytest.approx(0.05 / 3)
    with pytest.raises(ValueError, match="between 0 and 1"):
        bonferroni_alpha(1.0, 3)
    with pytest.raises(ValueError, match="positive integer"):
        bonferroni_alpha(0.05, 0)


def test_relative_noninferiority_boundary() -> None:
    desc = np.full(20, 0.5)
    exactly_margin = paired_relative_bootstrap_ci(np.full(20, 0.475), desc, samples=200)
    above_margin = paired_relative_bootstrap_ci(
        np.full(20, 0.48),
        desc,
        samples=200,
        alpha=0.025,
        interval_type="one_sided_lower",
    )
    assert np.isclose(exactly_margin["ci_low"], -0.05)
    assert above_margin["ci_low"] > -0.05
    assert above_margin["alpha"] == 0.025
    assert above_margin["interval_type"] == "one_sided_lower"
    assert above_margin["ci_high"] is None


def test_one_sided_lower_uses_alpha_quantile_not_alpha_over_two() -> None:
    control = np.linspace(0.2, 1.0, 20)
    treatment = control + np.linspace(-0.05, 0.05, 20)

    two_sided = paired_relative_bootstrap_ci(
        treatment, control, samples=1_000, seed=42, alpha=0.05
    )
    one_sided = paired_relative_bootstrap_ci(
        treatment,
        control,
        samples=1_000,
        seed=42,
        alpha=0.05,
        interval_type="one_sided_lower",
    )

    assert one_sided["interval_type"] == "one_sided_lower"
    assert one_sided["ci_low"] > two_sided["ci_low"]
    assert one_sided["ci_high"] is None


def test_relative_bootstrap_resamples_sparse_positive_control() -> None:
    control = np.array([1.0] + [0.0] * 19)
    treatment = control.copy()

    result = paired_relative_bootstrap_ci(treatment, control, samples=500, seed=42)

    assert result["relative_delta"] == result["ci_low"] == result["ci_high"] == 0.0
    assert result["bootstrap_samples"] == 500
    assert result["bootstrap_draws"] > result["bootstrap_samples"]
    assert result["zero_control_resamples"] > 0
    assert result["control_nonzero_users"] == 1
    assert result["conditional_on_positive_control"] is True


def test_relative_bootstrap_rejects_zero_full_control_mean() -> None:
    with pytest.raises(ValueError, match="positive control mean"):
        paired_relative_bootstrap_ci(np.ones(20), np.zeros(20), samples=100)
