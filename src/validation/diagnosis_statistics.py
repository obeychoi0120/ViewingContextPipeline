from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

import numpy as np

from .config import ValidationConfig
from .metrics import (
    bonferroni_alpha,
    paired_bootstrap_ci,
    paired_relative_bootstrap_ci,
)
from .recommendation_contracts import RECOMMENDATION_ARMS


def _error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    **details: Any,
) -> None:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    errors.append(error)


def multiple_comparison_policy(
    settings: dict[str, Any],
    valid: bool,
    primary_metric: str,
) -> dict[str, Any]:
    alpha = settings.get("familywise_alpha")
    id_alpha = bonferroni_alpha(alpha, 3) if valid else None
    ni_alpha = bonferroni_alpha(alpha, 2) if valid else None
    return {
        "primary_metric": primary_metric,
        "familywise_alpha": alpha,
        "correction": settings.get("multiple_comparison_correction"),
        "families": {
            "id_baseline_superiority": {
                "role": "confirmatory",
                "interval_type": "two_sided",
                "comparison_count": 3,
                "comparisons": [
                    "SASRec_GRAPH_QWEN-SASRec_ID",
                    "SASRec_GRAPH_GEMINI-SASRec_ID",
                    "SASRec_DESC-SASRec_ID",
                ],
                "per_comparison_alpha": id_alpha,
                "decision_rule": "ci_low > 0",
            },
            "graph_vs_description_non_inferiority": {
                "role": "confirmatory",
                "interval_type": "one_sided_lower",
                "comparison_count": 2,
                "comparisons": [
                    "SASRec_GRAPH_QWEN-SASRec_DESC",
                    "SASRec_GRAPH_GEMINI-SASRec_DESC",
                ],
                "per_comparison_alpha": ni_alpha,
                "decision_rule": (
                    "one-sided lower relative bound > -non_inferiority_margin"
                ),
            },
            "qwen_vs_gemini": {
                "role": "exploratory",
                "interval_type": "two_sided",
                "comparison_count": 1,
                "comparisons": ["SASRec_GRAPH_GEMINI-SASRec_GRAPH_QWEN"],
                "per_comparison_alpha": alpha if valid else None,
                "correction": "none",
                "decision_rule": "two-sided relative CI excludes 0",
            },
        },
    }


def _mean_by_user(
    by_key: dict[tuple[int, str, str], dict[str, Any]],
    users: list[str],
    seeds: list[int],
    arm: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            np.mean([by_key[(seed, user, arm)][metric] for seed in seeds])
            for user in users
        ],
        dtype=np.float64,
    )


def _seed_distribution(values: dict[int, float]) -> dict[str, Any]:
    observed = list(values.values())
    return {
        "by_seed": {str(seed): value for seed, value in values.items()},
        "mean": float(np.mean(observed)),
        "range": {
            "min": float(min(observed)),
            "max": float(max(observed)),
        },
    }


def statistics(
    by_key: dict[tuple[int, str, str], dict[str, Any]],
    *,
    users: list[str],
    seeds: list[int],
    config: ValidationConfig,
    policy: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    arms = list(RECOMMENDATION_ARMS)
    metric_names = [
        f"{name}@{cutoff}"
        for cutoff in config.evaluation.cutoffs
        for name in ("HR", "NDCG")
    ]
    canonical_rows = [
        by_key[(seed, user, arm)] for seed, user, arm in product(seeds, users, arms)
    ]
    summary = {
        arm: {
            metric: float(
                np.mean([row[metric] for row in canonical_rows if row["arm"] == arm])
            )
            for metric in metric_names
        }
        for arm in arms
    }
    diagnostics: dict[str, Any] = {}
    catalog_size = int(canonical_rows[0]["candidate_count"])
    for arm in arms:
        arm_rows = [row for row in canonical_rows if row["arm"] == arm]
        coverage_by_seed: dict[int, float] = {}
        concentration_by_seed: dict[int, float] = {}
        for seed in seeds:
            seed_rows = [row for row in arm_rows if row["seed"] == seed]
            recommended = [item for row in seed_rows for item in row["top_item_ids"]]
            top_one = Counter(row["top_item_ids"][0] for row in seed_rows)
            coverage_by_seed[seed] = len(set(recommended)) / catalog_size
            concentration_by_seed[seed] = max(top_one.values()) / len(seed_rows)
        buckets = sorted({row["target_frequency_bucket"] for row in arm_rows})
        diagnostics[arm] = {
            "top20_coverage": _seed_distribution(coverage_by_seed),
            "top1_concentration": _seed_distribution(concentration_by_seed),
            "frequency_bucket": {
                bucket: {
                    metric: float(
                        np.mean(
                            [
                                row[metric]
                                for row in arm_rows
                                if row["target_frequency_bucket"] == bucket
                            ]
                        )
                    )
                    for metric in metric_names
                }
                for bucket in buckets
            },
        }

    primary_metric = policy["primary_metric"]
    family = policy["families"]
    comparisons: dict[str, Any] = {}
    comparison_errors: list[dict[str, Any]] = []
    comparison_warnings: list[dict[str, Any]] = []
    baseline = _mean_by_user(by_key, users, seeds, "SASRec_ID", primary_metric)
    id_alpha = float(family["id_baseline_superiority"]["per_comparison_alpha"])
    for arm in ("SASRec_GRAPH_QWEN", "SASRec_GRAPH_GEMINI", "SASRec_DESC"):
        key = f"{arm}-SASRec_ID"
        try:
            treatment = _mean_by_user(by_key, users, seeds, arm, primary_metric)
            result = paired_bootstrap_ci(
                treatment - baseline,
                samples=config.evaluation.bootstrap_samples,
                alpha=id_alpha,
            )
        except (KeyError, TypeError, ValueError) as exc:
            _error(
                comparison_errors,
                "comparison_computation_failed",
                "failed to compute one declared comparison",
                family="id_baseline_superiority",
                comparison=key,
                error=str(exc),
            )
            continue
        result.update(
            {
                "family": "id_baseline_superiority",
                "decision": (
                    "superior"
                    if result["ci_low"] > 0
                    else "inferior"
                    if result["ci_high"] < 0
                    else "not_significant"
                ),
            }
        )
        comparisons[key] = result

    ni_alpha = float(
        family["graph_vs_description_non_inferiority"]["per_comparison_alpha"]
    )
    desc_values = _mean_by_user(by_key, users, seeds, "SASRec_DESC", primary_metric)
    for graph in ("SASRec_GRAPH_QWEN", "SASRec_GRAPH_GEMINI"):
        key = f"{graph}-SASRec_DESC"
        try:
            result = paired_relative_bootstrap_ci(
                _mean_by_user(by_key, users, seeds, graph, primary_metric),
                desc_values,
                samples=config.evaluation.bootstrap_samples,
                alpha=ni_alpha,
                interval_type="one_sided_lower",
            )
        except (KeyError, TypeError, ValueError) as exc:
            _error(
                comparison_errors,
                "comparison_computation_failed",
                "failed to compute one declared comparison",
                family="graph_vs_description_non_inferiority",
                comparison=key,
                error=str(exc),
            )
            continue
        margin = float(config.evaluation.non_inferiority_margin)
        sparse_control = result["conditional_on_positive_control"] is True
        if sparse_control:
            non_inferior: bool | None = None
            decision = "not_evaluable_sparse_control"
            _error(
                comparison_warnings,
                "non_inferiority_not_evaluable_sparse_control",
                "non-inferiority cannot be concluded from a conditional bootstrap",
                family="graph_vs_description_non_inferiority",
                comparison=key,
                zero_control_resamples=result["zero_control_resamples"],
                bootstrap_draws=result["bootstrap_draws"],
            )
        else:
            non_inferior = result["ci_low"] > -margin
            decision = (
                "non_inferior"
                if non_inferior
                else "non_inferiority_not_demonstrated"
            )
        result.update(
            {
                "family": "graph_vs_description_non_inferiority",
                "non_inferiority_margin": -margin,
                "non_inferior": non_inferior,
                "decision": decision,
            }
        )
        comparisons[key] = result

    exploratory_alpha = float(family["qwen_vs_gemini"]["per_comparison_alpha"])
    exploratory_key = "SASRec_GRAPH_GEMINI-SASRec_GRAPH_QWEN"
    try:
        exploratory = paired_relative_bootstrap_ci(
            _mean_by_user(by_key, users, seeds, "SASRec_GRAPH_GEMINI", primary_metric),
            _mean_by_user(by_key, users, seeds, "SASRec_GRAPH_QWEN", primary_metric),
            samples=config.evaluation.bootstrap_samples,
            alpha=exploratory_alpha,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _error(
            comparison_errors,
            "comparison_computation_failed",
            "failed to compute one declared comparison",
            family="qwen_vs_gemini",
            comparison=exploratory_key,
            error=str(exc),
        )
    else:
        exploratory.update(
            {
                "family": "qwen_vs_gemini",
                "exploratory": True,
                "decision": (
                    "gemini_higher"
                    if exploratory["ci_low"] > 0
                    else "qwen_higher"
                    if exploratory["ci_high"] < 0
                    else "inconclusive"
                ),
            }
        )
        comparisons[exploratory_key] = exploratory
    return (
        summary,
        diagnostics,
        comparisons,
        comparison_errors,
        comparison_warnings,
    )
