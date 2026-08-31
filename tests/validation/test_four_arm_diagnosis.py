from __future__ import annotations

from copy import deepcopy

import pytest

import validation.diagnosis_statistics as diagnosis_statistics
from validation.config import ValidationConfig
from validation.diagnosis import diagnose_recommendations
from validation.metrics import metrics_from_rank
from validation.recommendation import RECOMMENDATION_ARMS
from pipeline_runtime import read_jsonl, write_json, write_jsonl

from conftest import config_data


DECISION_CONFIG = {
    "min_scene_coverage": 0.8,
    "max_arm_coverage_gap": 0.1,
    "familywise_alpha": 0.05,
    "multiple_comparison_correction": "bonferroni",
}
ARM_RANKS = {
    "SASRec_ID": 4,
    "SASRec_GRAPH_QWEN": 2,
    "SASRec_GRAPH_GEMINI": 1,
    "SASRec_DESC": 3,
}


def _top_items(item_ids: list[str], target: str, rank: int) -> list[str]:
    ordered = [item for item in item_ids if item != target]
    ordered.insert(rank - 1, target)
    return ordered


def _write_success_scene_files(run_root, content_ids: list[str]) -> None:
    for content_id in content_ids:
        write_json(
            run_root
            / "data"
            / "cohort"
            / "source_assets"
            / content_id
            / "assets"
            / "timestamp_fixed_30s.json",
            [{"scene_start": 0, "scene_end": 30, "keyframe_timestamps": [5, 15, 25]}],
        )
        write_jsonl(
            run_root / "extraction/graph/qwen/scenes" / f"{content_id}.jsonl",
            [
                {
                    "scene_idx": 0,
                    "keyframes": [5, 15, 25],
                    "graph": {},
                    "parse_mode": "native",
                    "semantic_warnings": [],
                }
            ],
        )
        write_jsonl(
            run_root / "extraction/graph/gemini/scenes" / f"{content_id}.jsonl",
            [
                {
                    "scene_idx": 0,
                    "keyframes": [5, 15, 25],
                    "graph": {},
                    "parse_mode": "repaired" if content_id == "c1" else "native",
                    "semantic_warnings": ["dangling reference"] if content_id == "c1" else [],
                }
            ],
        )
        write_jsonl(
            run_root / "extraction/description/scenes" / f"{content_id}.jsonl",
            [
                {
                    "schema_version": "scene-description/v1",
                    "content_id": content_id,
                    "scene_idx": 0,
                    "keyframes": [5, 15, 25],
                    "description": "visible scene",
                }
            ],
        )


def _valid_run(tmp_path):
    config = ValidationConfig.model_validate(config_data(tmp_path, users=2))
    run_root = tmp_path / "run"
    cohort_dir = run_root / "data" / "cohort"
    item_ids = [f"i{index}" for index in range(1, 6)]
    content_ids = [f"c{index}" for index in range(1, 6)]
    item_content = dict(zip(item_ids, content_ids))
    write_jsonl(
        cohort_dir / "catalog.jsonl",
        [
            {"content_id": content_id, "item_id": item_id}
            for item_id, content_id in item_content.items()
        ],
    )
    sequences = [
        {
            "user_id": "u1",
            "train": ["i1", "i2", "i4"],
            "valid_target": "i3",
            "test_target": "i5",
        },
        {
            "user_id": "u2",
            "train": ["i1", "i2", "i3"],
            "valid_target": "i5",
            "test_target": "i4",
        },
    ]
    write_jsonl(cohort_dir / "sequences.jsonl", sequences)
    write_json(
        cohort_dir / "eligibility_summary.json",
        {
            "schema_version": "microlens-cohort-eligibility/v1",
            "eligible_items": len(item_ids),
            "eligible_users": len(sequences),
        },
    )
    _write_success_scene_files(run_root, content_ids)

    rows = []
    target_buckets = {"u1": "cold", "u2": "low"}
    for seed in config.model.seeds:
        for sequence in sequences:
            target = sequence["test_target"]
            for arm, branch in RECOMMENDATION_ARMS.items():
                rank = ARM_RANKS[arm]
                top_items = _top_items(item_ids, target, rank)
                rows.append(
                    {
                        "seed": seed,
                        "user_id": sequence["user_id"],
                        "arm": arm,
                        "branch": branch,
                        "candidate_count": len(item_ids),
                        "target_frequency_bucket": target_buckets[
                            sequence["user_id"]
                        ],
                        "rank": rank,
                        "target_item_id": target,
                        "target_content_id": item_content[target],
                        "top_item_ids": top_items,
                        "top_content_ids": [item_content[item] for item in top_items],
                        **metrics_from_rank(rank, config.evaluation.cutoffs),
                    }
                )
    recommendations_dir = run_root / "validation" / "recommendations"
    write_jsonl(recommendations_dir / "per_user_metrics.jsonl", rows)
    training_runs = []
    for seed in config.model.seeds:
        for arm in RECOMMENDATION_ARMS:
            checkpoint = (
                recommendations_dir
                / "checkpoints"
                / f"seed_{seed}"
                / arm.lower()
                / "sasrec.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            value = metrics_from_rank(ARM_RANKS[arm], config.evaluation.cutoffs)[
                "NDCG@10"
            ]
            training_runs.append(
                {
                    "schema_version": "sasrec-training-run/v1",
                    "run_id": "test",
                    "seed": seed,
                    "arm": arm,
                    "checkpoint": str(checkpoint),
                    "epochs": [{"epoch": 1, "loss": 1.0, "NDCG@10": value}],
                    "best_validation": {
                        "metric": "NDCG@10",
                        "value": value,
                        "epoch": 1,
                    },
                }
            )
    write_jsonl(recommendations_dir / "training_runs.jsonl", training_runs)
    runtime = {
        "run_id": "test",
        "run_root": str(run_root),
        "modality": "visual_only",
        "paths": {
            "recommendations_dir": str(recommendations_dir),
            "cohort_dir": str(cohort_dir),
        },
    }
    return config, runtime, rows, content_ids


def test_four_arm_runtime_decision_and_multiple_comparison_policy(tmp_path) -> None:
    assert RECOMMENDATION_ARMS == {
        "SASRec_ID": None,
        "SASRec_GRAPH_QWEN": "graph_qwen",
        "SASRec_GRAPH_GEMINI": "graph_gemini",
        "SASRec_DESC": "desc",
    }
    config, runtime, _, _ = _valid_run(tmp_path)

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert "report_ready" not in result
    assert result["schema_version"] == "diagnosis/v2"
    assert result["runtime_decision"] == {
        "status": "pass",
        "checks": {key: True for key in result["runtime_decision"]["checks"]},
        "errors": [],
    }
    assert result["statistical_analysis"] == {
        "status": "computed",
        "expected_comparison_count": 6,
        "computed_comparison_count": 6,
        "errors": [],
        "warnings": [],
    }
    grid = result["artifact_integrity"]["recommendation_grid"]
    assert grid["expected_cell_count"] == grid["observed_row_count"] == 24
    assert grid["duplicate_cell_count"] == grid["missing_cell_count"] == 0
    policy = result["multiple_comparison_policy"]
    assert policy["families"]["id_baseline_superiority"][
        "per_comparison_alpha"
    ] == pytest.approx(0.05 / 3)
    assert policy["families"]["graph_vs_description_non_inferiority"][
        "per_comparison_alpha"
    ] == pytest.approx(0.05 / 2)
    assert policy["families"]["graph_vs_description_non_inferiority"][
        "interval_type"
    ] == "one_sided_lower"
    assert policy["families"]["qwen_vs_gemini"]["role"] == "exploratory"
    assert set(result["comparisons"]) == {
        "SASRec_GRAPH_QWEN-SASRec_ID",
        "SASRec_GRAPH_GEMINI-SASRec_ID",
        "SASRec_DESC-SASRec_ID",
        "SASRec_GRAPH_QWEN-SASRec_DESC",
        "SASRec_GRAPH_GEMINI-SASRec_DESC",
        "SASRec_GRAPH_GEMINI-SASRec_GRAPH_QWEN",
    }
    assert result["comparisons"]["SASRec_GRAPH_QWEN-SASRec_ID"]["decision"] == "superior"
    assert result["comparisons"]["SASRec_GRAPH_QWEN-SASRec_DESC"][
        "decision"
    ] == "non_inferior"
    assert result["comparisons"]["SASRec_GRAPH_QWEN-SASRec_DESC"][
        "non_inferior"
    ] is True
    assert result["comparisons"]["SASRec_GRAPH_QWEN-SASRec_DESC"][
        "interval_type"
    ] == "one_sided_lower"
    assert result["comparisons"]["SASRec_GRAPH_QWEN-SASRec_DESC"]["ci_high"] is None
    coverage_diagnostic = result["diagnostics"]["SASRec_ID"]["top20_coverage"]
    assert coverage_diagnostic["by_seed"] == {"42": 1.0, "43": 1.0, "44": 1.0}
    assert coverage_diagnostic["mean"] == 1.0
    assert coverage_diagnostic["range"] == {"min": 1.0, "max": 1.0}
    coverage = result["scene_coverage"]
    assert coverage["common_scene_intersection_required"] is False
    assert coverage["arms"]["graph_qwen"]["parse_mode_counts"] == {"native": 5}
    assert coverage["arms"]["graph_gemini"]["parse_mode_counts"] == {
        "native": 4,
        "repaired": 1,
    }
    assert coverage["arms"]["graph_gemini"]["semantic_warning_scene_count"] == 1
    assert coverage["arms"]["graph_gemini"][
        "semantic_warning_rate_among_successes"
    ] == pytest.approx(0.2)


def test_duplicate_missing_and_invalid_metric_rows_return_fail_document(tmp_path) -> None:
    config, runtime, rows, _ = _valid_run(tmp_path)
    invalid = deepcopy(rows[:-1])
    duplicate = deepcopy(rows[0])
    duplicate["candidate_count"] = 999
    duplicate["target_item_id"] = "outside"
    duplicate["rank"] = 1
    duplicate["top_item_ids"][0] = "outside"
    invalid.append(duplicate)
    write_jsonl(
        tmp_path / "run/validation/recommendations/per_user_metrics.jsonl",
        invalid,
    )

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    checks = result["runtime_decision"]["checks"]
    assert checks["recommendation_grid_complete"] is False
    assert checks["recommendation_rows_valid"] is False
    assert result["statistical_analysis"]["status"] == "not_computed"
    grid = result["artifact_integrity"]["recommendation_grid"]
    assert grid["duplicate_cell_count"] == 1
    assert grid["missing_cell_count"] == 1
    assert grid["row_issue_counts"]["candidate_count_mismatch"] == 1
    assert grid["row_issue_counts"]["target_outside_catalog"] == 1
    codes = {error["code"] for error in result["runtime_decision"]["errors"]}
    assert {"recommendation_grid_incomplete", "invalid_recommendation_rows"} <= codes


def test_scene_failures_are_counted_and_coverage_thresholds_fail(tmp_path) -> None:
    config, runtime, _, content_ids = _valid_run(tmp_path)
    run_root = tmp_path / "run"
    for content_id in content_ids[2:]:
        write_jsonl(
            run_root / "extraction/graph/gemini/scenes" / f"{content_id}.jsonl",
            [],
        )
        write_jsonl(
            run_root / "extraction/graph/gemini/failures" / f"{content_id}.jsonl",
            [{"scene_idx": 0, "failure_kind": "generation"}],
        )

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    checks = result["runtime_decision"]["checks"]
    assert checks["scene_outcomes_complete"] is True
    assert checks["minimum_scene_coverage_met"] is False
    assert checks["arm_scene_coverage_gap_within_limit"] is False
    gemini = result["scene_coverage"]["arms"]["graph_gemini"]
    assert gemini["success_coverage"] == pytest.approx(0.4)
    assert gemini["failure_rate"] == pytest.approx(0.6)
    assert gemini["outcome_coverage"] == pytest.approx(1.0)
    assert result["scene_coverage"]["common_success_scene_count"] == 2


def test_missing_metrics_returns_structured_fail_document(tmp_path) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    metrics_path = tmp_path / "run/validation/recommendations/per_user_metrics.jsonl"
    metrics_path.unlink()

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["recommendation_grid_complete"] is False
    assert result["comparisons"] == {}
    assert any(
        error["code"] == "missing_artifact"
        and "per-user recommendation metrics" in error["message"]
        for error in result["runtime_decision"]["errors"]
    )


def test_diagnosis_reloads_metrics_on_every_call(tmp_path) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    first = diagnose_recommendations(config, runtime, DECISION_CONFIG)
    assert first["runtime_decision"]["status"] == "pass"
    path = tmp_path / "run/validation/recommendations/per_user_metrics.jsonl"
    rows = read_jsonl(path)
    write_jsonl(path, rows[:-1])

    second = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert second["runtime_decision"]["status"] == "fail"
    assert second["artifact_integrity"]["recommendation_grid"]["missing_cell_count"] == 1


@pytest.mark.parametrize("missing_file", [False, True])
def test_missing_or_partial_training_history_fails_runtime(
    tmp_path, missing_file
) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    path = tmp_path / "run/validation/recommendations/training_runs.jsonl"
    if missing_file:
        path.unlink()
    else:
        write_jsonl(path, read_jsonl(path)[:-1])

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["training_runs_complete"] is False
    training = result["artifact_integrity"]["training_runs"]
    assert training["expected_cell_count"] == 12
    assert training["observed_row_count"] == (0 if missing_file else 11)
    assert training["missing_cell_count"] == (12 if missing_file else 1)


def test_empty_checkpoint_fails_minimum_checkpoint_contract(tmp_path) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    checkpoint = (
        tmp_path
        / "run/validation/recommendations/checkpoints/seed_42/sasrec_id/sasrec.pt"
    )
    checkpoint.write_bytes(b"")

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["checkpoints_complete"] is False
    assert result["artifact_integrity"]["checkpoints"]["empty_count"] == 1


def test_training_history_validates_best_value_and_canonical_checkpoint(tmp_path) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    path = tmp_path / "run/validation/recommendations/training_runs.jsonl"
    rows = read_jsonl(path)
    rows[0]["best_validation"]["value"] = -1.0
    rows[1]["checkpoint"] = str(tmp_path / "unrelated.pt")
    write_jsonl(path, rows)

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    issues = result["artifact_integrity"]["training_runs"]["row_issue_counts"]
    assert issues["invalid_best_validation"] == 1
    assert issues["checkpoint_path_mismatch"] == 1


def test_invalid_success_scene_schemas_fail_runtime_and_success_coverage(
    tmp_path,
) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    run_root = tmp_path / "run"
    write_jsonl(
        run_root / "extraction/graph/qwen/scenes/c1.jsonl",
        [
            {
                "scene_idx": 0,
                "keyframes": [5, 15, 25],
                "parse_mode": "native",
                "semantic_warnings": [],
            }
        ],
    )
    write_jsonl(
        run_root / "extraction/description/scenes/c2.jsonl",
        [
            {
                "schema_version": "scene-description/v1",
                "content_id": "wrong",
                "scene_idx": 0,
                "keyframes": [5, 15, 25],
                "description": "visible scene",
            }
        ],
    )

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["scene_outcomes_complete"] is False
    assert result["scene_coverage"]["arms"]["graph_qwen"]["success_scene_count"] == 4
    assert result["scene_coverage"]["arms"]["graph_qwen"]["issue_counts"][
        "invalid_graph"
    ] == 1
    assert result["scene_coverage"]["arms"]["desc"]["issue_counts"][
        "description_content_id_mismatch"
    ] == 1


def test_malformed_runtime_returns_structured_fail_document(tmp_path) -> None:
    config, _, _, _ = _valid_run(tmp_path)

    result = diagnose_recommendations(
        config,
        {"run_id": "test", "paths": []},
        DECISION_CONFIG,
    )

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["runtime_paths_valid"] is False
    assert any(
        error["code"] in {"invalid_runtime_paths", "missing_runtime_paths"}
        for error in result["runtime_decision"]["errors"]
    )


def test_shallow_cohort_path_without_run_root_returns_fail_document(tmp_path) -> None:
    config, _, _, _ = _valid_run(tmp_path)

    result = diagnose_recommendations(
        config,
        {
            "run_id": "test",
            "paths": {
                "recommendations_dir": "recommendations",
                "cohort_dir": "cohort",
            },
        },
        DECISION_CONFIG,
    )

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["runtime_paths_valid"] is False


def test_unhashable_sequence_and_metric_targets_return_structured_fail(
    tmp_path,
) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    sequence_path = tmp_path / "run/data/cohort/sequences.jsonl"
    sequences = read_jsonl(sequence_path)
    sequences[0]["train"] = [{"item_id": "i1"}]
    sequences[0]["valid_target"] = ["i3"]
    write_jsonl(sequence_path, sequences)
    metrics_path = tmp_path / "run/validation/recommendations/per_user_metrics.jsonl"
    metrics = read_jsonl(metrics_path)
    metrics[0]["target_item_id"] = ["i5"]
    write_jsonl(metrics_path, metrics)

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    assert result["runtime_decision"]["checks"]["sequences_valid"] is False
    assert result["runtime_decision"]["checks"]["recommendation_rows_valid"] is False
    assert any(
        error["code"] == "invalid_sequences"
        for error in result["runtime_decision"]["errors"]
    )


def test_train_length_and_recomputed_frequency_bucket_are_enforced(tmp_path) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    sequence_path = tmp_path / "run/data/cohort/sequences.jsonl"
    sequences = read_jsonl(sequence_path)
    sequences[0]["train"] = ["i1", "i2"]
    write_jsonl(sequence_path, sequences)
    metrics_path = tmp_path / "run/validation/recommendations/per_user_metrics.jsonl"
    metrics = read_jsonl(metrics_path)
    metrics[0]["target_frequency_bucket"] = "warm"
    write_jsonl(metrics_path, metrics)

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "fail"
    sequence_error = next(
        error
        for error in result["runtime_decision"]["errors"]
        if error["code"] == "invalid_sequences"
    )
    assert sequence_error["details"]["issue_counts"]["train_too_short"] == 1
    recommendation = result["artifact_integrity"]["recommendation_grid"]
    assert recommendation["row_issue_counts"][
        "target_frequency_bucket_mismatch"
    ] >= 1


def test_all_statistical_failures_do_not_fail_runtime_artifact_contract(
    tmp_path, monkeypatch
) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)

    def reject_zero_control_mean(*args, **kwargs):
        raise ValueError("control mean must be non-zero for relative bootstrap")

    monkeypatch.setattr(
        diagnosis_statistics,
        "paired_relative_bootstrap_ci",
        reject_zero_control_mean,
    )
    monkeypatch.setattr(
        diagnosis_statistics,
        "paired_bootstrap_ci",
        reject_zero_control_mean,
    )

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "pass"
    assert all(result["runtime_decision"]["checks"].values())
    assert result["runtime_decision"]["errors"] == []
    assert result["statistical_analysis"]["status"] == "not_computed"
    assert result["statistical_analysis"]["computed_comparison_count"] == 0
    assert len(result["statistical_analysis"]["errors"]) == 6
    assert {
        error["code"] for error in result["statistical_analysis"]["errors"]
    } == {"comparison_computation_failed"}
    assert result["comparisons"] == {}


def test_relative_failures_preserve_id_family_results(tmp_path, monkeypatch) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)

    def reject_relative_comparison(*args, **kwargs):
        raise ValueError("relative control is not evaluable")

    monkeypatch.setattr(
        diagnosis_statistics,
        "paired_relative_bootstrap_ci",
        reject_relative_comparison,
    )

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "pass"
    assert result["statistical_analysis"]["status"] == "computed_with_errors"
    assert result["statistical_analysis"]["computed_comparison_count"] == 3
    assert len(result["statistical_analysis"]["errors"]) == 3
    assert set(result["comparisons"]) == {
        "SASRec_GRAPH_QWEN-SASRec_ID",
        "SASRec_GRAPH_GEMINI-SASRec_ID",
        "SASRec_DESC-SASRec_ID",
    }


def test_sparse_control_suppresses_non_inferiority_claims(tmp_path, monkeypatch) -> None:
    config, runtime, _, _ = _valid_run(tmp_path)
    original = diagnosis_statistics.paired_relative_bootstrap_ci

    def mark_conditional(*args, **kwargs):
        result = original(*args, **kwargs)
        result["conditional_on_positive_control"] = True
        result["zero_control_resamples"] = 1
        return result

    monkeypatch.setattr(
        diagnosis_statistics,
        "paired_relative_bootstrap_ci",
        mark_conditional,
    )

    result = diagnose_recommendations(config, runtime, DECISION_CONFIG)

    assert result["runtime_decision"]["status"] == "pass"
    assert result["statistical_analysis"]["status"] == "computed_with_warnings"
    assert len(result["statistical_analysis"]["warnings"]) == 2
    for comparison in (
        "SASRec_GRAPH_QWEN-SASRec_DESC",
        "SASRec_GRAPH_GEMINI-SASRec_DESC",
    ):
        assert result["comparisons"][comparison]["non_inferior"] is None
        assert (
            result["comparisons"][comparison]["decision"]
            == "not_evaluable_sparse_control"
        )
