from __future__ import annotations

from pathlib import Path
from typing import Any


from .config import ValidationConfig
from .cohort_selection import (
    CATALOG_SCOPE,
)
from .diagnosis_statistics import multiple_comparison_policy, statistics
from .recommendation_contracts import (
    RECOMMENDATION_ARMS,
    target_scope,
)


from .diagnosis_support import (
    _error,
    _finite_number,
    _read_json,
    _read_jsonl,
)
from .diagnosis_artifacts import (
    _catalog_contract,
    _cohort_plan_contract,
    _eligibility_contract,
    _metadata_title_contract,
    _representation_contract,
    _sequence_contract,
)
from .diagnosis_recommendations import (
    _checkpoint_contract,
    _recommendation_contract,
    _training_run_contract,
)
from .diagnosis_scenes import (
    _scene_coverage,
)


DECISION_CONFIG_KEYS = {
    "min_scene_coverage",
    "max_arm_coverage_gap",
    "familywise_alpha",
    "multiple_comparison_correction",
}


def _validate_decision_config(
    value: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, dict):
        _error(errors, "invalid_decision_config", "decision_config must be an object")
        return {}, False
    unknown = sorted(set(value) - DECISION_CONFIG_KEYS)
    missing = sorted(DECISION_CONFIG_KEYS - set(value))
    if unknown or missing:
        _error(
            errors,
            "invalid_decision_config",
            "decision_config keys do not match the runtime contract",
            missing=missing,
            unknown=unknown,
        )
        return dict(value), False

    valid = True
    settings = dict(value)
    for key in ("min_scene_coverage", "max_arm_coverage_gap"):
        raw = settings[key]
        if not _finite_number(raw) or not 0 <= float(raw) <= 1:
            _error(
                errors,
                "invalid_decision_config",
                f"{key} must be a finite number from 0 to 1",
                value=raw,
            )
            valid = False
        else:
            settings[key] = float(raw)
    alpha = settings["familywise_alpha"]
    if not _finite_number(alpha) or not 0 < float(alpha) < 1:
        _error(
            errors,
            "invalid_decision_config",
            "familywise_alpha must be a finite number between 0 and 1",
            value=alpha,
        )
        valid = False
    else:
        settings["familywise_alpha"] = float(alpha)
    if settings["multiple_comparison_correction"] != "bonferroni":
        _error(
            errors,
            "invalid_decision_config",
            "multiple_comparison_correction must be 'bonferroni'",
            value=settings["multiple_comparison_correction"],
        )
        valid = False
    return settings, valid


def _runtime_locations(
    runtime: Any,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path, Path, Path, bool]:
    runtime_object = runtime if isinstance(runtime, dict) else {}
    if not isinstance(runtime, dict):
        _error(errors, "invalid_runtime", "runtime must be an object")
    paths = runtime_object.get("paths")
    if not isinstance(paths, dict):
        _error(errors, "invalid_runtime_paths", "runtime.paths must be an object")
        paths = {}

    invalid: list[str] = []
    values: dict[str, Path] = {}
    placeholder_root = Path.cwd() / ".invalid-diagnosis-runtime"
    for key in ("recommendations_dir", "cohort_dir"):
        value = paths.get(key)
        if isinstance(value, Path) or (isinstance(value, str) and value.strip()):
            values[key] = Path(value)
        else:
            invalid.append(f"paths.{key}")
            values[key] = placeholder_root / key

    run_root_value = runtime_object.get("run_root")
    if isinstance(run_root_value, Path) or (
        isinstance(run_root_value, str) and run_root_value.strip()
    ):
        run_root = Path(run_root_value)
    else:
        invalid.append("run_root")
        cohort_valid = "paths.cohort_dir" not in invalid
        cohort_parents = values["cohort_dir"].parents
        run_root = (
            cohort_parents[1]
            if cohort_valid and len(cohort_parents) > 1
            else placeholder_root / "run_root"
        )
    if invalid:
        _error(
            errors,
            "missing_runtime_paths",
            "runtime does not define all required artifact paths",
            invalid=invalid,
        )
    return (
        runtime_object,
        values["recommendations_dir"],
        values["cohort_dir"],
        run_root,
        not invalid,
    )


def _statistical_analysis(statistics_inputs_valid, by_key, sequences, config, policy, arms=None):
    expected_count = sum(len(family["evaluated_comparisons"]) for family in policy["families"].values())
    summary: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    analysis_errors: list[dict[str, Any]] = []
    analysis_warnings: list[dict[str, Any]] = []
    if statistics_inputs_valid:
        try:
            (
                summary,
                diagnostics,
                comparisons,
                comparison_errors,
                comparison_warnings,
            ) = statistics(
                by_key,
                users=sorted(sequences),
                seeds=list(config.model.seeds),
                config=config,
                policy=policy,
                arms=arms,
            )
            analysis_errors.extend(comparison_errors)
            analysis_warnings.extend(comparison_warnings)
        except (KeyError, TypeError, ValueError) as exc:
            _error(
                analysis_errors,
                "comparison_computation_failed",
                "failed to compute the declared comparison families",
                error=str(exc),
            )

    if not summary or (expected_count and not comparisons):
        statistical_status = "not_computed"
    elif analysis_errors:
        statistical_status = "computed_with_errors"
    elif analysis_warnings:
        statistical_status = "computed_with_warnings"
    else:
        statistical_status = "computed"

    return (
        summary,
        diagnostics,
        comparisons,
        {
            "status": statistical_status,
            "expected_comparison_count": expected_count,
            "computed_comparison_count": len(comparisons),
            "errors": analysis_errors,
            "warnings": analysis_warnings,
        },
    )


def diagnose_recommendations(
    config: ValidationConfig,
    runtime: dict[str, Any],
    decision_config: dict[str, Any],
    *,
    scene_duration: int = 30,
    arms=None,
) -> dict[str, Any]:
    arms = RECOMMENDATION_ARMS if arms is None else arms
    errors: list[dict[str, Any]] = []
    settings, decision_config_valid = _validate_decision_config(decision_config, errors)
    primary_metric = f"NDCG@{config.evaluation.primary_cutoff}"
    policy = multiple_comparison_policy(settings, decision_config_valid, primary_metric, arms=arms)

    (
        runtime_object,
        recommendations_dir,
        cohort_dir,
        run_root,
        runtime_paths_valid,
    ) = _runtime_locations(runtime, errors)

    catalog_rows, catalog_loaded = _read_jsonl(
        cohort_dir / "catalog.jsonl", "cohort catalog", errors
    )
    item_content, content_ids, catalog_valid = _catalog_contract(
        catalog_rows, catalog_loaded, errors
    )
    sequence_rows, sequences_loaded = _read_jsonl(
        cohort_dir / "sequences.jsonl", "cohort sequences", errors
    )
    sequences, sequences_valid = _sequence_contract(
        sequence_rows,
        sequences_loaded,
        item_content,
        max(0, config.cohort.min_sequence_length - 2),
        errors,
    )
    eligibility, eligibility_loaded = _read_json(
        cohort_dir / "eligibility_summary.json", "cohort eligibility summary", errors
    )
    eligibility, eligibility_valid = _eligibility_contract(
        eligibility,
        eligibility_loaded,
        catalog_size=len(item_content),
        user_count=len(sequences),
        errors=errors,
    )
    cohort_summary, cohort_plan_valid = _cohort_plan_contract(config, cohort_dir, errors)
    metadata_title_rows, metadata_titles_loaded = _read_jsonl(
        cohort_dir / "metadata_titles.jsonl", "metadata titles", errors
    )
    metadata_title_summary, metadata_titles_valid = _metadata_title_contract(
        metadata_title_rows,
        metadata_titles_loaded,
        catalog_rows,
        errors,
    )
    representation_summary, representations_valid = _representation_contract(
        run_root,
        item_ids=list(item_content),
        embedding_dim=config.encoder.embedding_dim,
        errors=errors,
        arms=arms,
    )
    configured_user_count_matches = len(sequences) == config.cohort.user_count
    if not configured_user_count_matches:
        _error(
            errors,
            "configured_user_count_mismatch",
            "sequence user count does not match validation config",
            configured_user_count=config.cohort.user_count,
            sequence_users=len(sequences),
        )

    metric_rows, metrics_loaded = _read_jsonl(
        recommendations_dir / "per_user_metrics.jsonl",
        "per-user recommendation metrics",
        errors,
    )
    by_key, grid, grid_complete, rows_valid = _recommendation_contract(
        metric_rows,
        metrics_loaded,
        seeds=list(config.model.seeds),
        users=sequences,
        item_content=item_content,
        cutoffs=list(config.evaluation.cutoffs),
        errors=errors,
        arms=arms,
    )
    checkpoint_summary, checkpoints_complete = _checkpoint_contract(
        recommendations_dir, list(config.model.seeds), errors, arms=arms,
    )
    training_run_summary, training_runs_complete = _training_run_contract(
        recommendations_dir,
        run_id=runtime_object.get("run_id"),
        seeds=list(config.model.seeds),
        catalog_size=len(item_content),
        errors=errors,
        arms=arms,
    )

    (
        scene_coverage,
        scene_outcomes_complete,
        minimum_scene_coverage_met,
        coverage_gap_within_limit,
    ) = _scene_coverage(
        run_root,
        content_ids,
        errors,
        scene_duration,
        settings,
        decision_config_valid,
        runtime_paths_valid,
        branches=set(arms.values()),
    )

    statistics_inputs_valid = (
        decision_config_valid
        and catalog_valid
        and sequences_valid
        and grid_complete
        and rows_valid
        and cohort_plan_valid
        and eligibility_valid
    )
    summary, diagnostics, comparisons, analysis = _statistical_analysis(
        statistics_inputs_valid,
        by_key,
        sequences,
        config,
        policy,
        arms,
    )

    checks = {
        "runtime_paths_valid": runtime_paths_valid,
        "decision_config_valid": decision_config_valid,
        "catalog_valid": catalog_valid,
        "sequences_valid": sequences_valid,
        "eligibility_matches_artifacts": eligibility_valid,
        "cohort_plan_matches_artifacts": cohort_plan_valid,
        "metadata_titles_valid": metadata_titles_valid,
        "representations_valid": representations_valid,
        "configured_user_count_matches": configured_user_count_matches,
        "recommendation_grid_complete": grid_complete,
        "recommendation_rows_valid": rows_valid,
        "training_runs_complete": training_runs_complete,
        "checkpoints_complete": checkpoints_complete,
        "scene_outcomes_complete": scene_outcomes_complete,
        "minimum_scene_coverage_met": minimum_scene_coverage_met,
        "arm_scene_coverage_gap_within_limit": coverage_gap_within_limit,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": "diagnosis/v4",
        "run_id": runtime_object.get("run_id"),
        **target_scope(arms),
        "modality": runtime_object.get("modality"),
        "runtime_decision": {
            "status": status,
            "checks": checks,
            "errors": errors,
        },
        "artifact_integrity": {
            "catalog_size": len(item_content),
            "sequence_user_count": len(sequences),
            "configured_user_count": config.cohort.user_count,
            "catalog_scope": CATALOG_SCOPE,
            "required_catalog_size": eligibility.get("required_item_count"),
            "candidate_user_count": eligibility.get("candidate_user_count"),
            "metadata_titles": metadata_title_summary,
            "representations": representation_summary,
            "seed_count": len(config.model.seeds),
            "arm_count": len(arms),
            "recommendation_grid": grid,
            "training_runs": training_run_summary,
            "checkpoints": checkpoint_summary,
        },
        "cohort": cohort_summary,
        "scene_coverage": scene_coverage,
        "multiple_comparison_policy": policy,
        "statistical_analysis": analysis,
        "metrics": summary,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
    }
