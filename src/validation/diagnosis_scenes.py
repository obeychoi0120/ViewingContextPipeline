from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from visual_sampling import truncate_timestamp


from .diagnosis_support import (
    _bounded_examples,
    _error,
    _read_json,
    _read_jsonl,
)


SCENE_ARMS = {
    "graph_qwen": (
        "extraction/graph/qwen/scenes",
        "extraction/graph/qwen/scenes/failures",
    ),
    "graph_gemini": (
        "extraction/graph/gemini/scenes",
        "extraction/graph/gemini/scenes/failures",
    ),
    "desc": (
        "extraction/description/scenes",
        "extraction/description/scenes/failures",
    ),
}


def _expected_scenes(
    run_root: Path,
    content_ids: list[str],
    errors: list[dict[str, Any]],
    scene_duration: int = 30,
) -> tuple[set[tuple[str, int]], bool]:
    expected: set[tuple[str, int]] = set()
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for content_id in content_ids:
        path = (
            run_root
            / "data"
            / "cohort"
            / "source_assets"
            / content_id
            / "assets"
            / f"timestamp_fixed_{scene_duration}s.json"
        )
        value, loaded = _read_json(
            path,
            "fixed-window timestamp artifact",
            errors,
            report_error=False,
        )
        if not loaded:
            issues["missing_or_invalid_timestamp"] += 1
            examples.append({"content_id": content_id, "path": str(path)})
            continue
        if not isinstance(value, list) or not value:
            issues["empty_or_non_list_timestamp"] += 1
            examples.append({"content_id": content_id, "path": str(path)})
            continue
        seen: set[int] = set()
        for fallback_idx, scene in enumerate(value):
            if not isinstance(scene, dict):
                issues["timestamp_scene_not_object"] += 1
                examples.append({"content_id": content_id, "scene_index": fallback_idx})
                continue
            scene_idx = scene.get("scene_idx", fallback_idx)
            if not isinstance(scene_idx, int) or isinstance(scene_idx, bool) or scene_idx < 0:
                issues["invalid_expected_scene_idx"] += 1
                examples.append({"content_id": content_id, "scene_index": fallback_idx})
                continue
            if scene_idx in seen:
                issues["duplicate_expected_scene_idx"] += 1
                examples.append({"content_id": content_id, "scene_idx": scene_idx})
                continue
            seen.add(scene_idx)
            expected.add((content_id, scene_idx))
    valid = bool(expected) and not issues
    if issues:
        _error(
            errors,
            "invalid_scene_denominator",
            "fixed-window timestamp artifacts do not define one valid scene denominator",
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    if not expected:
        _error(errors, "empty_scene_denominator", "fixed-window scene denominator is empty")
    return expected, valid


def _nonempty_timestamp_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            type(item) in (int, float)
            and math.isfinite(item)
            and item >= 0
            and truncate_timestamp(item) == item
            for item in value
        )
        and value == sorted(set(value))
    )


def _success_scene_row_issues(
    arm: str,
    row: dict[str, Any],
    content_id: str,
) -> list[str]:
    invalid: list[str] = []
    if not _nonempty_timestamp_list(row.get("keyframes")):
        invalid.append("invalid_success_keyframes")
    if arm.startswith("graph_"):
        if set(row) != {
            "scene_idx",
            "keyframes",
            "graph",
            "parse_mode",
            "semantic_warnings",
        }:
            invalid.append("invalid_graph_scene_fields")
        if not isinstance(row.get("graph"), dict):
            invalid.append("invalid_graph")
        if row.get("parse_mode") not in {"native", "repaired", "unknown"}:
            invalid.append("invalid_parse_mode")
        if not isinstance(row.get("semantic_warnings"), list):
            invalid.append("invalid_semantic_warnings")
    else:
        if set(row) != {
            "schema_version",
            "content_id",
            "scene_idx",
            "keyframes",
            "description",
        }:
            invalid.append("invalid_description_scene_fields")
        if row.get("schema_version") != "scene-description/v1":
            invalid.append("invalid_description_schema_version")
        if row.get("content_id") != content_id:
            invalid.append("description_content_id_mismatch")
        description = row.get("description")
        if not isinstance(description, str) or not description.strip():
            invalid.append("empty_description")
    return invalid


def _scene_arm_contract(
    arm: str,
    run_root: Path,
    content_ids: list[str],
    expected: set[tuple[str, int]],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[tuple[str, int]], bool]:
    scene_relative, failure_relative = SCENE_ARMS[arm]
    scene_dir = run_root / scene_relative
    failure_dir = run_root / failure_relative
    catalog_contents = set(content_ids)
    success: set[tuple[str, int]] = set()
    failures: set[tuple[str, int]] = set()
    parse_modes: dict[tuple[str, int], str] = {}
    semantic_warning_scenes: set[tuple[str, int]] = set()
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    existing_scene_files = set()
    if scene_dir.is_dir():
        existing_scene_files = {path.stem for path in scene_dir.glob("*.jsonl")}
    else:
        issues["missing_scene_directory"] += 1
        examples.append({"path": str(scene_dir)})
    extra_scene_files = sorted(existing_scene_files - catalog_contents)
    if extra_scene_files:
        issues["extra_scene_files"] += len(extra_scene_files)
        examples.extend({"content_id": value} for value in _bounded_examples(extra_scene_files))

    existing_failure_files = (
        {path.stem for path in failure_dir.glob("*.jsonl")} if failure_dir.is_dir() else set()
    )
    extra_failure_files = sorted(existing_failure_files - catalog_contents)
    if extra_failure_files:
        issues["extra_failure_files"] += len(extra_failure_files)
        examples.extend({"content_id": value} for value in _bounded_examples(extra_failure_files))

    for content_id in content_ids:
        scene_rows, scene_loaded = _read_jsonl(
            scene_dir / f"{content_id}.jsonl",
            f"{arm} scene outcomes",
            errors,
            report_error=False,
        )
        if not scene_loaded:
            issues["missing_or_invalid_scene_file"] += 1
        failure_rows, failure_loaded = _read_jsonl(
            failure_dir / f"{content_id}.jsonl",
            f"{arm} failure outcomes",
            errors,
            required=False,
            report_error=False,
        )
        if not failure_loaded:
            issues["invalid_failure_file"] += 1
        for outcome, rows, destination in (
            ("success", scene_rows, success),
            ("failure", failure_rows, failures),
        ):
            local: set[int] = set()
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    issues[f"{outcome}_row_not_object"] += 1
                    examples.append(
                        {"content_id": content_id, "outcome": outcome, "row_index": row_index}
                    )
                    continue
                scene_idx = row.get("scene_idx")
                if not isinstance(scene_idx, int) or isinstance(scene_idx, bool) or scene_idx < 0:
                    issues[f"invalid_{outcome}_scene_idx"] += 1
                    examples.append(
                        {"content_id": content_id, "outcome": outcome, "row_index": row_index}
                    )
                    continue
                if scene_idx in local:
                    issues[f"duplicate_{outcome}_scene_idx"] += 1
                    examples.append(
                        {"content_id": content_id, "outcome": outcome, "scene_idx": scene_idx}
                    )
                    continue
                local.add(scene_idx)
                key = (content_id, scene_idx)
                if key not in expected:
                    issues[f"unexpected_{outcome}_scene"] += 1
                    examples.append(
                        {"content_id": content_id, "outcome": outcome, "scene_idx": scene_idx}
                    )
                    continue
                if outcome == "success":
                    row_issues = _success_scene_row_issues(arm, row, content_id)
                    if row_issues:
                        for issue in row_issues:
                            issues[issue] += 1
                        examples.append(
                            {
                                "content_id": content_id,
                                "outcome": outcome,
                                "scene_idx": scene_idx,
                                "reasons": row_issues,
                            }
                        )
                        continue
                destination.add(key)
                if outcome == "success" and arm.startswith("graph_"):
                    parse_mode = row["parse_mode"]
                    parse_modes[key] = str(parse_mode)
                    warnings = row["semantic_warnings"]
                    if warnings:
                        semantic_warning_scenes.add(key)

    overlap = success & failures
    missing = expected - (success | failures)
    if overlap:
        issues["success_failure_overlap"] += len(overlap)
        examples.extend(
            {"content_id": content_id, "scene_idx": scene_idx}
            for content_id, scene_idx in _bounded_examples(sorted(overlap))
        )
    if missing:
        issues["missing_scene_outcome"] += len(missing)
        examples.extend(
            {"content_id": content_id, "scene_idx": scene_idx}
            for content_id, scene_idx in _bounded_examples(sorted(missing))
        )

    denominator = len(expected)
    success_count = len(success)
    failure_count = len(failures)
    outcome_count = len(success | failures)
    document = {
        "expected_scene_count": denominator,
        "success_scene_count": success_count,
        "failure_scene_count": failure_count,
        "accounted_scene_count": outcome_count,
        "success_coverage": success_count / denominator if denominator else 0.0,
        "failure_rate": failure_count / denominator if denominator else 0.0,
        "outcome_coverage": outcome_count / denominator if denominator else 0.0,
        "missing_scene_count": len(missing),
        "overlap_scene_count": len(overlap),
        "issue_counts": dict(sorted(issues.items())),
        "issue_examples": _bounded_examples(examples),
    }
    if arm.startswith("graph_"):
        document["parse_mode_counts"] = dict(sorted(Counter(parse_modes.values()).items()))
        document["semantic_warning_scene_count"] = len(semantic_warning_scenes)
        document["semantic_warning_rate_among_successes"] = (
            len(semantic_warning_scenes) / success_count if success_count else 0.0
        )
    valid = bool(expected) and not issues
    if issues:
        _error(
            errors,
            "invalid_scene_outcomes",
            f"{arm} scene success/failure outcomes violate the coverage contract",
            arm=arm,
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    return document, success, valid


def _scene_coverage(
    run_root,
    content_ids,
    errors,
    scene_duration,
    settings,
    decision_config_valid,
    runtime_paths_valid,
):
    expected_scenes, scene_denominator_valid = _expected_scenes(
        run_root,
        content_ids,
        errors,
        scene_duration,
    )
    scene_documents: dict[str, Any] = {}
    successful_scenes: dict[str, set[tuple[str, int]]] = {}
    scene_arm_valid: dict[str, bool] = {}
    for arm in SCENE_ARMS:
        document, success, valid = _scene_arm_contract(
            arm, run_root, content_ids, expected_scenes, errors
        )
        scene_documents[arm] = document
        successful_scenes[arm] = success
        scene_arm_valid[arm] = valid

    coverages = [float(document["success_coverage"]) for document in scene_documents.values()]
    observed_gap = max(coverages) - min(coverages) if coverages else 1.0
    minimum_scene_coverage_met = (
        decision_config_valid
        and scene_denominator_valid
        and all(value >= float(settings["min_scene_coverage"]) for value in coverages)
    )
    coverage_gap_within_limit = (
        decision_config_valid
        and scene_denominator_valid
        and observed_gap <= float(settings["max_arm_coverage_gap"])
    )
    if not minimum_scene_coverage_met:
        _error(
            errors,
            "minimum_scene_coverage_not_met",
            "one or more arms are below the configured successful-scene coverage",
            minimum=settings.get("min_scene_coverage"),
            observed={arm: value["success_coverage"] for arm, value in scene_documents.items()},
        )
    if not coverage_gap_within_limit:
        _error(
            errors,
            "arm_scene_coverage_gap_exceeded",
            "successful-scene coverage gap exceeds the configured maximum",
            maximum=settings.get("max_arm_coverage_gap"),
            observed=observed_gap,
        )
    common_success = set.intersection(*successful_scenes.values()) if successful_scenes else set()
    scene_outcomes_complete = (
        runtime_paths_valid and scene_denominator_valid and all(scene_arm_valid.values())
    )
    scene_coverage = {
        "denominator_source": f"fixed_{scene_duration}s timestamp artifacts",
        "expected_scene_count": len(expected_scenes),
        "minimum_success_coverage": settings.get("min_scene_coverage"),
        "maximum_arm_coverage_gap": settings.get("max_arm_coverage_gap"),
        "observed_arm_coverage_gap": observed_gap,
        "common_success_scene_count": len(common_success),
        "common_success_coverage": (
            len(common_success) / len(expected_scenes) if expected_scenes else 0.0
        ),
        "common_scene_intersection_required": False,
        "arms": scene_documents,
    }

    return (
        scene_coverage,
        scene_outcomes_complete,
        minimum_scene_coverage_met,
        coverage_gap_within_limit,
    )
