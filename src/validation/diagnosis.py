from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .config import ValidationConfig
from .diagnosis_statistics import multiple_comparison_policy, statistics
from .io import read_jsonl
from .metrics import metrics_from_rank
from .recommendation_contracts import (
    ARCHITECTURE_VERSION,
    RECOMMENDATION_ARMS,
    TRAINING_RUN_SCHEMA_VERSION,
    TRAINING_RUNS_FILENAME,
)


MAX_ERROR_EXAMPLES = 10
DECISION_CONFIG_KEYS = {
    "min_scene_coverage",
    "max_arm_coverage_gap",
    "familywise_alpha",
    "multiple_comparison_correction",
}
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


def _read_jsonl(
    path: Path,
    label: str,
    errors: list[dict[str, Any]],
    *,
    required: bool = True,
    report_error: bool = True,
) -> tuple[list[Any], bool]:
    if not path.is_file():
        if required and report_error:
            _error(errors, "missing_artifact", f"missing {label}", path=str(path))
        return [], not required
    try:
        return list(read_jsonl(path)), True
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if report_error:
            _error(
                errors,
                "invalid_artifact",
                f"failed to read {label}",
                path=str(path),
                error=str(exc),
            )
        return [], False


def _read_json(
    path: Path,
    label: str,
    errors: list[dict[str, Any]],
    *,
    report_error: bool = True,
) -> tuple[Any, bool]:
    if not path.is_file():
        if report_error:
            _error(errors, "missing_artifact", f"missing {label}", path=str(path))
        return None, False
    try:
        return json.loads(path.read_text(encoding="utf-8")), True
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if report_error:
            _error(
                errors,
                "invalid_artifact",
                f"failed to read {label}",
                path=str(path),
                error=str(exc),
            )
        return None, False


def _bounded_examples(values: list[Any]) -> list[Any]:
    return values[:MAX_ERROR_EXAMPLES]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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


def _catalog_contract(
    rows: list[Any],
    loaded: bool,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str], bool]:
    item_content: dict[str, str] = {}
    content_ids: list[str] = []
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues["row_not_object"] += 1
            examples.append({"row_index": index, "reason": "row_not_object"})
            continue
        item_id = row.get("item_id")
        content_id = row.get("content_id")
        if not isinstance(item_id, str) or not item_id:
            issues["invalid_item_id"] += 1
            examples.append({"row_index": index, "reason": "invalid_item_id"})
            continue
        if not isinstance(content_id, str) or not content_id:
            issues["invalid_content_id"] += 1
            examples.append({"row_index": index, "reason": "invalid_content_id"})
            continue
        if item_id in item_content:
            issues["duplicate_item_id"] += 1
            examples.append({"row_index": index, "item_id": item_id})
            continue
        if content_id in content_ids:
            issues["duplicate_content_id"] += 1
            examples.append({"row_index": index, "content_id": content_id})
            continue
        item_content[item_id] = content_id
        content_ids.append(content_id)
    valid = loaded and bool(rows) and not issues and len(item_content) == len(rows)
    if loaded and not rows:
        issues["empty_catalog"] += 1
    if issues:
        _error(
            errors,
            "invalid_catalog",
            "catalog rows violate the item/content contract",
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    return item_content, content_ids, valid


def _sequence_contract(
    rows: list[Any],
    loaded: bool,
    item_content: dict[str, str],
    min_train_length: int,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    by_user: dict[str, dict[str, Any]] = {}
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues["row_not_object"] += 1
            examples.append({"row_index": index, "reason": "row_not_object"})
            continue
        user_id = row.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            issues["invalid_user_id"] += 1
            examples.append({"row_index": index, "reason": "invalid_user_id"})
            continue
        if user_id in by_user:
            issues["duplicate_user_id"] += 1
            examples.append({"row_index": index, "user_id": user_id})
            continue
        train = row.get("train")
        valid_target = row.get("valid_target")
        test_target = row.get("test_target")
        if not isinstance(train, list) or any(not isinstance(item, str) for item in train):
            issues["invalid_train"] += 1
            examples.append({"row_index": index, "user_id": user_id, "reason": "invalid_train"})
        elif len(train) < min_train_length:
            issues["train_too_short"] += 1
            examples.append(
                {
                    "row_index": index,
                    "user_id": user_id,
                    "reason": "train_too_short",
                    "minimum": min_train_length,
                    "actual": len(train),
                }
            )
        referenced = [*train] if isinstance(train, list) else []
        for label, target in (("valid_target", valid_target), ("test_target", test_target)):
            if not isinstance(target, str):
                issues[f"invalid_{label}"] += 1
                examples.append(
                    {"row_index": index, "user_id": user_id, "reason": f"invalid_{label}"}
                )
            else:
                referenced.append(target)
        missing_items = sorted(
            {item for item in referenced if isinstance(item, str) and item not in item_content}
        )
        if missing_items:
            issues["sequence_item_outside_catalog"] += 1
            examples.append(
                {
                    "row_index": index,
                    "user_id": user_id,
                    "missing_items": _bounded_examples(missing_items),
                }
            )
        by_user[user_id] = row
    valid = loaded and bool(rows) and not issues and len(by_user) == len(rows)
    if loaded and not rows:
        issues["empty_sequences"] += 1
    if issues:
        _error(
            errors,
            "invalid_sequences",
            "sequence rows violate the cohort contract",
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    return by_user, valid


def _eligibility_contract(
    value: Any,
    loaded: bool,
    *,
    catalog_size: int,
    user_count: int,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if not loaded or not isinstance(value, dict):
        if loaded:
            _error(
                errors,
                "invalid_eligibility",
                "eligibility summary must be an object",
            )
        return {}, False
    eligible_items = value.get("eligible_items")
    eligible_users = value.get("eligible_users")
    title_coverage = value.get("metadata_title_coverage")
    valid = (
        value.get("schema_version") == "microlens-cohort-eligibility/v2"
        and isinstance(eligible_items, int)
        and not isinstance(eligible_items, bool)
        and isinstance(eligible_users, int)
        and not isinstance(eligible_users, bool)
        and eligible_items == catalog_size
        and eligible_users >= user_count
        and isinstance(title_coverage, dict)
        and title_coverage.get("schema_version") == "metadata-title/v1"
        and title_coverage.get("catalog_item_count") == catalog_size
        and title_coverage.get("covered_item_count") == catalog_size
        and title_coverage.get("missing_item_count") == 0
    )
    if not valid:
        _error(
            errors,
            "eligibility_mismatch",
            "catalog/sequences are inconsistent with the eligible item/user counts",
            eligible_items=eligible_items,
            catalog_size=catalog_size,
            eligible_users=eligible_users,
            sequence_users=user_count,
        )
    return value, valid


def _metadata_title_contract(
    rows: list[Any],
    loaded: bool,
    catalog_rows: list[Any],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, (row, catalog_row) in enumerate(zip(rows, catalog_rows, strict=False)):
        if not isinstance(row, dict) or set(row) != {"item_id", "content_id", "title"}:
            _row_issue(issues, examples, "invalid_metadata_title_fields", index)
            continue
        if not isinstance(catalog_row, dict):
            _row_issue(issues, examples, "invalid_catalog_reference", index)
            continue
        if row.get("item_id") != catalog_row.get("item_id"):
            _row_issue(issues, examples, "metadata_item_order_mismatch", index)
        if row.get("content_id") != catalog_row.get("content_id"):
            _row_issue(issues, examples, "metadata_content_order_mismatch", index)
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            _row_issue(issues, examples, "invalid_metadata_title", index)
    if len(rows) != len(catalog_rows):
        issues["metadata_title_count_mismatch"] += 1
        examples.append(
            {
                "reason": "metadata_title_count_mismatch",
                "expected": len(catalog_rows),
                "observed": len(rows),
            }
        )
    valid = loaded and bool(rows) and not issues and len(rows) == len(catalog_rows)
    if issues or (loaded and not rows):
        if loaded and not rows:
            issues["empty_metadata_titles"] += 1
        _error(
            errors,
            "invalid_metadata_titles",
            "metadata titles must exactly cover the catalog in catalog order",
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    return {
        "schema_version": "metadata-title/v1",
        "expected_count": len(catalog_rows),
        "observed_count": len(rows),
        "issue_counts": dict(sorted(issues.items())),
    }, valid


def _representation_contract(
    run_root: Path,
    *,
    item_ids: list[str],
    embedding_dim: int,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    directory = run_root / "validation" / "representations"
    expected_index = {item_id: index for index, item_id in enumerate(item_ids)}
    item_index, index_loaded = _read_json(
        directory / "item_index.json", "representation item index", errors
    )
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    if index_loaded and item_index != expected_index:
        issues["item_index_mismatch"] += 1
        examples.append({"reason": "item_index_mismatch"})

    branches: dict[str, Any] = {}
    for branch in RECOMMENDATION_ARMS.values():
        path = directory / f"{branch}_embeddings.npz"
        branch_document: dict[str, Any] = {"path": str(path)}
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"values"}:
                    raise ValueError("NPZ must contain exactly the values array")
                values = archive["values"]
            branch_document["shape"] = list(values.shape)
            branch_document["finite"] = bool(np.isfinite(values).all())
            if values.shape != (len(item_ids), embedding_dim):
                issues[f"{branch}_shape_mismatch"] += 1
                examples.append(
                    {
                        "reason": f"{branch}_shape_mismatch",
                        "expected": [len(item_ids), embedding_dim],
                        "observed": list(values.shape),
                    }
                )
            if not branch_document["finite"]:
                issues[f"{branch}_non_finite"] += 1
                examples.append({"reason": f"{branch}_non_finite"})
        except (OSError, KeyError, TypeError, ValueError) as exc:
            issues[f"{branch}_unreadable"] += 1
            branch_document["error"] = str(exc)
            examples.append({"reason": f"{branch}_unreadable", "error": str(exc)})
        branches[branch] = branch_document

    valid = index_loaded and item_index == expected_index and not issues
    if issues:
        _error(
            errors,
            "invalid_representations",
            "all four embedding artifacts must match the catalog order and shape",
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    return {
        "item_index_path": str(directory / "item_index.json"),
        "expected_shape": [len(item_ids), embedding_dim],
        "branches": branches,
        "issue_counts": dict(sorted(issues.items())),
    }, valid


def _row_issue(
    counts: Counter[str],
    examples: list[dict[str, Any]],
    code: str,
    row_index: int,
    **details: Any,
) -> None:
    counts[code] += 1
    if len(examples) < MAX_ERROR_EXAMPLES:
        examples.append({"row_index": row_index, "reason": code, **details})


def _recommendation_contract(
    rows: list[Any],
    loaded: bool,
    *,
    seeds: list[int],
    users: dict[str, dict[str, Any]],
    item_content: dict[str, str],
    cutoffs: list[int],
    errors: list[dict[str, Any]],
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Any], bool, bool]:
    arms = list(RECOMMENDATION_ARMS)
    expected = set(product(seeds, sorted(users), arms))
    cell_counts: Counter[tuple[int, str, str]] = Counter()
    malformed_cell_rows = 0
    value_issues: Counter[str] = Counter()
    value_examples: list[dict[str, Any]] = []
    canonical: dict[tuple[int, str, str], dict[str, Any]] = {}
    catalog_size = len(item_content)
    expected_top_count = min(20, catalog_size)
    train_frequency = Counter(
        item
        for sequence in users.values()
        for item in sequence.get("train", [])
        if isinstance(item, str)
    )
    nonzero_frequency = sorted(train_frequency.values())
    median_frequency = nonzero_frequency[len(nonzero_frequency) // 2] if nonzero_frequency else 0
    expected_buckets = {
        user_id: (
            "cold"
            if train_frequency[sequence.get("test_target")] == 0
            else "low"
            if train_frequency[sequence.get("test_target")] <= median_frequency
            else "warm"
        )
        for user_id, sequence in users.items()
        if isinstance(sequence.get("test_target"), str)
    }

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            malformed_cell_rows += 1
            _row_issue(value_issues, value_examples, "row_not_object", index)
            continue
        seed = row.get("seed")
        user_id = row.get("user_id")
        arm = row.get("arm")
        cell_valid = (
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(user_id, str)
            and isinstance(arm, str)
        )
        if not cell_valid:
            malformed_cell_rows += 1
            _row_issue(value_issues, value_examples, "invalid_cell_key", index)
            continue
        key = (seed, user_id, arm)
        cell_counts[key] += 1
        canonical.setdefault(key, row)

        candidate_count = row.get("candidate_count")
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count != catalog_size
        ):
            _row_issue(
                value_issues,
                value_examples,
                "candidate_count_mismatch",
                index,
                cell=list(key),
                value=candidate_count,
            )
        if arm in RECOMMENDATION_ARMS and row.get("branch") != RECOMMENDATION_ARMS[arm]:
            _row_issue(
                value_issues,
                value_examples,
                "branch_mismatch",
                index,
                cell=list(key),
                value=row.get("branch"),
            )

        expected_target = users.get(user_id, {}).get("test_target")
        target = row.get("target_item_id")
        target_is_catalog_item = isinstance(target, str) and target in item_content
        if not target_is_catalog_item:
            _row_issue(
                value_issues,
                value_examples,
                "target_outside_catalog",
                index,
                cell=list(key),
                value=target,
            )
        if expected_target is not None and target != expected_target:
            _row_issue(
                value_issues,
                value_examples,
                "target_mismatch",
                index,
                cell=list(key),
                expected=expected_target,
                value=target,
            )
        expected_content = item_content.get(target) if isinstance(target, str) else None
        if expected_content is not None and row.get("target_content_id") != expected_content:
            _row_issue(
                value_issues,
                value_examples,
                "target_content_mismatch",
                index,
                cell=list(key),
            )

        rank = row.get("rank")
        rank_valid = (
            isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= catalog_size
        )
        if not rank_valid:
            _row_issue(
                value_issues,
                value_examples,
                "invalid_rank",
                index,
                cell=list(key),
                value=rank,
            )
        else:
            expected_metrics = metrics_from_rank(rank, cutoffs)
            for metric, expected_value in expected_metrics.items():
                actual = row.get(metric)
                if not _finite_number(actual) or not math.isclose(
                    float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-12
                ):
                    _row_issue(
                        value_issues,
                        value_examples,
                        "metric_mismatch",
                        index,
                        cell=list(key),
                        metric=metric,
                        expected=expected_value,
                        value=actual,
                    )

        top_items = row.get("top_item_ids")
        top_contents = row.get("top_content_ids")
        top_valid = (
            isinstance(top_items, list)
            and len(top_items) == expected_top_count
            and all(isinstance(item, str) for item in top_items)
            and len(set(top_items)) == len(top_items)
        )
        if not top_valid:
            _row_issue(
                value_issues,
                value_examples,
                "invalid_top_items",
                index,
                cell=list(key),
            )
        else:
            outside = sorted({item for item in top_items if item not in item_content})
            if outside:
                _row_issue(
                    value_issues,
                    value_examples,
                    "top_item_outside_catalog",
                    index,
                    cell=list(key),
                    items=_bounded_examples(outside),
                )
            expected_contents = [item_content[item] for item in top_items if item in item_content]
            if top_contents != expected_contents:
                _row_issue(
                    value_issues,
                    value_examples,
                    "top_content_mismatch",
                    index,
                    cell=list(key),
                )
            if rank_valid:
                expected_position = rank - 1
                if expected_position < expected_top_count:
                    if top_items[expected_position] != target:
                        _row_issue(
                            value_issues,
                            value_examples,
                            "rank_top_order_mismatch",
                            index,
                            cell=list(key),
                        )
                elif target in top_items:
                    _row_issue(
                        value_issues,
                        value_examples,
                        "rank_top_order_mismatch",
                        index,
                        cell=list(key),
                    )
        bucket = row.get("target_frequency_bucket")
        if not isinstance(bucket, str):
            _row_issue(
                value_issues,
                value_examples,
                "invalid_target_frequency_bucket",
                index,
                cell=list(key),
            )
        elif user_id in expected_buckets and bucket != expected_buckets[user_id]:
            _row_issue(
                value_issues,
                value_examples,
                "target_frequency_bucket_mismatch",
                index,
                cell=list(key),
                expected=expected_buckets[user_id],
                value=bucket,
            )

    observed = set(cell_counts)
    duplicates = sorted(key for key, count in cell_counts.items() if count > 1)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    grid = {
        "expected_cell_count": len(expected),
        "observed_row_count": len(rows),
        "unique_cell_count": len(observed),
        "duplicate_cell_count": len(duplicates),
        "missing_cell_count": len(missing),
        "extra_cell_count": len(extra),
        "malformed_cell_row_count": malformed_cell_rows,
        "duplicate_examples": [list(value) for value in _bounded_examples(duplicates)],
        "missing_examples": [list(value) for value in _bounded_examples(missing)],
        "extra_examples": [list(value) for value in _bounded_examples(extra)],
        "row_issue_counts": dict(sorted(value_issues.items())),
        "row_issue_examples": value_examples,
    }
    grid_complete = (
        loaded
        and not duplicates
        and not missing
        and not extra
        and not malformed_cell_rows
        and len(rows) == len(expected)
    )
    rows_valid = loaded and not value_issues
    if not grid_complete:
        _error(
            errors,
            "recommendation_grid_incomplete",
            "per-user metrics do not contain exactly one row for every seed/user/arm cell",
            expected_cell_count=grid["expected_cell_count"],
            observed_row_count=grid["observed_row_count"],
            duplicate_cell_count=grid["duplicate_cell_count"],
            missing_cell_count=grid["missing_cell_count"],
            extra_cell_count=grid["extra_cell_count"],
            malformed_cell_row_count=grid["malformed_cell_row_count"],
            duplicate_examples=grid["duplicate_examples"],
            missing_examples=grid["missing_examples"],
            extra_examples=grid["extra_examples"],
        )
    if value_issues:
        _error(
            errors,
            "invalid_recommendation_rows",
            "per-user metric rows violate target/rank/metric/catalog contracts",
            issue_counts=dict(sorted(value_issues.items())),
            examples=value_examples,
        )
    return canonical, grid, grid_complete, rows_valid


def _checkpoint_contract(
    recommendations_dir: Path,
    seeds: list[int],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    paths = [
        recommendations_dir / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
        for seed in seeds
        for arm in RECOMMENDATION_ARMS
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    empty: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            if path.stat().st_size <= 0:
                empty.append(str(path))
        except OSError:
            empty.append(str(path))
    if missing:
        _error(
            errors,
            "missing_checkpoints",
            "one or more recommendation checkpoints are missing",
            missing_count=len(missing),
            examples=_bounded_examples(missing),
        )
    if empty:
        _error(
            errors,
            "empty_checkpoints",
            "one or more recommendation checkpoints are empty or unreadable",
            empty_count=len(empty),
            examples=_bounded_examples(empty),
        )
    return {
        "expected_count": len(paths),
        "existing_count": len(paths) - len(missing),
        "nonempty_count": len(paths) - len(missing) - len(empty),
        "missing_count": len(missing),
        "empty_count": len(empty),
        "missing_examples": _bounded_examples(missing),
        "empty_examples": _bounded_examples(empty),
    }, not missing and not empty


def _canonical_checkpoint_path(
    recommendations_dir: Path,
    seed: int,
    arm: str,
) -> Path:
    return recommendations_dir / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"


def _paths_match(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).resolve() == expected.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _best_validation_is_consistent(value: Any, epochs: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(epochs, list) or not epochs:
        return False
    metric = value.get("metric")
    best_value = value.get("value")
    best_epoch = value.get("epoch")
    if (
        metric != "NDCG@10"
        or not _finite_number(best_value)
        or not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or best_epoch <= 0
    ):
        return False
    epoch_values: dict[int, float] = {}
    for row in epochs:
        if not isinstance(row, dict):
            return False
        epoch = row.get("epoch")
        metric_value = row.get(metric)
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch <= 0
            or epoch in epoch_values
            or not _finite_number(metric_value)
        ):
            return False
        epoch_values[epoch] = float(metric_value)
    declared = float(best_value)
    return (
        best_epoch in epoch_values
        and math.isclose(epoch_values[best_epoch], declared, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(max(epoch_values.values()), declared, rel_tol=1e-9, abs_tol=1e-12)
    )


def _refit_history_is_consistent(value: Any, best_epoch: Any) -> bool:
    if not isinstance(value, dict) or value.get("data") != "train+valid_target":
        return False
    epochs = value.get("epochs")
    completed = value.get("epochs_completed")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or completed != best_epoch
        or not isinstance(epochs, list)
        or len(epochs) != best_epoch
    ):
        return False
    for expected_epoch, row in enumerate(epochs, start=1):
        if (
            not isinstance(row, dict)
            or row.get("epoch") != expected_epoch
            or not _finite_number(row.get("loss"))
        ):
            return False
    return True


def _training_run_contract(
    recommendations_dir: Path,
    *,
    run_id: Any,
    seeds: list[int],
    catalog_size: int,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    path = recommendations_dir / TRAINING_RUNS_FILENAME
    rows, loaded = _read_jsonl(path, "recommendation training runs", errors)
    expected = set(product(seeds, RECOMMENDATION_ARMS))
    cell_counts: Counter[tuple[int, str]] = Counter()
    malformed = 0
    issues: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            malformed += 1
            issues["row_not_object"] += 1
            examples.append({"row_index": index, "reason": "row_not_object"})
            continue
        seed = row.get("seed")
        arm = row.get("arm")
        if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(arm, str):
            malformed += 1
            issues["invalid_cell_key"] += 1
            examples.append({"row_index": index, "reason": "invalid_cell_key"})
            continue
        key = (seed, arm)
        cell_counts[key] += 1
        if row.get("schema_version") != TRAINING_RUN_SCHEMA_VERSION:
            issues["schema_version_mismatch"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        if row.get("architecture_version") != ARCHITECTURE_VERSION:
            issues["architecture_version_mismatch"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        if row.get("run_id") != run_id:
            issues["run_id_mismatch"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        if row.get("branch") != RECOMMENDATION_ARMS.get(arm):
            issues["branch_mismatch"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        if row.get("candidate_count") != catalog_size:
            issues["candidate_count_mismatch"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        selection = row.get("selection")
        if not isinstance(selection, dict):
            issues["missing_epoch_history"] += 1
            examples.append({"row_index": index, "cell": list(key)})
            selection = {}
        epochs = selection.get("epochs")
        if (
            not isinstance(epochs, list)
            or not epochs
            or selection.get("epochs_completed") != len(epochs)
            or not isinstance(selection.get("early_stopped"), bool)
        ):
            issues["missing_epoch_history"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        best_validation = selection.get("best_validation")
        if not _best_validation_is_consistent(best_validation, epochs):
            issues["invalid_best_validation"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        best_epoch = best_validation.get("epoch") if isinstance(best_validation, dict) else None
        if not _refit_history_is_consistent(row.get("refit"), best_epoch):
            issues["invalid_refit_history"] += 1
            examples.append({"row_index": index, "cell": list(key)})
        canonical = _canonical_checkpoint_path(recommendations_dir, seed, arm)
        if not _paths_match(row.get("checkpoint"), canonical):
            issues["checkpoint_path_mismatch"] += 1
            examples.append(
                {
                    "row_index": index,
                    "cell": list(key),
                    "expected": str(canonical),
                    "value": row.get("checkpoint"),
                }
            )

    observed = set(cell_counts)
    duplicates = sorted(key for key, count in cell_counts.items() if count > 1)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    summary = {
        "path": str(path),
        "expected_cell_count": len(expected),
        "observed_row_count": len(rows),
        "unique_cell_count": len(observed),
        "duplicate_cell_count": len(duplicates),
        "missing_cell_count": len(missing),
        "extra_cell_count": len(extra),
        "malformed_cell_row_count": malformed,
        "duplicate_examples": [list(value) for value in _bounded_examples(duplicates)],
        "missing_examples": [list(value) for value in _bounded_examples(missing)],
        "extra_examples": [list(value) for value in _bounded_examples(extra)],
        "row_issue_counts": dict(sorted(issues.items())),
        "row_issue_examples": _bounded_examples(examples),
    }
    complete = (
        loaded
        and len(rows) == len(expected)
        and not duplicates
        and not missing
        and not extra
        and not malformed
        and not issues
    )
    if not complete:
        _error(
            errors,
            "training_runs_incomplete",
            "training history must contain one valid record for every seed/arm cell",
            duplicate_cell_count=len(duplicates),
            missing_cell_count=len(missing),
            extra_cell_count=len(extra),
            malformed_cell_row_count=malformed,
            row_issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    return summary, complete


def _expected_scenes(
    run_root: Path,
    content_ids: list[str],
    errors: list[dict[str, Any]],
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
            / "timestamp_fixed_30s.json"
        )
        value, loaded = _read_json(
            path,
            "fixed-30s timestamp artifact",
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
            "fixed-30s timestamp artifacts do not define one valid scene denominator",
            issue_counts=dict(sorted(issues.items())),
            examples=_bounded_examples(examples),
        )
    if not expected:
        _error(errors, "empty_scene_denominator", "fixed-30s scene denominator is empty")
    return expected, valid


def _nonempty_int_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _success_scene_row_issues(
    arm: str,
    row: dict[str, Any],
    content_id: str,
) -> list[str]:
    invalid: list[str] = []
    if not _nonempty_int_list(row.get("keyframes")):
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


def diagnose_recommendations(
    config: ValidationConfig,
    runtime: dict[str, Any],
    decision_config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    settings, decision_config_valid = _validate_decision_config(decision_config, errors)
    primary_metric = f"NDCG@{config.evaluation.primary_cutoff}"
    policy = multiple_comparison_policy(settings, decision_config_valid, primary_metric)

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
    )
    checkpoint_summary, checkpoints_complete = _checkpoint_contract(
        recommendations_dir, list(config.model.seeds), errors
    )
    training_run_summary, training_runs_complete = _training_run_contract(
        recommendations_dir,
        run_id=runtime_object.get("run_id"),
        seeds=list(config.model.seeds),
        catalog_size=len(item_content),
        errors=errors,
    )

    expected_scenes, scene_denominator_valid = _expected_scenes(run_root, content_ids, errors)
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
        "denominator_source": "fixed_30s timestamp artifacts",
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

    summary: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    analysis_errors: list[dict[str, Any]] = []
    analysis_warnings: list[dict[str, Any]] = []
    statistics_inputs_valid = (
        decision_config_valid and catalog_valid and sequences_valid and grid_complete and rows_valid
    )
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

    if not comparisons:
        statistical_status = "not_computed"
    elif analysis_errors:
        statistical_status = "computed_with_errors"
    elif analysis_warnings:
        statistical_status = "computed_with_warnings"
    else:
        statistical_status = "computed"

    checks = {
        "runtime_paths_valid": runtime_paths_valid,
        "decision_config_valid": decision_config_valid,
        "catalog_valid": catalog_valid,
        "sequences_valid": sequences_valid,
        "eligibility_matches_artifacts": eligibility_valid,
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
        "schema_version": "diagnosis/v3",
        "run_id": runtime_object.get("run_id"),
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
            "eligible_catalog_size": eligibility.get("eligible_items"),
            "eligible_user_count": eligibility.get("eligible_users"),
            "metadata_titles": metadata_title_summary,
            "representations": representation_summary,
            "seed_count": len(config.model.seeds),
            "arm_count": len(RECOMMENDATION_ARMS),
            "recommendation_grid": grid,
            "training_runs": training_run_summary,
            "checkpoints": checkpoint_summary,
        },
        "scene_coverage": scene_coverage,
        "multiple_comparison_policy": policy,
        "statistical_analysis": {
            "status": statistical_status,
            "expected_comparison_count": 6,
            "computed_comparison_count": len(comparisons),
            "errors": analysis_errors,
            "warnings": analysis_warnings,
        },
        "metrics": summary,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
    }
