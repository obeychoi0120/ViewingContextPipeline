from __future__ import annotations

import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any


from .metrics import metrics_from_rank
from .recommendation_contracts import (
    ARCHITECTURE_VERSION,
    RECOMMENDATION_ARMS,
    TRAINING_RUN_SCHEMA_VERSION,
    TRAINING_RUNS_FILENAME,
)


from .diagnosis_support import (
    _bounded_examples,
    _error,
    _finite_number,
    _read_jsonl,
    _row_issue,
)


def _row_target_contract(row, index, key, catalog_size, users, item_content, issues, examples):
    _, user_id, arm = key
    candidate_count = row.get("candidate_count")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count != catalog_size
    ):
        _row_issue(
            issues,
            examples,
            "candidate_count_mismatch",
            index,
            cell=list(key),
            value=candidate_count,
        )
    if arm in RECOMMENDATION_ARMS and row.get("branch") != RECOMMENDATION_ARMS[arm]:
        _row_issue(
            issues,
            examples,
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
            issues,
            examples,
            "target_outside_catalog",
            index,
            cell=list(key),
            value=target,
        )
    if expected_target is not None and target != expected_target:
        _row_issue(
            issues,
            examples,
            "target_mismatch",
            index,
            cell=list(key),
            expected=expected_target,
            value=target,
        )
    expected_content = item_content.get(target) if isinstance(target, str) else None
    if expected_content is not None and row.get("target_content_id") != expected_content:
        _row_issue(
            issues,
            examples,
            "target_content_mismatch",
            index,
            cell=list(key),
        )
    return target


def _row_rank_contract(row, index, key, catalog_size, cutoffs, issues, examples):
    rank = row.get("rank")
    rank_valid = isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= catalog_size
    if not rank_valid:
        _row_issue(
            issues,
            examples,
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
                    issues,
                    examples,
                    "metric_mismatch",
                    index,
                    cell=list(key),
                    metric=metric,
                    expected=expected_value,
                    value=actual,
                )
    return rank, rank_valid


def _row_top_contract(
    row, index, key, target, rank, rank_valid, expected_top_count, item_content, issues, examples
):
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
            issues,
            examples,
            "invalid_top_items",
            index,
            cell=list(key),
        )
    else:
        outside = sorted({item for item in top_items if item not in item_content})
        if outside:
            _row_issue(
                issues,
                examples,
                "top_item_outside_catalog",
                index,
                cell=list(key),
                items=_bounded_examples(outside),
            )
        expected_contents = [item_content[item] for item in top_items if item in item_content]
        if top_contents != expected_contents:
            _row_issue(
                issues,
                examples,
                "top_content_mismatch",
                index,
                cell=list(key),
            )
        if rank_valid:
            expected_position = rank - 1
            if expected_position < expected_top_count:
                if top_items[expected_position] != target:
                    _row_issue(
                        issues,
                        examples,
                        "rank_top_order_mismatch",
                        index,
                        cell=list(key),
                    )
            elif target in top_items:
                _row_issue(
                    issues,
                    examples,
                    "rank_top_order_mismatch",
                    index,
                    cell=list(key),
                )


def _row_bucket_contract(row, index, key, user_id, expected_buckets, issues, examples):
    bucket = row.get("target_frequency_bucket")
    if not isinstance(bucket, str):
        _row_issue(
            issues,
            examples,
            "invalid_target_frequency_bucket",
            index,
            cell=list(key),
        )
    elif user_id in expected_buckets and bucket != expected_buckets[user_id]:
        _row_issue(
            issues,
            examples,
            "target_frequency_bucket_mismatch",
            index,
            cell=list(key),
            expected=expected_buckets[user_id],
            value=bucket,
        )


def _recommendation_contract(
    rows: list[Any],
    loaded: bool,
    *,
    seeds: list[int],
    users: dict[str, dict[str, Any]],
    item_content: dict[str, str],
    cutoffs: list[int],
    errors: list[dict[str, Any]],
    arms=None,
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Any], bool, bool]:
    arms = list(RECOMMENDATION_ARMS if arms is None else arms)
    rows = _selected_rows(rows, arms)
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

        target = _row_target_contract(
            row,
            index,
            key,
            catalog_size,
            users,
            item_content,
            value_issues,
            value_examples,
        )
        rank, rank_valid = _row_rank_contract(
            row,
            index,
            key,
            catalog_size,
            cutoffs,
            value_issues,
            value_examples,
        )
        _row_top_contract(
            row,
            index,
            key,
            target,
            rank,
            rank_valid,
            expected_top_count,
            item_content,
            value_issues,
            value_examples,
        )
        _row_bucket_contract(
            row, index, key, user_id, expected_buckets, value_issues, value_examples
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


def _selected_rows(rows, arms):
    # Keep malformed/unknown rows visible to validation; ignore only known excluded arms.
    excluded = set(RECOMMENDATION_ARMS) - set(arms)
    return [row for row in rows if not (
        isinstance(row, dict) and isinstance(row.get("arm"), str) and row["arm"] in excluded
    )]


def _checkpoint_contract(
    recommendations_dir: Path,
    seeds: list[int],
    errors: list[dict[str, Any]],
    arms=None,
) -> tuple[dict[str, Any], bool]:
    arms = RECOMMENDATION_ARMS if arms is None else arms
    paths = [
        recommendations_dir / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
        for seed in seeds
        for arm in arms
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
    arms=None,
) -> tuple[dict[str, Any], bool]:
    path = recommendations_dir / TRAINING_RUNS_FILENAME
    rows, loaded = _read_jsonl(path, "recommendation training runs", errors)
    arms = RECOMMENDATION_ARMS if arms is None else arms
    rows = _selected_rows(rows, arms)
    expected = set(product(seeds, arms))
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
