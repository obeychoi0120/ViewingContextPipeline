from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import ValidationConfig
from .cohort import load_ready_cohort
from .cohort_selection import (
    CATALOG_SCOPE,
    COHORT_SAMPLING,
    ELIGIBILITY_SCHEMA_VERSION,
    CohortError,
)
from .recommendation_contracts import (
    RECOMMENDATION_ARMS,
)


from .diagnosis_support import (
    _bounded_examples,
    _error,
    _read_json,
    _row_issue,
)


def _catalog_contract(
    rows: list[Any],
    loaded: bool,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str], bool]:
    item_content: dict[str, str] = {}
    content_ids: list[str] = []
    seen_content_ids: set[str] = set()
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
        if content_id in seen_content_ids:
            issues["duplicate_content_id"] += 1
            examples.append({"row_index": index, "content_id": content_id})
            continue
        item_content[item_id] = content_id
        content_ids.append(content_id)
        seen_content_ids.add(content_id)
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
    available_items = value.get("available_item_count")
    candidate_users = value.get("candidate_user_count")
    title_coverage = value.get("metadata_title_coverage")
    valid = (
        value.get("schema_version") == ELIGIBILITY_SCHEMA_VERSION
        and value.get("status") == "ready"
        and value.get("cohort_sampling") == COHORT_SAMPLING
        and value.get("catalog_scope") == CATALOG_SCOPE
        and type(available_items) is int
        and type(candidate_users) is int
        and available_items == value.get("required_item_count") == catalog_size
        and value.get("requested_users") == value.get("selected_user_count") == user_count
        and candidate_users >= user_count
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
            "catalog/sequences are inconsistent with the ready user-first cohort",
            status=value.get("status"),
            available_items=available_items,
            catalog_size=catalog_size,
            candidate_users=candidate_users,
            sequence_users=user_count,
        )
    return value, valid


def _cohort_plan_contract(
    config: ValidationConfig,
    cohort_dir: Path,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    try:
        cohort = load_ready_cohort(
            cohort_dir,
            run_id=config.run_id,
        )
    except (CohortError, KeyError, TypeError, ValueError) as exc:
        _error(
            errors,
            "cohort_plan_mismatch",
            "cohort must be ready and match its frozen selection, sequence union, and inventory",
            error=str(exc),
        )
        return {}, False
    eligibility = cohort["eligibility"]
    return {
        "status": eligibility["status"],
        "cohort_sampling": COHORT_SAMPLING,
        "catalog_scope": CATALOG_SCOPE,
        "pairs_user_count": eligibility["pairs_user_count"],
        "candidate_user_count": eligibility["candidate_user_count"],
        "selected_user_count": len(cohort["selected_users"]),
        "required_item_count": len(cohort["required_items"]),
        "available_item_count": eligibility["available_item_count"],
        "statistics": cohort["plan"]["statistics"],
        "media": eligibility["media"],
    }, True


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
    arms=None,
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
    for branch in (RECOMMENDATION_ARMS if arms is None else arms).values():
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
