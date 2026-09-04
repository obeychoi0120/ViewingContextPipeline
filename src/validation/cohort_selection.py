from __future__ import annotations

import hashlib
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any


PLAN_SCHEMA_VERSION = "microlens-user-cohort-plan/v1"
ELIGIBILITY_SCHEMA_VERSION = "microlens-cohort-eligibility/v3"
COHORT_SAMPLING = "user_first_nested_stratified"
CATALOG_SCOPE = "selected_user_sequence_union"
METADATA_TITLE_SCHEMA_VERSION = "metadata-title/v1"


class CohortError(RuntimeError):
    pass


def normalize_item_id(value: object) -> str:
    text = str(value).strip()
    if not text.isdigit() or int(text) <= 0:
        raise CohortError(f"invalid item id: {value!r}")
    return str(int(text))


def content_id_for_item(item_id: str) -> str:
    return f"microlens_100k_{int(item_id):05d}"


def history_stratum(length: int, boundaries: list[int]) -> str:
    for start, end in zip(boundaries, boundaries[1:]):
        if start <= length < end:
            return f"{start}-{end - 1}"
    return f"{boundaries[-1]}+"


def _stable_key(seed: int, user_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest(), user_id


@dataclass
class _StratumCursor:
    lower: int
    rows: list[tuple[str, list[str]]]
    index: int = 0

    def __lt__(self, other: "_StratumCursor") -> bool:
        # Compare (2r+1)/(2n) exactly; the common factor 2 cancels.
        left = (2 * self.index + 1) * len(other.rows)
        right = (2 * other.index + 1) * len(self.rows)
        if left != right:
            return left < right
        return (self.lower, self.rows[self.index][0]) < (other.lower, other.rows[other.index][0])


def split_record(row: dict[str, Any]) -> dict[str, Any]:
    sequence = row["sequence"]
    return {
        **row,
        "train": sequence[:-2],
        "valid_target": sequence[-2],
        "test_target": sequence[-1],
    }


def select_users(
    pairs: list[tuple[str, list[str]]],
    *,
    count: int,
    seed: int,
    boundaries: list[int],
    min_length: int,
    max_length: int,
) -> list[dict[str, Any]]:
    """Return a prefix of one asset-independent, stratified user order."""
    grouped: dict[int, list[tuple[str, list[str]]]] = defaultdict(list)
    for user_id, sequence in pairs:
        if len(sequence) >= min_length:
            lower = max(start for start in boundaries if start <= len(sequence))
            grouped[lower].append((user_id, sequence))
    available = sum(map(len, grouped.values()))
    if count <= 0 or count > available:
        raise CohortError(f"requested {count} users but only {available} are candidates")
    queue = [
        _StratumCursor(lower, sorted(rows, key=lambda row: _stable_key(seed, row[0])))
        for lower, rows in grouped.items()
    ]
    heapq.heapify(queue)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        cursor = heapq.heappop(queue)
        user_id, sequence = cursor.rows[cursor.index]
        selected.append(
            split_record(
                {
                    "cohort_rank": len(selected) + 1,
                    "user_id": user_id,
                    "original_length": len(sequence),
                    "stratum": history_stratum(len(sequence), boundaries),
                    "sequence": sequence[-max_length:],
                }
            )
        )
        if cursor.index + 1 < len(cursor.rows):
            heapq.heappush(queue, _StratumCursor(cursor.lower, cursor.rows, cursor.index + 1))
    return selected


def required_items(selected: list[dict[str, Any]]) -> list[dict[str, str]]:
    items = {item for row in selected for item in row["sequence"]}
    return [
        {"item_id": item, "content_id": content_id_for_item(item)}
        for item in sorted(items, key=int)
    ]


def _length_statistics(lengths: list[int]) -> dict[str, int | float]:
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "median": median(lengths),
    }


def cohort_statistics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Cold targets refer to the global training item set for each phase."""
    train_items = {item for row in selected for item in row["train"]}
    refit_items = train_items | {row["valid_target"] for row in selected}
    count = len(selected)

    def unseen(field: str, seen: set[str]) -> dict[str, int | float]:
        missing = sum(row[field] not in seen for row in selected)
        return {"count": missing, "user_count": count, "fraction": missing / count}

    truncated = sum(len(row["sequence"]) < row["original_length"] for row in selected)
    return {
        "user_count": count,
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in selected).items())),
        "original_history_length": _length_statistics([row["original_length"] for row in selected]),
        "retained_sequence_length": _length_statistics([len(row["sequence"]) for row in selected]),
        "truncated_user_count": truncated,
        "truncated_user_fraction": truncated / count,
        "required_item_count": len({item for row in selected for item in row["sequence"]}),
        "selection": {
            "interaction_count": sum(len(row["train"]) for row in selected),
            "unique_item_count": len(train_items),
            "valid_target_unseen": unseen("valid_target", train_items),
            "test_target_unseen": unseen("test_target", train_items),
        },
        "refit": {
            "interaction_count": sum(len(row["train"]) + 1 for row in selected),
            "unique_item_count": len(refit_items),
            "test_target_unseen": unseen("test_target", refit_items),
        },
    }


def build_cohort_plan(
    pairs: list[tuple[str, list[str]]],
    *,
    run_id: str,
    settings: dict[str, Any],
    inputs: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    candidate_lengths = [
        len(sequence) for _, sequence in pairs if len(sequence) >= settings["min_sequence_length"]
    ]
    candidate_count = len(candidate_lengths)
    count = settings["user_count"]
    if count > candidate_count:
        raise CohortError(f"requested {count} users but only {candidate_count} are candidates")
    sizes = sorted({count, 1000, 10000, 100000})
    largest = max(size for size in sizes if size <= candidate_count)
    ordered = select_users(
        pairs,
        count=largest,
        seed=settings["seed"],
        boundaries=settings["history_strata"],
        min_length=settings["min_sequence_length"],
        max_length=settings["max_sequence_length"],
    )
    selected = ordered[:count]
    items = required_items(selected)
    scales = {}
    for size in sizes:
        if size <= candidate_count:
            scales[str(size)] = {"status": "available", **cohort_statistics(ordered[:size])}
        else:
            scales[str(size)] = {
                "status": "insufficient_candidates",
                "requested_users": size,
                "candidate_user_count": candidate_count,
            }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "cohort_sampling": COHORT_SAMPLING,
        "catalog_scope": CATALOG_SCOPE,
        "settings": settings,
        "inputs": inputs,
        "pairs_user_count": len(pairs),
        "candidate_user_count": candidate_count,
        "candidate_stratum_counts": dict(
            sorted(
                Counter(
                    history_stratum(length, settings["history_strata"])
                    for length in candidate_lengths
                ).items()
            )
        ),
        "selected_user_count": count,
        "required_item_count": len(items),
        "statistics": cohort_statistics(selected),
        "scale_statistics": scales,
    }
    return plan, selected, items


def validate_plan(
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    run_id: str,
    settings: dict[str, Any],
    inputs: dict[str, str],
) -> None:
    """Validate saved selection without consulting external media or annotation files."""
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("run_id") != run_id
        or plan.get("cohort_sampling") != COHORT_SAMPLING
        or plan.get("catalog_scope") != CATALOG_SCOPE
        or plan.get("settings") != settings
        or plan.get("inputs") != inputs
    ):
        raise CohortError("cohort plan/config mismatch; use a new run_id")
    if (
        type(plan.get("selected_user_count")) is not int
        or len(selected) != settings["user_count"]
        or plan["selected_user_count"] != len(selected)
    ):
        raise CohortError("selected user count does not match cohort plan")
    candidate_count = plan.get("candidate_user_count")
    pairs_count = plan.get("pairs_user_count")
    strata = plan.get("candidate_stratum_counts")
    if (
        type(candidate_count) is not int
        or type(pairs_count) is not int
        or not pairs_count >= candidate_count >= len(selected)
        or not isinstance(strata, dict)
        or any(type(value) is not int or value < 0 for value in strata.values())
        or sum(strata.values()) != candidate_count
    ):
        raise CohortError("invalid candidate pool counts in cohort plan")
    seen = set()
    fields = {
        "cohort_rank",
        "user_id",
        "original_length",
        "stratum",
        "sequence",
        "train",
        "valid_target",
        "test_target",
    }
    for rank, row in enumerate(selected, start=1):
        if not isinstance(row, dict) or set(row) != fields:
            raise CohortError("invalid selected user fields")
        user_id, sequence, original = row["user_id"], row["sequence"], row["original_length"]
        if (
            type(row["cohort_rank"]) is not int
            or row["cohort_rank"] != rank
            or not isinstance(user_id, str)
            or not user_id.strip()
            or user_id in seen
            or type(original) is not int
            or original < settings["min_sequence_length"]
            or not isinstance(sequence, list)
            or len(sequence) != min(original, settings["max_sequence_length"])
        ):
            raise CohortError(f"invalid selected user at cohort rank {rank}")
        for item in sequence:
            if not isinstance(item, str) or normalize_item_id(item) != item:
                raise CohortError(f"invalid sequence item at cohort rank {rank}")
        if (
            row["train"] != sequence[:-2]
            or row["valid_target"] != sequence[-2]
            or row["test_target"] != sequence[-1]
            or row["stratum"] != history_stratum(original, settings["history_strata"])
        ):
            raise CohortError(f"sequence/split mismatch at cohort rank {rank}")
        seen.add(user_id)
    if (
        items != required_items(selected)
        or type(plan.get("required_item_count")) is not int
        or plan["required_item_count"] != len(items)
    ):
        raise CohortError("required items do not match the selected sequence union")
    if any(
        count > strata.get(stratum, 0)
        for stratum, count in Counter(row["stratum"] for row in selected).items()
    ):
        raise CohortError("selected strata exceed the candidate pool")
    statistics = cohort_statistics(selected)
    scales = plan.get("scale_statistics")
    if (
        plan.get("statistics") != statistics
        or not isinstance(scales, dict)
        or scales.get(str(len(selected))) != {"status": "available", **statistics}
    ):
        raise CohortError("cohort plan statistics do not match selected users")
