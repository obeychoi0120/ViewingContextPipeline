"""Compare complete Graph and Reference video context sets."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.common.manifest import read_manifest_rows
from src.scene_context_extraction.graph_core.scoring import CONTENT_AXIS_ORDER
from src.scene_context_extraction.graph_core.video_context import (
    CONTEXT_FIELDS,
    VIDEO_CONTEXT_FIELDS,
)


TOP_LIST_FIELDS = (
    "top_styles",
    "top_moods",
    "top_scene_functions",
    "top_entities",
    "top_motifs",
)
AXIS_BUCKETS = ("negative", "neutral", "positive")
REPORT_FILENAME = "vc_graph_ref_comparison.json"
PER_CONTENT_FILENAME = "vc_graph_ref_per_content.csv"
ROUND_DIGITS = 6
ERROR_PREVIEW_LIMIT = 20
WORST_CONTENT_LIMIT = 20


class ContextComparisonPreflightError(ValueError):
    """Raised when the manifest does not have two complete valid VC sets."""

    def __init__(self, issues: dict[str, list[str]]) -> None:
        self.issues = {
            name: values
            for name, values in issues.items()
            if values
        }
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = ["video context preflight failed"]
        for name, values in self.issues.items():
            preview = ", ".join(values[:ERROR_PREVIEW_LIMIT])
            remaining = len(values) - ERROR_PREVIEW_LIMIT
            suffix = f", ... {remaining} more" if remaining > 0 else ""
            parts.append(f"{name}={len(values)} [{preview}{suffix}]")
        return "; ".join(parts)


def run_video_context_comparison(
    *,
    manifest_path: str | Path,
    context_dir: str | Path,
    context_ref_dir: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    manifest_rows = read_manifest_rows(manifest_path)
    context_path = Path(context_dir)
    context_ref_path = Path(context_ref_dir)
    pairs = _load_complete_pairs(
        manifest_rows,
        context_path,
        context_ref_path,
    )
    content_rows = [
        compare_context_documents(content_id, context, reference)
        for content_id, context, reference in pairs
    ]
    report = _build_summary_report(
        manifest_path=Path(manifest_path),
        context_dir=context_path,
        context_ref_dir=context_ref_path,
        content_rows=content_rows,
    )
    report_path = Path(report_dir)
    _write_text_atomic(
        report_path / REPORT_FILENAME,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text_atomic(
        report_path / PER_CONTENT_FILENAME,
        _per_content_csv(content_rows),
    )
    return report


def compare_context_documents(
    content_id: str,
    context_document: dict[str, Any],
    reference_document: dict[str, Any],
) -> dict[str, Any]:
    graph = context_document["context"]
    reference = reference_document["context"]
    row: dict[str, Any] = {
        "content_id": content_id,
        "category": content_id.split("_", 1)[0],
    }
    absolute_errors: list[float] = []
    distribution_distances: list[float] = []
    for axis in CONTENT_AXIS_ORDER:
        graph_value = float(graph["content_axes_4d"][axis])
        reference_value = float(reference["content_axes_4d"][axis])
        delta = graph_value - reference_value
        absolute_error = abs(delta)
        distribution_tvd = total_variation_distance(
            graph["content_axis_distribution"][axis],
            reference["content_axis_distribution"][axis],
        )
        row[f"{axis}_graph"] = _rounded(graph_value)
        row[f"{axis}_ref"] = _rounded(reference_value)
        row[f"{axis}_delta"] = _rounded(delta)
        row[f"{axis}_absolute_error"] = _rounded(absolute_error)
        row[f"{axis}_distribution_tvd"] = _rounded(distribution_tvd)
        absolute_errors.append(absolute_error)
        distribution_distances.append(distribution_tvd)

    row["axis_mae"] = _rounded(mean(absolute_errors))
    row["mean_distribution_tvd"] = _rounded(
        mean(distribution_distances)
    )
    top_list_similarities: list[float] = []
    for field in TOP_LIST_FIELDS:
        similarity = jaccard_similarity(
            {item["id"] for item in graph[field]},
            {item["id"] for item in reference[field]},
        )
        row[f"{field}_jaccard"] = _rounded(similarity)
        top_list_similarities.append(similarity)
    row["mean_top_list_jaccard"] = _rounded(
        mean(top_list_similarities)
    )
    row["graph_warning_count"] = len(
        context_document["aggregation_warnings"]
    )
    row["reference_warning_count"] = len(
        reference_document["aggregation_warnings"]
    )
    return row


def total_variation_distance(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    return 0.5 * sum(
        abs(float(left.get(bucket, 0.0)) - float(right.get(bucket, 0.0)))
        for bucket in AXIS_BUCKETS
    )


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _load_complete_pairs(
    manifest_rows: list[dict[str, str]],
    context_dir: Path,
    context_ref_dir: Path,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    issues = {
        "graph_missing": [],
        "reference_missing": [],
        "graph_invalid": [],
        "reference_invalid": [],
    }
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for manifest_row in manifest_rows:
        content_id = manifest_row["content_id"]
        graph = _load_context_side(
            context_dir / f"{content_id}_context_graph_ond.json",
            content_id,
            "graph",
            issues,
        )
        reference = _load_context_side(
            context_ref_dir / f"{content_id}_context_graph_ref.json",
            content_id,
            "reference",
            issues,
        )
        if graph is not None and reference is not None:
            pairs.append((content_id, graph, reference))
    if any(issues.values()):
        raise ContextComparisonPreflightError(issues)
    return pairs


def _load_context_side(
    path: Path,
    content_id: str,
    side: str,
    issues: dict[str, list[str]],
) -> dict[str, Any] | None:
    if not path.is_file():
        issues[f"{side}_missing"].append(content_id)
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        _validate_context_document(document, content_id, path)
        return document
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues[f"{side}_invalid"].append(f"{content_id} ({exc})")
        return None


def _validate_context_document(
    document: Any,
    content_id: str,
    path: Path,
) -> None:
    if not isinstance(document, dict) or set(document) != VIDEO_CONTEXT_FIELDS:
        raise ValueError(f"invalid top-level fields: {path}")
    if document["content_id"] != content_id:
        raise ValueError(
            f"content_id mismatch: expected {content_id}, "
            f"got {document['content_id']!r}"
        )
    if not isinstance(document["source_scene_context_path"], str):
        raise ValueError(f"source_scene_context_path must be a string: {path}")
    warnings = document["aggregation_warnings"]
    if (
        not isinstance(warnings, list)
        or any(not isinstance(warning, str) for warning in warnings)
    ):
        raise ValueError(f"aggregation_warnings must be a string list: {path}")

    context = document["context"]
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
        raise ValueError(f"invalid context fields: {path}")
    _validate_axes(context["content_axes_4d"], path)
    _validate_axis_distributions(
        context["content_axis_distribution"],
        path,
    )
    for field in TOP_LIST_FIELDS:
        _validate_top_list(context[field], field, path)


def _validate_axes(value: Any, path: Path) -> None:
    if not isinstance(value, dict) or set(value) != set(CONTENT_AXIS_ORDER):
        raise ValueError(f"content_axes_4d has invalid axes: {path}")
    for axis in CONTENT_AXIS_ORDER:
        number = _finite_number(value[axis], f"content_axes_4d.{axis}", path)
        if not -1.0 <= number <= 1.0:
            raise ValueError(f"content_axes_4d.{axis} is outside [-1, 1]: {path}")


def _validate_axis_distributions(value: Any, path: Path) -> None:
    if not isinstance(value, dict) or set(value) != set(CONTENT_AXIS_ORDER):
        raise ValueError(
            f"content_axis_distribution has invalid axes: {path}"
        )
    for axis in CONTENT_AXIS_ORDER:
        distribution = value[axis]
        if (
            not isinstance(distribution, dict)
            or not distribution
            or not set(distribution).issubset(AXIS_BUCKETS)
        ):
            raise ValueError(
                f"content_axis_distribution.{axis} has invalid buckets: {path}"
            )
        total = 0.0
        for bucket, raw in distribution.items():
            number = _finite_number(
                raw,
                f"content_axis_distribution.{axis}.{bucket}",
                path,
            )
            if not 0.0 <= number <= 1.0:
                raise ValueError(
                    f"content_axis_distribution.{axis}.{bucket} "
                    f"is outside [0, 1]: {path}"
                )
            total += number
        if not math.isclose(
            total,
            1.0,
            abs_tol=0.01,
        ):
            raise ValueError(
                f"content_axis_distribution.{axis} must sum to 1: {path}"
            )


def _validate_top_list(value: Any, field: str, path: Path) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list: {path}")
    seen_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "count"}:
            raise ValueError(f"{field} has an invalid item: {path}")
        item_id = item["id"]
        count = item["count"]
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{field} item id must be non-empty: {path}")
        if item_id in seen_ids:
            raise ValueError(f"{field} has duplicate id {item_id}: {path}")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError(f"{field} item count must be positive: {path}")
        seen_ids.add(item_id)


def _finite_number(value: Any, field: str, path: Path) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number: {path}")
    return float(value)


def _build_summary_report(
    *,
    manifest_path: Path,
    context_dir: Path,
    context_ref_dir: Path,
    content_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in content_rows:
        categories[row["category"]].append(row)
    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "manifest_path": str(manifest_path.resolve()),
        "context_dir": str(context_dir.resolve()),
        "context_ref_dir": str(context_ref_dir.resolve()),
        "expected_content_count": len(content_rows),
        "paired_content_count": len(content_rows),
        "metrics": _aggregate_metrics(content_rows),
        "categories": {
            category: {
                "content_count": len(rows),
                "metrics": _aggregate_metrics(rows),
                "warning_counts": _warning_counts(rows),
            }
            for category, rows in sorted(categories.items())
        },
        "worst_contents_by_axis_mae": [
            {
                "content_id": row["content_id"],
                "category": row["category"],
                "axis_mae": row["axis_mae"],
            }
            for row in sorted(
                content_rows,
                key=lambda item: (-item["axis_mae"], item["content_id"]),
            )[:WORST_CONTENT_LIMIT]
        ],
        "warning_counts": _warning_counts(content_rows),
    }


def _aggregate_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "overall": _overall_metric_summaries(rows),
        "axes": {
            axis: {
                "signed_delta": _metric_summary(
                    [row[f"{axis}_delta"] for row in rows]
                ),
                "absolute_error": _metric_summary(
                    [row[f"{axis}_absolute_error"] for row in rows]
                ),
                "distribution_tvd": _metric_summary(
                    [row[f"{axis}_distribution_tvd"] for row in rows]
                ),
            }
            for axis in CONTENT_AXIS_ORDER
        },
        "top_lists": {
            field: _metric_summary(
                [row[f"{field}_jaccard"] for row in rows]
            )
            for field in TOP_LIST_FIELDS
        },
    }


def _warning_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "graph": sum(
            row["graph_warning_count"] for row in rows
        ),
        "reference": sum(
            row["reference_warning_count"] for row in rows
        ),
    }


def _overall_metric_summaries(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "axis_mae": _metric_summary(
            [row["axis_mae"] for row in rows]
        ),
        "mean_distribution_tvd": _metric_summary(
            [row["mean_distribution_tvd"] for row in rows]
        ),
        "mean_top_list_jaccard": _metric_summary(
            [row["mean_top_list_jaccard"] for row in rows]
        ),
    }


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _rounded(mean(values)),
        "median": _rounded(median(values)),
        "max": _rounded(max(values)),
    }


def _per_content_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=_per_content_fields(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _per_content_fields() -> list[str]:
    fields = ["content_id", "category"]
    for axis in CONTENT_AXIS_ORDER:
        fields.extend(
            [
                f"{axis}_graph",
                f"{axis}_ref",
                f"{axis}_delta",
                f"{axis}_absolute_error",
                f"{axis}_distribution_tvd",
            ]
        )
    fields.extend(["axis_mae", "mean_distribution_tvd"])
    fields.extend(f"{field}_jaccard" for field in TOP_LIST_FIELDS)
    fields.extend(
        [
            "mean_top_list_jaccard",
            "graph_warning_count",
            "reference_warning_count",
        ]
    )
    return fields


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _rounded(value: float) -> float:
    return round(float(value), ROUND_DIGITS)
