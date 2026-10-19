from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from artifact_io import read_jsonl, write_json


MAX_ERROR_EXAMPLES = 10


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
    reader=read_jsonl,
) -> tuple[list[Any], bool]:
    if not path.is_file():
        if required and report_error:
            _error(errors, "missing_artifact", f"missing {label}", path=str(path))
        return [], not required
    try:
        return list(reader(path)), True
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


def write_diagnosis(path, document):
    """Publish final scores and numerical comparisons without per-item evidence."""
    from validation.recommendation_contracts import EVALUATION_CONTRACT

    result = {
        "schema_version": "leave-one-out-diagnosis/v4",
        "evaluation_contract": dict(EVALUATION_CONTRACT),
        "run_id": document["run_id"],
        "representation_mode": document["representation_mode"],
        **({"scene_aggregation": document["scene_aggregation"]}
           if "scene_aggregation" in document else {}),
        "runtime_decision": document["runtime_decision"],
        "statistics": {key: document["statistics"][key]
                       for key in ("status", "comparisons", "comparison_policy", "targets", "comparison_count", "correction") if key in document["statistics"]},
    }
    if "comparisons" in result["statistics"]:
        result["statistics"]["metric"] = "NDCG@10"
    if "recommendations" in document:
        result["recommendations"] = {"means": document["recommendations"]["means"]}
    if "run_comparison" in document:
        comparison = document["run_comparison"]
        result["run_comparison"] = {
            key: comparison[key]
            for key in ("reference_run_id", "direction", "metric", "comparisons", "targets", "correction", "family_size")
            if key in comparison
        }
    write_json(path, result)
