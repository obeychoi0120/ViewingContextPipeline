from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


from .io import read_jsonl


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
