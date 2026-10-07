from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from artifact_io import read_jsonl


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
