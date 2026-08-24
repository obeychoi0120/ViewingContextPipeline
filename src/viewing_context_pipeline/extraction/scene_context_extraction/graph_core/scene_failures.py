"""Terminal Scene Context failure sidecar helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCENE_FAILURE_FIELDS = {
    "content_id",
    "scene_idx",
    "keyframes",
    "error",
}
SCENE_FAILURE_OPTIONAL_FIELDS = {"raw_output_text"}


def build_scene_failure_record(
    *,
    content_id: str,
    scene_idx: Any,
    keyframes: Iterable[Any],
    error: str,
    raw_output_text: str | None = None,
) -> dict[str, Any]:
    message = str(error).strip() or "scene extraction failed"
    record = {
        "content_id": str(content_id),
        "scene_idx": scene_idx,
        "keyframes": list(keyframes),
        "error": message,
    }
    if raw_output_text is not None:
        record["raw_output_text"] = str(raw_output_text)
    return record


def read_scene_failures(path: str | Path) -> list[dict[str, Any]]:
    failure_path = Path(path)
    if not failure_path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()
    with failure_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{failure_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if (
                not isinstance(record, dict)
                or not SCENE_FAILURE_FIELDS.issubset(record)
                or not (
                    set(record) - SCENE_FAILURE_FIELDS
                ).issubset(SCENE_FAILURE_OPTIONAL_FIELDS)
            ):
                raise ValueError(
                    f"{failure_path}:{line_number}: invalid failure record fields"
                )
            if not isinstance(record["content_id"], str) or not record[
                "content_id"
            ]:
                raise ValueError(
                    f"{failure_path}:{line_number}: content_id must be non-empty"
                )
            if not isinstance(record["keyframes"], list):
                raise ValueError(
                    f"{failure_path}:{line_number}: keyframes must be a list"
                )
            if not isinstance(record["error"], str) or not record["error"].strip():
                raise ValueError(
                    f"{failure_path}:{line_number}: error must be non-empty"
                )
            if "raw_output_text" in record and not isinstance(
                record["raw_output_text"],
                str,
            ):
                raise ValueError(
                    f"{failure_path}:{line_number}: raw_output_text must be a string"
                )
            scene_key = str(record["scene_idx"])
            if scene_key in seen_scene_ids:
                raise ValueError(
                    f"{failure_path}:{line_number}: duplicate scene_idx "
                    f"{record['scene_idx']}"
                )
            seen_scene_ids.add(scene_key)
            records.append(record)
    return records


def matching_scene_failures(
    records: Iterable[dict[str, Any]],
    *,
    content_id: str,
    expected_keyframes: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    matching: dict[str, dict[str, Any]] = {}
    for record in records:
        scene_key = str(record.get("scene_idx"))
        if (
            record.get("content_id") == content_id
            and scene_key in expected_keyframes
            and record.get("keyframes") == expected_keyframes[scene_key]
        ):
            matching[scene_key] = record
    return [
        matching[scene_key]
        for scene_key in expected_keyframes
        if scene_key in matching
    ]


def replace_jsonl_atomic(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    output_path = Path(path)
    records_to_write = list(records)
    if not records_to_write:
        output_path.unlink(missing_ok=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            for record in records_to_write:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def failure_aggregation_warning(
    failures: Iterable[dict[str, Any]],
) -> str | None:
    records = list(failures)
    if not records:
        return None
    scene_ids = ",".join(str(record["scene_idx"]) for record in records)
    return (
        "excluded terminal Scene Context failures: "
        f"count={len(records)} scene_ids={scene_ids}"
    )
