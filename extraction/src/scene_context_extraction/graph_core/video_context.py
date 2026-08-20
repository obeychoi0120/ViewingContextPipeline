"""Build content-level video contexts from scene-context JSONL records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypedDict

from .aggregator import build_session_aggregate
from .build_graph import ingest_scene
from .graph_store import GraphStore


CONTEXT_FIELDS = {
    "content_axes_4d",
    "content_axis_distribution",
    "top_styles",
    "top_moods",
    "top_scene_functions",
    "top_entities",
    "top_motifs",
}
VIDEO_CONTEXT_FIELDS = {
    "content_id",
    "source_scene_context_path",
    "context",
    "aggregation_warnings",
}


class GraphVideoContextDocument(TypedDict):
    content_id: str
    source_scene_context_path: str
    context: dict[str, Any]
    aggregation_warnings: list[str]


def read_scene_context_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"scene context record must be an object: {path}")
                records.append(record)
    return records


def aggregate_scene_context(
    content_id: str,
    scene_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    store = GraphStore(":memory:")
    try:
        forest_id = store.get_or_create_forest("current")
        session_id = store.get_or_create_session(
            forest_id=forest_id,
            session_key=content_id,
            session_index=0,
        )
        for fallback_idx, record in enumerate(scene_records):
            observation = record.get("vlm_visual_graph")
            if observation is None:
                continue
            if not isinstance(observation, dict):
                raise ValueError("vlm_visual_graph must be an object or null")
            scene_idx = _parse_int(record.get("scene_idx"), fallback_idx)
            stats = ingest_scene(
                store=store,
                forest_id=forest_id,
                session_db_id=session_id,
                session_key=content_id,
                scene_key=f"scene_{scene_idx:03d}",
                scene_index=scene_idx,
                observation=observation,
            )
            warnings.extend(
                f"scene_{scene_idx:03d}: {warning}"
                for warning in stats.get("warnings", [])
            )
        return build_session_aggregate(store, session_id), warnings
    finally:
        store.close()


def build_video_context_document(
    content_id: str,
    source_scene_context_path: str | Path,
    *,
    extra_warnings: list[str] | None = None,
) -> GraphVideoContextDocument:
    source_path = Path(source_scene_context_path)
    context, warnings = aggregate_scene_context(
        content_id,
        read_scene_context_records(source_path),
    )
    warnings.extend(extra_warnings or [])
    return {
        "content_id": content_id,
        "source_scene_context_path": str(source_scene_context_path),
        "context": context,
        "aggregation_warnings": warnings,
    }


def write_video_context(
    content_id: str,
    source_scene_context_path: str | Path,
    output_path: str | Path,
    *,
    extra_warnings: list[str] | None = None,
) -> GraphVideoContextDocument:
    document = build_video_context_document(
        content_id,
        source_scene_context_path,
        extra_warnings=extra_warnings,
    )
    _write_json_atomic(output_path, document)
    return document


def video_context_is_valid(
    path: str | Path,
    *,
    content_id: str,
    source_scene_context_path: str | Path,
    extra_warnings: list[str] | None = None,
) -> bool:
    context_path = Path(path)
    if not context_path.is_file():
        return False
    try:
        document = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    structurally_valid = (
        isinstance(document, dict)
        and set(document) == VIDEO_CONTEXT_FIELDS
        and document.get("content_id") == content_id
        and document.get("source_scene_context_path") == str(source_scene_context_path)
        and isinstance(document.get("context"), dict)
        and set(document["context"]) == CONTEXT_FIELDS
        and isinstance(document.get("aggregation_warnings"), list)
    )
    if not structurally_valid:
        return False
    try:
        return document == build_video_context_document(
            content_id,
            source_scene_context_path,
            extra_warnings=extra_warnings,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _write_json_atomic(path: str | Path, document: dict[str, Any]) -> None:
    output_path = Path(path)
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
            json.dump(document, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default
