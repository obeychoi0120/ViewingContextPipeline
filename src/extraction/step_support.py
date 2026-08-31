from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.errors import ExtractionStepError
from extraction.evidence import build_scene_evidence
from extraction.monitoring import video_names
from pipeline_runtime import RunContext, read_jsonl, write_jsonl


def write_progress(progress: tqdm, message: str) -> None:
    tqdm.write(message, file=progress.fp)


def complete_content_progress(progress: tqdm) -> None:
    progress.update(1)
    write_progress(progress, "")


def write_failure_jsonl(path: Path, failures: list[dict[str, Any]]) -> None:
    if failures:
        write_jsonl(path, failures)
    else:
        path.unlink(missing_ok=True)


def write_scene_checkpoint(
    scene_path: Path,
    failure_path: Path,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    """Atomically persist the completed subset of one content's scenes."""
    records.sort(key=lambda row: int(row["scene_idx"]))
    failures.sort(key=lambda row: int(row["scene_idx"]))
    write_jsonl(scene_path, records)
    write_failure_jsonl(failure_path, failures)


def video_name_map(context: RunContext) -> dict[str, str]:
    return video_names(read_jsonl(context.cohort_dir / "catalog.jsonl"))


def scene_generation_rows(
    visual: dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    scenes = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in build_scene_evidence(
        scenes, visual["frames_dir"], visual["timestamp_json"]
    ):
        fallback_idx = scene["fallback_idx"]
        scene_idx = scene["scene_idx"]
        keyframes = scene["keyframes"]
        image_paths = scene["image_paths"]
        if not keyframes or len(image_paths) != len(keyframes):
            raise ExtractionStepError(
                f"{visual['content_id']} scene {scene_idx} has "
                f"{len(image_paths)} of {len(keyframes)} keyframes"
            )
        task_id = f"{visual['content_id']}:{fallback_idx}"
        rows.append(
            {
                "task": QwenGenerationTask(
                    task_id=task_id,
                    image_paths=tuple(image_paths),
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                ),
                "scene_idx": scene_idx,
                "scene_start_seconds": scene["scene_start_seconds"],
                "scene_end_seconds": scene["scene_end_seconds"],
                "keyframes": keyframes,
                "image_paths": image_paths,
            }
        )
    if not rows:
        raise ExtractionStepError(f"{visual['content_id']} has no scenes")
    return rows


def result(
    stage: str,
    *,
    content_count: int,
    failure_count: int = 0,
    retry_count: int | None = None,
) -> dict[str, Any]:
    document = {
        "stage": stage,
        "content_count": content_count,
        "failure_count": failure_count,
    }
    if retry_count is not None:
        document["retry_count"] = retry_count
    return document


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ExtractionStepError(f"missing {label}: {path}")
    return path


def minimal_graph_records(
    records: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    required = {
        "scene_idx",
        "keyframes",
        "graph",
        "parse_mode",
        "semantic_warnings",
    }
    invalid = [index for index, row in enumerate(records) if set(row) != required]
    if invalid:
        raise ExtractionStepError(
            f"incompatible graph scene output at rows {invalid[:10]}: {path}; "
            "use --force or a new run_id"
        )
    return records


def minimal_description_records(
    records: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    required = {
        "schema_version",
        "content_id",
        "scene_idx",
        "keyframes",
        "description",
    }
    invalid = [index for index, row in enumerate(records) if set(row) != required]
    if invalid:
        raise ExtractionStepError(
            f"incompatible description scene output at rows {invalid[:10]}: {path}; "
            "use --force or a new run_id"
        )
    return records


def minimal_graph_failures(
    failures: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    minimal = [
        {
            key: row[key]
            for key in (
                "scene_idx",
                "keyframes",
                "failure_kind",
                "error",
                "raw_response",
            )
            if key in row
        }
        for row in failures
    ]
    if minimal != failures:
        write_failure_jsonl(path, minimal)
    return minimal


def visual_rows(context: RunContext) -> list[dict[str, Any]]:
    catalog_path = require_file(context.cohort_dir / "catalog.jsonl", "cohort catalog")
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(catalog_path):
        content_id = str(item["content_id"])
        frames_dir = context.evidence_dir / "resized_keyframes" / content_id
        timestamp = (
            context.cohort_dir
            / "source_assets"
            / content_id
            / "assets"
            / "timestamp_fixed_30s.json"
        )
        frames = (
            sorted(
                path
                for path in frames_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
            if frames_dir.is_dir()
            else []
        )
        if not frames or not timestamp.is_file():
            raise ExtractionStepError(f"missing visual evidence for {content_id}")
        rows.append(
            {
                "content_id": content_id,
                "item_id": str(item["item_id"]),
                "frames_dir": str(frames_dir),
                "timestamp_json": str(timestamp),
            }
        )
    if not rows:
        raise ExtractionStepError(f"empty cohort catalog: {catalog_path}")
    return rows
