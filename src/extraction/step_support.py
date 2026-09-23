from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.errors import ExtractionStepError
from extraction.evidence import build_scene_evidence
from extraction.monitoring import video_names
from extraction.progress import InferenceProgress
from extraction.raw_output import is_raw_graph, valid_raw_graph
from extraction.scene_storage import write_scene_records
from pipeline_runtime import RunContext, read_jsonl


def write_progress(progress: tqdm | InferenceProgress, message: str) -> None:
    if isinstance(progress, InferenceProgress):
        progress.write_log(message)
    else:
        tqdm.write(message, file=progress.fp)


def complete_content_progress(progress: tqdm) -> None:
    progress.update(1)
    write_progress(progress, "")


def write_scene_results(scene_path: Path, records: list[dict[str, Any]]) -> None:
    """Publish completed scene results without a recovery journal."""
    records.sort(key=lambda row: int(row["scene_idx"]))
    write_scene_records(scene_path, records)


def video_name_map(context: RunContext) -> dict[str, str]:
    return video_names(read_jsonl(context.cohort_dir / "catalog.jsonl"))


def scene_generation_rows(
    visual: dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    scenes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if scenes is None:
        scenes = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in build_scene_evidence(
        scenes, visual["frames_dir"], visual["timestamp_json"], prepared=True
    ):
        scene_idx = scene["scene_idx"]
        keyframes = scene["keyframes"]
        image_paths = scene["image_paths"]
        if not keyframes or len(image_paths) != len(keyframes):
            raise ExtractionStepError(
                f"{visual['content_id']} scene {scene_idx} has "
                f"{len(image_paths)} of {len(keyframes)} keyframes"
            )
        task_id = f"{visual['content_id']}:{scene_idx}"
        rows.append(
            {
                "task": QwenGenerationTask(
                    task_id=task_id,
                    image_paths=tuple(image_paths),
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
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
) -> dict[str, Any]:
    return {
        "stage": stage,
        "content_count": content_count,
        "failure_count": failure_count,
    }


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
    invalid = [index for index, row in enumerate(records)
               if (not valid_raw_graph(row) if is_raw_graph(row) else set(row) - {"provenance", "generation", "tokens"} != required)]
    if invalid:
        raise ExtractionStepError(
            f"incompatible graph scene output at rows {invalid[:10]}: {path}; "
            "use --force or a new run_id"
        )
    from extraction.structured_output import validate_graph_structure, OutputValidationError
    for row in records:
        if not is_raw_graph(row) and not (row.get("parse_mode") == "text" and isinstance(row.get("graph"), str)):
            try:
                validate_graph_structure(row["graph"])
            except OutputValidationError as exc:
                raise ExtractionStepError(f"invalid TOBE graph: {path}: {exc}") from exc
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
    invalid = [index for index, row in enumerate(records) if set(row) - {"provenance", "generation", "status", "tokens"} != required
               or row.get("status", "raw_fallback") != "raw_fallback"]
    if invalid:
        raise ExtractionStepError(
            f"incompatible description scene output at rows {invalid[:10]}: {path}; "
            "use --force or a new run_id"
        )
    from extraction.descriptions import SCENE_SCHEMA_VERSION
    if any(row.get("schema_version") != SCENE_SCHEMA_VERSION
           or row.get("content_id") != path.stem
           or not isinstance(row.get("description"), str) or not row["description"].strip()
           for row in records):
        raise ExtractionStepError(f"invalid description scene: {path}")
    return records


def minimal_graph_failures(
    failures: list[dict[str, Any]],
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
                "response_diagnostics",
            )
            if key in row
        }
        for row in failures
    ]
    return minimal


def visual_rows(context: RunContext) -> list[dict[str, Any]]:
    cohort = context.require_ready_cohort()
    from visual_sampling import timestamp_filename
    sampling = context.config["extraction"]["visual_evidence"]
    rows: list[dict[str, Any]] = []
    for item in cohort["catalog"]:
        content_id = str(item["content_id"])
        frames_dir = context.keyframes_dir / content_id
        timestamp = (
            context.source_assets_dir
            / content_id
            / timestamp_filename(sampling["scene_duration"], sampling["num_keyframes"])
        )
        rows.append(
            {
                "content_id": content_id,
                "item_id": str(item["item_id"]),
                "frames_dir": str(frames_dir),
                "timestamp_json": str(timestamp),
            }
        )
    if not rows:
        raise ExtractionStepError(f"empty cohort catalog: {context.cohort_dir}")
    return rows
