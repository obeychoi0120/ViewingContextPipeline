from __future__ import annotations

import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from google.genai import types
from tqdm import tqdm

from viewing_context_pipeline.extraction.common.gemini import make_extraction_config, parse_json_response
from viewing_context_pipeline.extraction.common.local_images import local_image_part
from ..graph_core.prompt import SCENE_EXTRACTION_PROMPT, USER_MESSAGE
from ..graph_core.multimodal import (
    MULTIMODAL_USER_MESSAGE,
    shot_reference_text,
    shot_references,
    validate_image_reference_alignment,
)
from ..graph_core.scene_failures import (
    build_scene_failure_record,
    failure_aggregation_warning,
    matching_scene_failures,
    read_scene_failures,
    replace_jsonl_atomic,
)
from ..graph_core.validator import compact_observation, validate_observation
from ..graph_core.video_context import video_context_is_valid, write_video_context
from ..graph_core.fingerprint import (
    build_input_fingerprint,
    fingerprint_matches,
    write_fingerprint,
)
from viewing_context_pipeline.extraction.data_preparation.raw_pipeline import (
    normalize_shot_interval,
    viewing_context_relative_path,
)


SCENE_CONTEXT_INPUT_SUFFIX = "_multimodal_ref.jsonl"
SCENE_EXTRACTION_WORKERS = 8
GRAPH_RECORD_FIELDS = {
    "scene_idx",
    "keyframes",
    "vlm_visual_graph",
    "vlm_visual_graph_warnings",
}
LEGACY_GRAPH_RECORD_FIELDS = GRAPH_RECORD_FIELDS | {"raw_data"}


@dataclass(frozen=True)
class SceneContextGeminiConfig:
    gcp_project_id: str
    output_dir: str = "output"
    location: str = "global"
    model: str = "gemini-3.5-flash"
    thinking_level: str = "medium"
    shot_interval: str = "fixed_30s"
    multimodal: bool = False


def build_extraction_config(thinking_level: str, multimodal: bool = False):
    system_instruction = SCENE_EXTRACTION_PROMPT
    if multimodal:
        system_instruction += "\n\n" + MULTIMODAL_USER_MESSAGE
    return make_extraction_config(
        system_instruction=system_instruction,
        thinking_level=thinking_level,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )


def load_scene_context_rows(
    multimodal_ref_dir: str | Path,
    content_ids: Iterable[str] | None = None,
    *,
    multimodal: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    input_dir = Path(multimodal_ref_dir)
    if not input_dir.is_dir():
        raise ValueError(f"multimodal_ref directory not found: {input_dir}")
    if content_ids is None:
        multimodal_ref_paths = sorted(
            path
            for path in input_dir.glob(f"*{SCENE_CONTEXT_INPUT_SUFFIX}")
            if path.is_file() and path.name != SCENE_CONTEXT_INPUT_SUFFIX
        )
    else:
        multimodal_ref_paths = [
            input_dir / f"{content_id}{SCENE_CONTEXT_INPUT_SUFFIX}"
            for content_id in content_ids
        ]
        multimodal_ref_paths = [
            path
            for path in multimodal_ref_paths
            if path.is_file()
        ]
    if not multimodal_ref_paths:
        qualifier = "manifest-listed " if content_ids is not None else ""
        raise ValueError(
            f"no {qualifier}*{SCENE_CONTEXT_INPUT_SUFFIX} files found in {input_dir}"
        )

    rows: list[dict[str, Any]] = []
    for multimodal_ref_path in multimodal_ref_paths:
        content_id = multimodal_ref_path.name[: -len(SCENE_CONTEXT_INPUT_SUFFIX)]
        source_uri = str(multimodal_ref_path)
        records = parse_multimodal_ref(multimodal_ref_path.read_text(encoding="utf-8"), source_uri)
        seen_scene_idxs: set[int] = set()
        content_rows: list[dict[str, Any]] = []
        for line_number, record in records:
            if record.get("_type") == "video_metadata":
                continue
            scene = multimodal_scene_to_context(
                record,
                source_uri,
                line_number,
                multimodal=multimodal,
            )
            scene_idx = scene["scene_idx"]
            if scene_idx in seen_scene_idxs:
                raise ValueError(f"{source_uri}: duplicate scene_idx {scene_idx}")
            seen_scene_idxs.add(scene_idx)
            row = {
                "content_id": content_id,
                "scene_idx": scene_idx,
                "keyframes": scene["keyframes"],
            }
            if multimodal:
                row["timeline"] = scene["timeline"]
            content_rows.append(row)
        if not content_rows:
            raise ValueError(f"{source_uri}: multimodal_ref contains no scene records")
        rows.extend(sorted(content_rows, key=lambda row: row["scene_idx"]))
    return rows, len(multimodal_ref_paths)


def parse_multimodal_ref(text: str, source_uri: str) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source_uri}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{source_uri}:{line_number}: scene record must be an object")
        records.append((line_number, record))
    return records


def multimodal_scene_to_context(
    record: dict[str, Any],
    source_uri: str,
    line_number: int,
    *,
    multimodal: bool = False,
) -> dict[str, Any]:
    location = f"{source_uri}:{line_number}"
    scene_idx = record.get("scene_idx")
    if type(scene_idx) is not int or scene_idx < 0:
        raise ValueError(f"{location}: scene_idx must be a non-negative integer")

    timeline = record.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError(f"{location}: timeline must be a non-empty list")

    keyframes: list[int] = []
    for item_index, item in enumerate(timeline):
        item_location = f"{location}: timeline[{item_index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_location} must be an object")
        timestamp = item.get("timestamp")
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError(f"{item_location}.timestamp must be a non-negative integer")
        keyframes.append(timestamp)

    if any(current >= following for current, following in zip(keyframes, keyframes[1:])):
        raise ValueError(f"{location}: timeline timestamps must be strictly increasing")

    scene = {
        "scene_idx": scene_idx,
        "keyframes": keyframes,
    }
    if multimodal:
        shot_references(record)
        scene["timeline"] = timeline
    return scene


def group_scene_context_rows_by_content(
    scene_context_rows: list[dict[str, Any]],
) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in scene_context_rows:
        content_id = str(row.get("content_id") or "").strip()
        if not content_id:
            raise ValueError("context row is missing content_id")
        grouped.setdefault(content_id, []).append(row)
    return grouped


def extract_scene_context_gemini(
    client: Any,
    model_name: str,
    config: Any,
    scene: dict[str, Any],
    image_paths: list[str],
    multimodal: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    if not image_paths:
        raise ValueError("no keyframe images found for scene")

    contents = build_gemini_contents(
        scene,
        image_paths,
        multimodal=multimodal,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    parsed = parse_json_response(response.text)
    target_graph, warnings = validate_observation(parsed)
    return compact_observation(target_graph), warnings


def build_gemini_contents(
    scene: dict[str, Any],
    image_paths: list[str],
    *,
    multimodal: bool = False,
) -> list[Any]:
    if not multimodal:
        return [*[image_part(path) for path in image_paths], USER_MESSAGE]
    references = shot_references(scene)
    validate_image_reference_alignment(len(image_paths), references)
    parts: list[Any] = []
    for path, reference in zip(image_paths, references):
        parts.append(image_part(path))
        parts.append(types.Part.from_text(text=shot_reference_text(reference)))
    parts.append(MULTIMODAL_USER_MESSAGE)
    return parts


def image_part(path: str | Path) -> types.Part:
    return local_image_part(path)


def build_user_text(scene: dict[str, Any]) -> str:
    return USER_MESSAGE


def build_training_messages(image_paths: list[str], target_graph: dict[str, Any], scene: dict[str, Any]) -> list[dict[str, Any]]:
    assistant_content = json.dumps(target_graph, ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": SCENE_EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image_path", "image_path": path} for path in image_paths]
            + [{"type": "text", "text": build_user_text(scene)}],
        },
        {"role": "assistant", "content": assistant_content},
    ]


def visual_graph_record(
    scene: dict[str, Any],
    target_graph: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "scene_idx": scene["scene_idx"],
        "keyframes": list(scene["keyframes"]),
        "vlm_visual_graph": target_graph,
        "vlm_visual_graph_warnings": warnings,
    }


def training_output_record(
    content_id: str,
    scene: dict[str, Any],
    image_paths: list[str],
    target_graph: dict[str, Any] | None,
    model_name: str,
    error: str = "",
) -> dict[str, Any]:
    messages = build_training_messages(image_paths, target_graph, scene) if target_graph is not None else []
    return {
        "content_id": content_id,
        "scene_idx": scene.get("scene_idx", ""),
        "image_paths": image_paths,
        "target_graph": target_graph,
        "messages": messages,
        "gemini_model": model_name,
        "gemini_error": error,
    }


def read_existing_outputs(path: str | Path) -> list[dict[str, Any]]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def successful_scene_records(
    records: list[dict[str, Any]],
    scene_context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {row["scene_idx"]: row for row in scene_context_rows}
    successful: dict[int, dict[str, Any]] = {}
    for record in records:
        scene_idx = record.get("scene_idx")
        source = expected.get(scene_idx)
        record_fields = set(record)
        if source is None or (
            record_fields != GRAPH_RECORD_FIELDS
            and record_fields != LEGACY_GRAPH_RECORD_FIELDS
        ):
            continue
        if record.get("keyframes") != source["keyframes"]:
            continue
        if (
            isinstance(record.get("vlm_visual_graph"), dict)
            and record["vlm_visual_graph"]
            and isinstance(
            record.get("vlm_visual_graph_warnings"), list
            )
        ):
            successful[scene_idx] = {
                "scene_idx": scene_idx,
                "keyframes": list(record["keyframes"]),
                "vlm_visual_graph": record["vlm_visual_graph"],
                "vlm_visual_graph_warnings": list(record["vlm_visual_graph_warnings"]),
            }
    return [successful[row["scene_idx"]] for row in scene_context_rows if row["scene_idx"] in successful]


def replace_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    replace_jsonl_atomic(path, records)


def append_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def gemini_failure_path(
    scene_context_path: str | Path,
    content_id: str,
) -> Path:
    scene_path = Path(scene_context_path)
    if scene_path.parent.name == "scene_context_graph_gemini":
        mode_root = scene_path.parent.parent
        modality_root = mode_root.parent
        output_root = modality_root.parent.parent
        return (
            output_root
            / "failures"
            / "viewing_context"
            / modality_root.name
            / mode_root.name
            / "scene_context_graph_gemini"
            / f"{content_id}_failures.jsonl"
        )
    return scene_path.parent / f"{content_id}_failures.jsonl"


def expected_failure_keyframes(
    rows: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    return {
        str(row["scene_idx"]): list(row["keyframes"])
        for row in rows
    }


def load_gemini_state(
    *,
    content_id: str,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    failure_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = successful_scene_records(
        read_existing_outputs(output_path),
        rows,
    )
    failures = matching_scene_failures(
        read_scene_failures(failure_path),
        content_id=content_id,
        expected_keyframes=expected_failure_keyframes(rows),
    )
    return successful, failures


def persist_gemini_state(
    *,
    content_id: str,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    failure_path: str | Path,
    successful_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = successful_scene_records(successful_records, rows)
    failures = matching_scene_failures(
        failure_records,
        content_id=content_id,
        expected_keyframes=expected_failure_keyframes(rows),
    )
    replace_jsonl(output_path, successful)
    replace_jsonl_atomic(failure_path, failures)
    return successful, failures


def gemini_state_complete(
    rows: list[dict[str, Any]],
    successful_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> bool:
    expected = {int(row["scene_idx"]) for row in rows}
    successful = {
        int(record["scene_idx"]) for record in successful_records
    }
    failed = {int(record["scene_idx"]) for record in failure_records}
    return not successful.intersection(failed) and successful | failed == expected


def write_gemini_context(
    *,
    content_id: str,
    scene_context_path: str | Path,
    context_path: str | Path,
    successful_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> bool:
    output_context_path = Path(context_path)
    if not successful_records:
        output_context_path.unlink(missing_ok=True)
        return False
    warning = failure_aggregation_warning(failure_records)
    extra_warnings = [warning] if warning else None
    if video_context_is_valid(
        output_context_path,
        content_id=content_id,
        source_scene_context_path=scene_context_path,
        extra_warnings=extra_warnings,
    ):
        return False
    write_video_context(
        content_id,
        scene_context_path,
        output_context_path,
        extra_warnings=extra_warnings,
    )
    return True


def extract_scene_contexts_gemini(
    scene_context_rows: list[dict[str, Any]],
    output_path: str | Path,
    client: Any,
    config: SceneContextGeminiConfig,
    extraction_config: Any,
    sleep_sec: float = 0.5,
    resume: bool = True,
    max_scenes: int | None = None,
    output_dir: str | Path | None = None,
    training_output_path: str | Path | None = None,
    video_context_output_dir: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    extracted_count = 0
    summary = {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "video_contexts_written": 0,
    }
    grouped_rows = group_scene_context_rows_by_content(scene_context_rows)

    with tqdm(total=len(grouped_rows), desc="Contents", unit="content", disable=not show_progress) as progress:
        for content_id, rows in grouped_rows.items():
            if max_scenes is not None and extracted_count >= max_scenes:
                tqdm.write(f"[done] max_scenes reached: {max_scenes}")
                break

            current_output_path = output_path_for_content(output_path=output_path, output_dir=output_dir, content_id=content_id)
            current_failure_path = gemini_failure_path(
                current_output_path,
                content_id,
            )
            current_context_path = context_path_for_content(
                video_context_output_dir,
                content_id,
                current_output_path,
            )
            fingerprint = build_input_fingerprint(
                content_id=content_id,
                scenes=rows,
                frames_dir=(
                    Path(config.output_dir)
                    / "data"
                    / config.shot_interval
                    / "resized_keyframes"
                    / content_id
                ),
                multimodal=config.multimodal,
                backend="gemini",
                shot_interval=config.shot_interval,
                model_config={
                    "model": config.model,
                    "location": config.location,
                    "thinking_level": config.thinking_level,
                },
            )
            if resume and not fingerprint_matches(current_output_path, fingerprint):
                current_output_path.unlink(missing_ok=True)
                current_failure_path.unlink(missing_ok=True)
                current_context_path.unlink(missing_ok=True)
            if not resume:
                current_output_path.unlink(missing_ok=True)
                current_failure_path.unlink(missing_ok=True)
                current_context_path.unlink(missing_ok=True)

            successful_records, failure_records = load_gemini_state(
                content_id=content_id,
                rows=rows,
                output_path=current_output_path,
                failure_path=current_failure_path,
            )
            processed = {
                int(record["scene_idx"])
                for record in successful_records + failure_records
            }
            if gemini_state_complete(
                rows,
                successful_records,
                failure_records,
            ):
                successful_records, failure_records = persist_gemini_state(
                    content_id=content_id,
                    rows=rows,
                    output_path=current_output_path,
                    failure_path=current_failure_path,
                    successful_records=successful_records,
                    failure_records=failure_records,
                )
                summary["skipped"] += len(rows)
                summary["failed"] += len(failure_records)
                if write_gemini_context(
                    content_id=content_id,
                    scene_context_path=current_output_path,
                    context_path=current_context_path,
                    successful_records=successful_records,
                    failure_records=failure_records,
                ):
                    summary["video_contexts_written"] += 1
                    tqdm.write(
                        f"[profile] {content_id}: rebuilt from existing scene context"
                    )
                else:
                    tqdm.write(
                        f"[skip] {content_id}: existing Gemini state is terminal"
                    )
                write_fingerprint(current_output_path, fingerprint)
                progress.update(1)
                continue

            current_context_path.unlink(missing_ok=True)
            if show_progress:
                if processed:
                    tqdm.write(f"[resume] {content_id}: {len(processed)}/{len(rows)} scene(s) already completed")
                else:
                    tqdm.write(f"[start] {content_id}: processing {len(rows)} scene(s)")

            pending_rows = [row for row in rows if row["scene_idx"] not in processed]
            summary["skipped"] += len(rows) - len(pending_rows)
            if max_scenes is not None:
                remaining = max_scenes - extracted_count
                if len(pending_rows) > remaining:
                    pending_rows = pending_rows[:remaining]
            if not pending_rows:
                progress.update(1)
                break

            missing_by_scene = missing_keyframe_paths_by_scene(
                config.output_dir,
                pending_rows,
                config.shot_interval,
            )
            new_successes: list[dict[str, Any]] = []
            new_failures: list[dict[str, Any]] = []
            training_records: list[dict[str, Any]] = []
            extractable_rows: list[dict[str, Any]] = []
            for row in pending_rows:
                scene_idx = int(row["scene_idx"])
                missing_paths = missing_by_scene.get(scene_idx, [])
                if not missing_paths:
                    extractable_rows.append(row)
                    continue
                error = (
                    f"missing {len(missing_paths)} local keyframe file(s): "
                    + ", ".join(missing_paths)
                )
                new_failures.append(
                    build_scene_failure_record(
                        content_id=content_id,
                        scene_idx=scene_idx,
                        keyframes=row["keyframes"],
                        error=error,
                    )
                )
                if training_output_path is not None:
                    scene = scene_from_context_row(row)
                    training_records.append(
                        training_output_record(
                            content_id,
                            scene,
                            scene_context_image_paths(row, config.output_dir, config.shot_interval),
                            None,
                            config.model,
                            error=error,
                        )
                    )

            with ThreadPoolExecutor(max_workers=SCENE_EXTRACTION_WORKERS) as executor:
                futures = [
                    executor.submit(
                        extract_scene_context_row,
                        client=client,
                        config=config,
                        extraction_config=extraction_config,
                        content_id=content_id,
                        row=row,
                        sleep_sec=sleep_sec,
                        include_training_record=training_output_path is not None,
                    )
                    for row in extractable_rows
                ]
                for future in futures:
                    record, failure_record, training_record = future.result()
                    if record is not None:
                        new_successes.append(record)
                    if failure_record is not None:
                        new_failures.append(failure_record)
                    if training_record is not None:
                        training_records.append(training_record)

            extracted_count += len(pending_rows)
            summary["processed"] += len(pending_rows)
            summary["succeeded"] += len(new_successes)
            successful_records, failure_records = persist_gemini_state(
                content_id=content_id,
                rows=rows,
                output_path=current_output_path,
                failure_path=current_failure_path,
                successful_records=successful_records + new_successes,
                failure_records=failure_records + new_failures,
            )
            summary["failed"] += len(failure_records)
            if training_output_path is not None:
                append_jsonl_records(training_output_path, training_records)

            if failure_records:
                tqdm.write(
                    f"[error] {content_id}: "
                    f"{len(failure_records)} terminal scene failure(s)"
                )

            if gemini_state_complete(
                rows,
                successful_records,
                failure_records,
            ) and write_gemini_context(
                content_id=content_id,
                scene_context_path=current_output_path,
                context_path=current_context_path,
                successful_records=successful_records,
                failure_records=failure_records,
            ):
                summary["video_contexts_written"] += 1
            write_fingerprint(current_output_path, fingerprint)

            progress.update(1)

            if max_scenes is not None and extracted_count >= max_scenes:
                tqdm.write(f"[done] max_scenes reached: {max_scenes}")
                break

    tqdm.write(
        "[done] "
        f"processed={summary['processed']} "
        f"succeeded={summary['succeeded']} "
        f"failed={summary['failed']} "
        f"skipped={summary['skipped']} "
        f"video_contexts_written={summary['video_contexts_written']}"
    )
    return summary


def extract_scene_context_row(
    *,
    client: Any,
    config: SceneContextGeminiConfig,
    extraction_config: Any,
    content_id: str,
    row: dict[str, Any],
    sleep_sec: float,
    include_training_record: bool,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    scene = scene_from_context_row(row)
    image_paths = scene_context_image_paths(
        row,
        config.output_dir,
        config.shot_interval,
    )
    try:
        target_graph, warnings = extract_scene_context_gemini(
            client=client,
            model_name=config.model,
            config=extraction_config,
            scene=scene,
            image_paths=image_paths,
            multimodal=config.multimodal,
        )
        record = visual_graph_record(scene, target_graph, warnings)
        training_record = (
            training_output_record(content_id, scene, image_paths, target_graph, config.model)
            if include_training_record
            else None
        )
        return record, None, training_record
    except Exception as exc:
        error = str(exc)
        failure_record = build_scene_failure_record(
            content_id=content_id,
            scene_idx=scene["scene_idx"],
            keyframes=scene["keyframes"],
            error=error,
        )
        training_record = (
            training_output_record(content_id, scene, image_paths, None, config.model, error=error)
            if include_training_record
            else None
        )
        return None, failure_record, training_record
    finally:
        if sleep_sec > 0:
            time.sleep(sleep_sec)


def scene_from_context_row(row: dict[str, Any]) -> dict[str, Any]:
    scene = {
        "scene_idx": row["scene_idx"],
        "keyframes": list(row["keyframes"]),
    }
    if "timeline" in row:
        scene["timeline"] = list(row["timeline"])
    return scene


def scene_context_image_paths(
    row: dict[str, Any],
    output_dir: str | Path,
    shot_interval: str = "fixed_30s",
) -> list[str]:
    frame_dir = (
        Path(output_dir)
        / "data"
        / normalize_shot_interval(shot_interval)
        / "resized_keyframes"
        / row["content_id"]
    )
    return [
        str(frame_dir / f"{timestamp:04d}.png")
        for timestamp in row["keyframes"]
    ]


def missing_keyframe_paths(
    output_dir: str | Path,
    scene_context_rows: list[dict[str, Any]],
    shot_interval: str = "fixed_30s",
) -> list[str]:
    expected_paths = [
        path
        for row in scene_context_rows
        for path in scene_context_image_paths(row, output_dir, shot_interval)
    ]
    return sorted(path for path in set(expected_paths) if not Path(path).is_file())


def missing_keyframe_paths_by_scene(
    output_dir: str | Path,
    scene_context_rows: list[dict[str, Any]],
    shot_interval: str = "fixed_30s",
) -> dict[int, list[str]]:
    missing: dict[int, list[str]] = {}
    for row in scene_context_rows:
        row_missing = sorted(
            path
            for path in scene_context_image_paths(
                row,
                output_dir,
                shot_interval,
            )
            if not Path(path).is_file()
        )
        if row_missing:
            missing[int(row["scene_idx"])] = row_missing
    return missing


def output_path_for_content(output_path: str | Path, output_dir: str | Path | None, content_id: str) -> Path:
    if output_dir:
        return Path(output_dir) / f"{content_id}_scene_context_gemini.jsonl"
    return Path(output_path)


def scene_context_gemini_output_dir(
    output_dir: str | Path,
    shot_interval: str = "fixed_30s",
    multimodal: bool = False,
) -> Path:
    return (
        Path(output_dir)
        / viewing_context_relative_path(multimodal, shot_interval)
        / "scene_context_graph_gemini"
    )


def video_context_graph_gemini_output_dir(
    output_dir: str | Path,
    shot_interval: str = "fixed_30s",
    multimodal: bool = False,
) -> Path:
    return (
        Path(output_dir)
        / viewing_context_relative_path(multimodal, shot_interval)
        / "video_context_graph_gemini"
    )


def context_path_for_content(
    output_dir: str | Path | None,
    content_id: str,
    scene_context_path: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        profile_dir = Path(output_dir)
    elif scene_context_path is not None:
        scene_dir = Path(scene_context_path).parent
        output_root = (
            scene_dir.parent
            if scene_dir.name == "scene_context_graph_gemini"
            else scene_dir
        )
        profile_dir = output_root / "video_context_graph_gemini"
    else:
        raise ValueError("video_context_output_dir or scene_context_path is required")
    return profile_dir / f"{content_id}_context_graph_gemini.json"
