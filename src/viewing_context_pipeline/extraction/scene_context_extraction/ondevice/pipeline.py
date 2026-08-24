from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from viewing_context_pipeline.extraction.common.manifest import read_manifest_rows
from viewing_context_pipeline.extraction.data_preparation.raw_pipeline import (
    modality_dirname,
    multimodal_from_config,
    normalize_shot_interval,
    multimodal_ref_relative_path,
    resized_keyframes_relative_path,
    shot_interval_from_config,
    viewing_context_relative_path,
    timestamp_filename,
)
from ..graph_core.scene_failures import (
    failure_aggregation_warning,
    matching_scene_failures,
    read_scene_failures,
    replace_jsonl_atomic,
)
from ..graph_core.video_context import write_video_context
from ..graph_core.fingerprint import (
    build_input_fingerprint,
    fingerprint_matches,
    write_fingerprint,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
GRAPH_RECORD_FIELDS = {
    "scene_idx",
    "keyframes",
    "vlm_visual_graph",
}
LEGACY_GRAPH_RECORD_FIELDS = GRAPH_RECORD_FIELDS | {"raw_data", "vlm_visual_graph_warnings"}


@dataclass(frozen=True)
class SceneContextJob:
    content_id: str
    multimodal_ref: str
    scene_context_jsonl: str
    frames_dir: str
    timestamp_json: str
    metadata_json: str = ""
    video_id: str = ""
    shot_interval: str = "fixed_30s"
    multimodal: bool = False


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_metadata_and_scenes(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if records and records[0].get("_type") == "video_metadata":
        return records[0], records[1:]
    return {}, records


def scene_output_key(scene: dict[str, Any], fallback_idx: int) -> str:
    return str(scene.get("scene_idx", scene.get("scene_id", fallback_idx)))


def output_covers_scenes(
    output_path: str | Path,
    scenes: list[dict[str, Any]],
    *,
    content_id: str = "",
    failure_path: str | Path | None = None,
) -> bool:
    path = Path(output_path)
    if not path.exists() and not (failure_path and Path(failure_path).exists()):
        return False

    expected_scene_keys = {
        scene_output_key(scene, index)
        for index, scene in enumerate(scenes)
    }
    if not expected_scene_keys:
        return True

    try:
        records = read_jsonl(path) if path.exists() else []
        successful_records = successful_graph_records(records, scenes)
        failure_records = (
            matching_scene_failures(
                read_scene_failures(failure_path),
                content_id=content_id,
                expected_keyframes=expected_scene_keyframes(scenes),
            )
            if failure_path
            else []
        )
    except (OSError, json.JSONDecodeError):
        return False

    successful_keys = {
        str(record["scene_idx"]) for record in successful_records
    }
    failure_keys = {str(record["scene_idx"]) for record in failure_records}
    processed_scene_keys = successful_keys | failure_keys
    complete = (
        not successful_keys.intersection(failure_keys)
        and len(processed_scene_keys) == len(expected_scene_keys)
        and processed_scene_keys == expected_scene_keys
    )
    if complete:
        if successful_records != records:
            replace_jsonl_atomic(path, successful_records)
        if failure_path:
            replace_jsonl_atomic(failure_path, failure_records)
    return complete


def successful_graph_records(
    records: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {
        scene_output_key(scene, index): scene
        for index, scene in enumerate(scenes)
    }
    expected_keyframes = expected_scene_keyframes(scenes)
    successful: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        scene_key = str(record.get("scene_idx"))
        record_fields = set(record)
        graph = record.get("vlm_visual_graph")
        keyframes = record.get("keyframes")
        if (
            scene_key not in expected
            or (
                record_fields != GRAPH_RECORD_FIELDS
                and record_fields != LEGACY_GRAPH_RECORD_FIELDS
            )
            or not isinstance(keyframes, list)
            or not isinstance(graph, dict)
            or not graph
        ):
            continue
        source_keyframes = expected_keyframes.get(scene_key)
        if source_keyframes and source_keyframes != keyframes:
            continue
        successful[scene_key] = {
            "scene_idx": record["scene_idx"],
            "keyframes": list(keyframes),
            "vlm_visual_graph": graph,
        }
    return [
        successful[scene_output_key(scene, index)]
        for index, scene in enumerate(scenes)
        if scene_output_key(scene, index) in successful
    ]


def expected_scene_keyframes(
    scenes: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    expected: dict[str, list[Any]] = {}
    for index, scene in enumerate(scenes):
        timeline = scene.get("timeline")
        if isinstance(timeline, list) and timeline:
            keyframes = [
                item.get("timestamp")
                for item in timeline
                if isinstance(item, dict) and item.get("timestamp") is not None
            ]
        else:
            raw_keyframes = scene.get("keyframe_timestamps")
            keyframes = (
                list(raw_keyframes)
                if isinstance(raw_keyframes, list)
                else []
            )
        expected[scene_output_key(scene, index)] = keyframes
    return expected


def scene_context_job_video_id(job: SceneContextJob) -> str:
    return job.video_id or job.content_id


def required_scene_context_input_errors(job: SceneContextJob) -> list[str]:
    errors: list[str] = []
    if job.multimodal and not has_scene_context_source(job):
        errors.append(f"missing or empty multimodal_ref: {job.multimodal_ref}")

    frames_path = Path(job.frames_dir)
    if not frames_path.is_dir():
        errors.append(f"missing frames_dir: {job.frames_dir}")
    elif not any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        for child in frames_path.iterdir()
    ):
        errors.append(f"empty frames_dir: {job.frames_dir}")
    else:
        try:
            with Path(job.timestamp_json).open("r", encoding="utf-8") as f:
                timestamp_scenes = json.load(f)
            if not any(
                scene.get("keyframe_timestamps")
                for scene in timestamp_scenes
                if isinstance(scene, dict)
            ):
                errors.append(f"no keyframes in timestamp_json: {job.timestamp_json}")
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"missing or invalid timestamp_json: {job.timestamp_json}")
    return errors


def has_scene_context_source(job: SceneContextJob) -> bool:
    multimodal_ref_path = Path(job.multimodal_ref) if job.multimodal_ref else None
    if multimodal_ref_path and multimodal_ref_path.exists() and multimodal_ref_path.stat().st_size > 0:
        try:
            _, scenes = split_metadata_and_scenes(read_jsonl(multimodal_ref_path))
        except (OSError, json.JSONDecodeError):
            scenes = []
        if scenes:
            return True
    return False


def manifest_row_to_job(
    row: dict[str, object],
    shot_interval: str = "fixed_30s",
    multimodal: bool = False,
) -> SceneContextJob:
    mode = normalize_shot_interval(shot_interval)
    content_id = str(row.get("content_id") or "").strip()
    if not content_id:
        raise ValueError("manifest row is missing content_id")
    timestamp_json = str(assets_save_path() / content_id / "assets" / timestamp_filename(mode))

    scene_context_jsonl = infer_scene_context_jsonl(
        content_id,
        mode,
        multimodal,
    )
    return SceneContextJob(
        content_id=content_id,
        multimodal_ref=str(default_multimodal_ref_path(content_id, mode)),
        scene_context_jsonl=scene_context_jsonl,
        frames_dir=resolve_frames_dir(content_id, mode),
        timestamp_json=timestamp_json,
        metadata_json=str(default_metadata_path(content_id)),
        video_id=content_id,
        shot_interval=mode,
        multimodal=multimodal,
    )


def infer_scene_context_jsonl(
    content_id: str,
    shot_interval: str = "fixed_30s",
    multimodal: bool = False,
) -> str:
    return str(
        default_scene_context_path(
            content_id,
            shot_interval,
            multimodal,
        )
    )


def default_scene_context_path(
    content_id: str,
    shot_interval: str = "fixed_30s",
    multimodal: bool = False,
) -> Path:
    mode = normalize_shot_interval(shot_interval)
    return (
        output_save_path()
        / viewing_context_relative_path(multimodal, mode)
        / "scene_context_graph_qwen"
        / f"{content_id}_scene_context.jsonl"
    )


def default_multimodal_ref_path(
    content_id: str,
    shot_interval: str = "fixed_30s",
) -> Path:
    return (
        output_save_path()
        / multimodal_ref_relative_path(shot_interval)
        / f"{content_id}_multimodal_ref.jsonl"
    )


def default_metadata_path(content_id: str) -> Path:
    return output_save_path() / "data" / "cohort" / "metadata" / f"{content_id}.json"


def output_save_path() -> Path:
    value = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not value:
        raise ValueError("OUTPUT_SAVE_PATH is required by the pipeline runtime")
    return Path(value)


def assets_save_path() -> Path:
    value = os.getenv("LINUX_ASSETS_SAVE_PATH", "").strip()
    if not value:
        raise ValueError("LINUX_ASSETS_SAVE_PATH is required by the pipeline runtime")
    return Path(value)


def resolve_frames_dir(
    content_id: str,
    shot_interval: str = "fixed_30s",
) -> str:
    return str(
        output_save_path()
        / resized_keyframes_relative_path(shot_interval)
        / content_id
    )


def read_manifest(
    path: str | Path,
    shot_interval: str = "fixed_30s",
    multimodal: bool = False,
) -> list[SceneContextJob]:
    return [
        manifest_row_to_job(row, shot_interval, multimodal)
        for row in read_manifest_rows(path)
    ]


def write_scene_body_jsonl(scenes: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(json.dumps(scene, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_scene_context_source(job: SceneContextJob) -> list[dict[str, Any]]:
    if job.multimodal:
        multimodal_ref_path = Path(job.multimodal_ref) if job.multimodal_ref else None
        if multimodal_ref_path and multimodal_ref_path.is_file() and multimodal_ref_path.stat().st_size > 0:
            _, scenes = split_metadata_and_scenes(read_jsonl(multimodal_ref_path))
            if scenes:
                return scenes
            raise ValueError(f"{job.content_id}: multimodal_ref has no scene records")
        raise FileNotFoundError(f"{job.content_id}: multimodal extraction requires multimodal_ref")
    try:
        raw = json.loads(Path(job.timestamp_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{job.content_id}: invalid visual timestamp source") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{job.content_id}: visual timestamp source has no scenes")
    return [
        {
            "scene_idx": scene.get("scene_idx", index),
            "keyframe_timestamps": scene.get("keyframe_timestamps", []),
        }
        for index, scene in enumerate(raw)
        if isinstance(scene, dict)
    ]


def ondevice_failure_path(job: SceneContextJob) -> Path:
    scene_context_path = Path(job.scene_context_jsonl)
    postfix = ondevice_model_postfix(scene_context_path)
    failure_dir = f"scene_context_graph_{postfix or 'unknown'}"
    parts = scene_context_path.parts
    if "viewing_context" not in parts:
        return scene_context_path.parent / f"{job.content_id}_failures.jsonl"
    return (
        output_save_path()
        / "failures"
        / "viewing_context"
        / modality_dirname(job.multimodal)
        / job.shot_interval
        / failure_dir
        / f"{job.content_id}_failures.jsonl"
    )


def ondevice_output_root(scene_context_path: Path) -> Path:
    scene_context_dir = scene_context_path.parent.name
    if scene_context_dir == "scene_context_graph" or scene_context_dir.startswith("scene_context_graph_"):
        mode_root = scene_context_path.parent.parent
        if mode_root.name == "fixed_30s":
            return mode_root.parent
        return mode_root
    return scene_context_path.parent


def ondevice_model_postfix(scene_context_path: Path) -> str | None:
    scene_context_dir = scene_context_path.parent.name
    prefix = "scene_context_graph_"
    if not scene_context_dir.startswith(prefix):
        return None
    postfix = scene_context_dir.removeprefix(prefix)
    if postfix != "qwen":
        raise ValueError(f"unsupported on-device artifact postfix: {postfix!r}")
    return postfix


def load_ondevice_state(
    job: SceneContextJob,
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_path = Path(job.scene_context_jsonl)
    existing_records = read_jsonl(scene_path) if scene_path.exists() else []
    successful_records = successful_graph_records(existing_records, scenes)
    failure_records = matching_scene_failures(
        read_scene_failures(ondevice_failure_path(job)),
        content_id=job.content_id,
        expected_keyframes=expected_scene_keyframes(scenes),
    )
    return successful_records, failure_records


def state_covers_scenes(
    successful_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> bool:
    expected = {
        scene_output_key(scene, index)
        for index, scene in enumerate(scenes)
    }
    successful = {
        str(record["scene_idx"]) for record in successful_records
    }
    failed = {str(record["scene_idx"]) for record in failure_records}
    return (
        not successful.intersection(failed)
        and successful | failed == expected
    )


def pending_scenes_for_state(
    scenes: list[dict[str, Any]],
    successful_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    processed = {
        str(record["scene_idx"])
        for record in successful_records + failure_records
    }
    return [
        scene
        for index, scene in enumerate(scenes)
        if scene_output_key(scene, index) not in processed
    ]


def persist_ondevice_state(
    job: SceneContextJob,
    scenes: list[dict[str, Any]],
    successful_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_successes = successful_graph_records(successful_records, scenes)
    canonical_failures = matching_scene_failures(
        failure_records,
        content_id=job.content_id,
        expected_keyframes=expected_scene_keyframes(scenes),
    )
    replace_jsonl_atomic(job.scene_context_jsonl, canonical_successes)
    replace_jsonl_atomic(ondevice_failure_path(job), canonical_failures)
    return canonical_successes, canonical_failures


def run_scene_context_job(
    job: SceneContextJob,
    model: Any,
    processor: Any,
    vlm_config: dict[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    generation_callback: Callable[[int, float], None] | None = None,
) -> dict[str, int]:
    vlm_config = vlm_config or {}
    input_errors = required_scene_context_input_errors(job)
    if input_errors:
        raise FileNotFoundError(f"{job.content_id}: {'; '.join(input_errors)}")
    scenes = load_scene_context_source(job)
    output_path = Path(job.scene_context_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = build_input_fingerprint(
        content_id=job.content_id,
        scenes=scenes,
        frames_dir=job.frames_dir,
        multimodal=job.multimodal,
        backend=ondevice_model_postfix(output_path) or "ondevice",
        shot_interval=job.shot_interval,
        model_config=vlm_config,
    )
    if not force and not fingerprint_matches(output_path, fingerprint):
        output_path.unlink(missing_ok=True)
        ondevice_failure_path(job).unlink(missing_ok=True)
        ondevice_context_path(job).unlink(missing_ok=True)
        force = True
    successful_records, failure_records = (
        ([], []) if force else load_ondevice_state(job, scenes)
    )
    if state_covers_scenes(successful_records, failure_records, scenes):
        successful_records, failure_records = persist_ondevice_state(
            job,
            scenes,
            successful_records,
            failure_records,
        )
        write_ondevice_context(job, failure_records)
        write_fingerprint(output_path, fingerprint)
        return {
            "success": len(successful_records),
            "failed": len(failure_records),
            "warnings": 0,
        }

    pending_scenes = pending_scenes_for_state(
        scenes,
        successful_records,
        failure_records,
    )
    ondevice_context_path(job).unlink(missing_ok=True)
    from . import extractor as graph_extractor

    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        tmp_dir = Path(tmp)
        pending_output = tmp_dir / "scene_context.jsonl"
        pending_failures = tmp_dir / "failures.jsonl"
        extraction_kwargs = dict(
            model=model,
            processor=processor,
            frame_save_folder=job.frames_dir,
            scenes=pending_scenes,
            vlm_config=vlm_config,
            timestamp_json_path=job.timestamp_json,
            final_output_path=str(pending_output),
            content_id=job.content_id,
            failure_output_path=str(pending_failures),
        )
        if progress_callback is not None:
            extraction_kwargs["on_scene_complete"] = lambda: progress_callback(1)
        if generation_callback is not None:
            extraction_kwargs["on_generation_complete"] = generation_callback
        extraction_summary = graph_extractor.extract_visual_graphs(
            **extraction_kwargs,
        )
        new_successes = read_jsonl(pending_output)
        new_failures = read_scene_failures(pending_failures)
        successful_records, failure_records = persist_ondevice_state(
            job,
            scenes,
            successful_records + new_successes,
            failure_records + new_failures,
        )

    write_ondevice_context(job, failure_records)
    write_fingerprint(output_path, fingerprint)
    return {
        "success": len(successful_records),
        "failed": len(failure_records),
        "warnings": extraction_summary["warnings"],
    }


class SceneContextWorkerPool:
    def __init__(
        self,
        gpus: list[str],
        model_path: str,
        vlm_config: dict[str, Any],
    ) -> None:
        if not gpus:
            raise ValueError("at least one GPU is required")

        self.gpus = gpus
        self.context = mp.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.task_queues = []
        self.processes = []
        for worker_index, gpu_id in enumerate(gpus):
            task_queue = self.context.Queue()
            process = self.context.Process(
                target=_context_worker_main,
                args=(
                    worker_index,
                    gpu_id,
                    model_path,
                    vlm_config,
                    task_queue,
                    self.result_queue,
                ),
            )
            process.start()
            self.task_queues.append(task_queue)
            self.processes.append(process)

    def run_tasks(
        self,
        tasks: list[dict[str, Any]],
        on_task_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        pending: dict[str, int] = {}
        for task in tasks:
            task_id = str(task["task_id"])
            worker_index = int(task["worker_index"])
            pending[task_id] = worker_index
            self.task_queues[worker_index].put(task)

        results: dict[str, dict[str, Any]] = {}
        while pending:
            try:
                result = self.result_queue.get(timeout=5)
            except queue.Empty:
                dead_workers = {
                    worker_index
                    for worker_index in pending.values()
                    if not self.processes[worker_index].is_alive()
                }
                if dead_workers:
                    raise RuntimeError(f"VLM worker(s) exited before finishing tasks: {sorted(dead_workers)}")
                continue

            if not result.get("ok"):
                worker_index = result.get("worker_index", "unknown")
                gpu_id = result.get("gpu_id", "unknown")
                error = result.get("error", "unknown error")
                raise RuntimeError(f"VLM worker {worker_index} on GPU {gpu_id} failed:\n{error}")

            task_id = str(result.get("task_id"))
            if task_id in pending:
                results[task_id] = result
                del pending[task_id]
                if on_task_complete is not None:
                    on_task_complete(result)

        return [results[str(task["task_id"])] for task in tasks]

    def close(self) -> None:
        for task_queue in self.task_queues:
            task_queue.put(None)
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def __enter__(self) -> "SceneContextWorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_scene_context_job_parallel(
    job: SceneContextJob,
    worker_pool: SceneContextWorkerPool,
    vlm_config: dict[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    generation_callback: Callable[[int, float], None] | None = None,
) -> dict[str, int]:
    vlm_config = vlm_config or {}
    input_errors = required_scene_context_input_errors(job)
    if input_errors:
        raise FileNotFoundError(f"{job.content_id}: {'; '.join(input_errors)}")
    scenes = load_scene_context_source(job)
    output_path = Path(job.scene_context_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = build_input_fingerprint(
        content_id=job.content_id,
        scenes=scenes,
        frames_dir=job.frames_dir,
        multimodal=job.multimodal,
        backend=ondevice_model_postfix(output_path) or "ondevice",
        shot_interval=job.shot_interval,
        model_config=vlm_config,
    )
    if not force and not fingerprint_matches(output_path, fingerprint):
        output_path.unlink(missing_ok=True)
        ondevice_failure_path(job).unlink(missing_ok=True)
        ondevice_context_path(job).unlink(missing_ok=True)
        force = True

    successful_records, failure_records = (
        ([], []) if force else load_ondevice_state(job, scenes)
    )
    if state_covers_scenes(successful_records, failure_records, scenes):
        successful_records, failure_records = persist_ondevice_state(
            job,
            scenes,
            successful_records,
            failure_records,
        )
        write_ondevice_context(job, failure_records)
        write_fingerprint(output_path, fingerprint)
        return {
            "success": len(successful_records),
            "failed": len(failure_records),
            "warnings": 0,
        }

    pending_scenes = pending_scenes_for_state(
        scenes,
        successful_records,
        failure_records,
    )
    ondevice_context_path(job).unlink(missing_ok=True)
    indexed_pending = [
        (scene_order, scene)
        for scene_order, scene in enumerate(scenes)
        if scene in pending_scenes
    ]
    chunks = split_indexed_scene_chunks(
        indexed_pending,
        len(worker_pool.gpus),
    )
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        tmp_dir = Path(tmp)
        tasks = []
        for worker_index, chunk_scenes in enumerate(chunks):
            for scene_order, scene in chunk_scenes:
                input_jsonl = tmp_dir / f"{job.content_id}_scene_{scene_order}_body.jsonl"
                output_jsonl = tmp_dir / f"{job.content_id}_scene_{scene_order}_output.jsonl"
                failure_jsonl = tmp_dir / f"{job.content_id}_scene_{scene_order}_failures.jsonl"
                scene_record = dict(scene)
                scene_record.setdefault("scene_idx", scene_order)
                write_scene_body_jsonl([scene_record], input_jsonl)
                tasks.append(
                    {
                        "task_id": f"{job.content_id}:{scene_order}",
                        "worker_index": worker_index,
                        "scene_order": scene_order,
                        "frame_save_folder": job.frames_dir,
                        "input_json_path": str(input_jsonl),
                        "timestamp_json_path": job.timestamp_json,
                        "final_output_path": str(output_jsonl),
                        "failure_output_path": str(failure_jsonl),
                        "content_id": job.content_id,
                    }
                )

        tasks.sort(key=lambda item: int(item["scene_order"]))

        def task_complete(result: dict[str, Any]) -> None:
            task_summary = result.get("summary") or {}
            if generation_callback is not None:
                generation_callback(
                    int(task_summary.get("generated_tokens", 0)),
                    float(task_summary.get("generation_seconds", 0.0)),
                )
            if progress_callback is not None:
                progress_callback(1)

        if progress_callback is None and generation_callback is None:
            results = worker_pool.run_tasks(tasks)
        else:
            results = worker_pool.run_tasks(
                tasks,
                on_task_complete=task_complete,
            )
        merged_path = tmp_dir / f"{job.content_id}_merged_context.jsonl"
        merge_chunk_outputs(
            [Path(result["output_jsonl"]) for result in sorted(results, key=lambda item: item["scene_order"])],
            merged_path,
        )
        merged_failures_path = tmp_dir / f"{job.content_id}_merged_failures.jsonl"
        merge_chunk_outputs(
            [
                Path(result["failure_jsonl"])
                for result in sorted(
                    results,
                    key=lambda item: item["scene_order"],
                )
            ],
            merged_failures_path,
        )
        successful_records, failure_records = persist_ondevice_state(
            job,
            scenes,
            successful_records + read_jsonl(merged_path),
            failure_records + read_scene_failures(merged_failures_path),
        )

    write_ondevice_context(job, failure_records)
    write_fingerprint(output_path, fingerprint)
    task_summary = summarize_task_results(results)
    return {
        "success": len(successful_records),
        "failed": len(failure_records),
        "warnings": task_summary["warnings"],
    }


def write_ondevice_context(
    job: SceneContextJob,
    failure_records: list[dict[str, Any]],
) -> Path | None:
    output_path = ondevice_context_path(job)
    scene_context_path = Path(job.scene_context_jsonl)
    if not scene_context_path.exists() or not read_jsonl(scene_context_path):
        output_path.unlink(missing_ok=True)
        return None
    warning = failure_aggregation_warning(failure_records)
    write_video_context(
        job.content_id,
        job.scene_context_jsonl,
        output_path,
        extra_warnings=[warning] if warning else None,
    )
    return output_path


def ondevice_context_path(job: SceneContextJob) -> Path:
    scene_context_path = Path(job.scene_context_jsonl)
    postfix = ondevice_model_postfix(scene_context_path)
    profile_dir = f"video_context_graph_{postfix or 'unknown'}"
    return (
        scene_context_path.parent.parent
        / profile_dir
        / f"{job.content_id}_context_graph_ond.json"
    )


def split_indexed_scene_chunks(
    scenes: list[tuple[int, dict[str, Any]]],
    max_chunks: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if not scenes:
        return []
    chunk_count = min(len(scenes), max_chunks)
    chunks: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in range(chunk_count)
    ]
    for offset, scene in enumerate(scenes):
        chunks[offset % chunk_count].append(scene)
    return chunks


def summarize_task_results(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int((result.get("summary") or {}).get(key, 0)) for result in results)
        for key in ("failed", "warnings")
    }


def merge_chunk_outputs(chunk_paths: list[Path], output_path: str | Path) -> None:
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as out:
        for chunk_path in chunk_paths:
            with chunk_path.open("r", encoding="utf-8") as chunk:
                shutil.copyfileobj(chunk, out)


def _context_worker_main(
    worker_index: int,
    gpu_id: str,
    model_path: str,
    vlm_config: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    init_log = io.StringIO()
    try:
        with contextlib.redirect_stdout(init_log), contextlib.redirect_stderr(init_log):
            print(f"[INFO] Starting VLM worker {worker_index} on GPU {gpu_id}")
            model, processor = init_qwen3vl_model(model_path, use_fc_patch=True)
    except BaseException:
        result_queue.put(
            {
                "ok": False,
                "worker_index": worker_index,
                "gpu_id": gpu_id,
                "task_id": None,
                "error": init_log.getvalue() + traceback.format_exc(),
            }
        )
        return

    pending_init_log = init_log.getvalue()
    while True:
        task = task_queue.get()
        if task is None:
            return

        task_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(task_log), contextlib.redirect_stderr(task_log):
                from . import extractor as graph_extractor

                summary = graph_extractor.extract_visual_graphs(
                    model=model,
                    processor=processor,
                    frame_save_folder=task["frame_save_folder"],
                    scenes=read_jsonl(task["input_json_path"]),
                    vlm_config=vlm_config,
                    timestamp_json_path=task["timestamp_json_path"],
                    final_output_path=task["final_output_path"],
                    content_id=task["content_id"],
                    failure_output_path=task["failure_output_path"],
                )
            result_queue.put(
                {
                    "ok": True,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "task_id": task["task_id"],
                    "scene_order": task["scene_order"],
                    "output_jsonl": task["final_output_path"],
                    "failure_jsonl": task["failure_output_path"],
                    "summary": summary,
                }
            )
            pending_init_log = ""
        except BaseException:
            result_queue.put(
                {
                    "ok": False,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "task_id": task.get("task_id"),
                    "scene_order": task.get("scene_order"),
                    "log_scene_idx": task.get("log_scene_idx"),
                    "error": pending_init_log + task_log.getvalue() + traceback.format_exc(),
                }
            )
            pending_init_log = ""


def init_qwen3vl_model(model_path: str, use_fc_patch: bool = False):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path, device_map="cuda")

    if use_fc_patch:
        from .fc_patch import convert_to_fc_patch

        convert_to_fc_patch(model=model)
    return model, processor


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        raise ValueError("on-device context config path is required")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"on-device context config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["shot_interval"] = shot_interval_from_config(config)
    config["multimodal"] = multimodal_from_config(config)
    return config
