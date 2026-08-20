from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from src.common.manifest import read_manifest_rows
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
from ..ondevice.pipeline import (
    SceneContextJob,
    expected_scene_keyframes,
    load_config,
    load_scene_context_source,
    manifest_row_to_job as manifest_row_to_ondevice_job,
    merge_chunk_outputs,
    output_save_path,
    pending_scenes_for_state,
    read_jsonl,
    required_scene_context_input_errors,
    scene_context_job_video_id,
    split_indexed_scene_chunks,
    state_covers_scenes,
    successful_graph_records,
    summarize_task_results,
    write_scene_body_jsonl,
)
from src.video_data_collection.raw_pipeline import normalize_shot_interval
from src.video_data_collection.raw_pipeline import (
    modality_dirname,
    viewing_context_relative_path,
)

from .api import GaussApiClient
from .extractor import extract_visual_graphs


GAUSS_ARTIFACT_POSTFIX = "gaussa_gemma4_e2b_v0_3"


def manifest_row_to_job(
    row: dict[str, object],
    shot_interval: str = "fixed_15s",
    multimodal: bool = False,
) -> SceneContextJob:
    mode = normalize_shot_interval(shot_interval)
    base_job = manifest_row_to_ondevice_job(
        row,
        shot_interval=mode,
        multimodal=multimodal,
    )
    return replace(
        base_job,
        scene_context_jsonl=str(
            gauss_scene_context_path(base_job.content_id, mode, multimodal)
        ),
    )


def read_manifest(
    path: str | Path,
    shot_interval: str = "fixed_15s",
    multimodal: bool = False,
) -> list[SceneContextJob]:
    return [
        manifest_row_to_job(row, shot_interval, multimodal)
        for row in read_manifest_rows(path)
    ]


def gauss_scene_context_path(
    content_id: str,
    shot_interval: str = "fixed_15s",
    multimodal: bool = False,
) -> Path:
    return (
        output_save_path()
        / viewing_context_relative_path(multimodal, shot_interval)
        / f"scene_context_graph_{GAUSS_ARTIFACT_POSTFIX}"
        / f"{content_id}_scene_context.jsonl"
    )


def gauss_failure_path(job: SceneContextJob) -> Path:
    return (
        output_save_path()
        / "failures"
        / "viewing_context"
        / modality_dirname(job.multimodal)
        / normalize_shot_interval(job.shot_interval)
        / f"scene_context_graph_{GAUSS_ARTIFACT_POSTFIX}"
        / f"{job.content_id}_failures.jsonl"
    )


def gauss_context_path(job: SceneContextJob) -> Path:
    return (
        output_save_path()
        / viewing_context_relative_path(job.multimodal, job.shot_interval)
        / f"video_context_graph_{GAUSS_ARTIFACT_POSTFIX}"
        / f"{job.content_id}_context_graph_ond.json"
    )


def load_gauss_state(
    job: SceneContextJob,
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_path = Path(job.scene_context_jsonl)
    existing_records = read_jsonl(scene_path) if scene_path.exists() else []
    successful_records = successful_graph_records(existing_records, scenes)
    failure_records = matching_scene_failures(
        read_scene_failures(gauss_failure_path(job)),
        content_id=job.content_id,
        expected_keyframes=expected_scene_keyframes(scenes),
    )
    return successful_records, failure_records


def persist_gauss_state(
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
    replace_jsonl_atomic(gauss_failure_path(job), canonical_failures)
    return canonical_successes, canonical_failures


class GaussApiWorkerPool:
    def __init__(
        self,
        client: GaussApiClient,
        worker_count: int,
        config: dict[str, Any],
    ) -> None:
        if type(worker_count) is not int or worker_count <= 0:
            raise ValueError("API worker count must be a positive integer")
        self.client = client
        self.worker_count = worker_count
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=worker_count)

    def run_tasks(
        self,
        tasks: list[dict[str, Any]],
        on_task_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        futures = {
            self.executor.submit(
                _run_gauss_api_task,
                task,
                self.client,
                self.config,
            ): task
            for task in tasks
        }
        results: dict[str, dict[str, Any]] = {}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Gauss API worker failed for task {task['task_id']}: {exc}"
                ) from exc
            results[str(task["task_id"])] = result
            if on_task_complete is not None:
                on_task_complete(result)
        return [results[str(task["task_id"])] for task in tasks]

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "GaussApiWorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_scene_context_job(
    job: SceneContextJob,
    worker_pool: GaussApiWorkerPool,
    config: dict[str, Any] | None = None,
    force: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    generation_callback: Callable[[int, float], None] | None = None,
) -> dict[str, int]:
    config = config or {}
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
        backend=GAUSS_ARTIFACT_POSTFIX,
        shot_interval=job.shot_interval,
        model_config=config,
    )
    if not force and not fingerprint_matches(output_path, fingerprint):
        output_path.unlink(missing_ok=True)
        gauss_failure_path(job).unlink(missing_ok=True)
        gauss_context_path(job).unlink(missing_ok=True)
        force = True
    successful_records, failure_records = (
        ([], []) if force else load_gauss_state(job, scenes)
    )
    if state_covers_scenes(successful_records, failure_records, scenes):
        successful_records, failure_records = persist_gauss_state(
            job,
            scenes,
            successful_records,
            failure_records,
        )
        write_gauss_context(job, failure_records)
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
    gauss_context_path(job).unlink(missing_ok=True)
    indexed_pending = [
        (scene_order, scene)
        for scene_order, scene in enumerate(scenes)
        if scene in pending_scenes
    ]
    chunks = split_indexed_scene_chunks(
        indexed_pending,
        worker_pool.worker_count,
    )
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        tmp_dir = Path(tmp)
        tasks = []
        for worker_index, chunk_scenes in enumerate(chunks):
            for scene_order, scene in chunk_scenes:
                input_jsonl = (
                    tmp_dir
                    / f"{job.content_id}_scene_{scene_order}_body.jsonl"
                )
                output_jsonl = (
                    tmp_dir
                    / f"{job.content_id}_scene_{scene_order}_output.jsonl"
                )
                failure_jsonl = (
                    tmp_dir
                    / f"{job.content_id}_scene_{scene_order}_failures.jsonl"
                )
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
        ordered_results = sorted(
            results,
            key=lambda item: int(item["scene_order"]),
        )
        merged_path = tmp_dir / f"{job.content_id}_merged_context.jsonl"
        merge_chunk_outputs(
            [Path(result["output_jsonl"]) for result in ordered_results],
            merged_path,
        )
        merged_failures_path = (
            tmp_dir / f"{job.content_id}_merged_failures.jsonl"
        )
        merge_chunk_outputs(
            [Path(result["failure_jsonl"]) for result in ordered_results],
            merged_failures_path,
        )
        successful_records, failure_records = persist_gauss_state(
            job,
            scenes,
            successful_records + read_jsonl(merged_path),
            failure_records + read_scene_failures(merged_failures_path),
        )

    write_gauss_context(job, failure_records)
    write_fingerprint(output_path, fingerprint)
    task_summary = summarize_task_results(results)
    return {
        "success": len(successful_records),
        "failed": len(failure_records),
        "warnings": task_summary["warnings"],
    }


def write_gauss_context(
    job: SceneContextJob,
    failure_records: list[dict[str, Any]],
) -> Path | None:
    output_path = gauss_context_path(job)
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


def _run_gauss_api_task(
    task: dict[str, Any],
    client: GaussApiClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = extract_visual_graphs(
        client=client,
        frame_save_folder=task["frame_save_folder"],
        scenes=read_jsonl(task["input_json_path"]),
        config=config,
        timestamp_json_path=task["timestamp_json_path"],
        final_output_path=task["final_output_path"],
        content_id=task["content_id"],
        failure_output_path=task["failure_output_path"],
    )
    return {
        "ok": True,
        "worker_index": task["worker_index"],
        "task_id": task["task_id"],
        "scene_order": task["scene_order"],
        "output_jsonl": task["final_output_path"],
        "failure_jsonl": task["failure_output_path"],
        "summary": summary,
    }
