from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm
from ..graph_core.report import write_viewing_context_report

from src.common.manifest import CANONICAL_MANIFEST_PATH
from src.common.output_paths import custom_output_root
from src.video_data_collection.raw_pipeline import (
    multimodal_from_config,
    shot_interval_from_config,
)

from .pipeline import (
    SceneContextWorkerPool,
    init_vlm_model,
    load_config,
    load_ondevice_state,
    load_scene_context_source,
    model_family_from_config,
    model_family_postfix,
    pending_scenes_for_state,
    read_manifest,
    required_scene_context_input_errors,
    run_scene_context_job,
    run_scene_context_job_parallel,
    scene_context_job_video_id,
)


SCENE_CONTEXT_ONDEVICE_CONFIG_PATH = "config/scene_context_extraction_ondevice.json"
SCENE_PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"


@dataclass
class SceneProgress:
    pbar: Any
    started_at: float
    resumed_content_ids: frozenset[str]
    pending_scenes: int
    completed_scenes: int = 0
    generated_tokens: int = 0
    generation_seconds: float = 0.0

    def __enter__(self) -> "SceneProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image-only visual graph JSONL from resized keyframes.")
    parser.add_argument(
        "--manifest",
        default=str(CANONICAL_MANIFEST_PATH),
        help="Input video manifest CSV. Defaults to contracts/manifest.csv.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--settings",
        default=SCENE_CONTEXT_ONDEVICE_CONFIG_PATH,
        help="On-device extraction settings JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every selected scene instead of resuming existing outputs.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for scene-level VLM parallelism, e.g. 0,1,2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv("config/.env")
    output_root = resolve_output_root()
    input_manifest = Path(args.manifest)
    config = load_config(args.settings)
    vlm_config = config
    model_path = vlm_config.get("MODEL_PATH")
    if not model_path:
        raise ValueError(f"model path is required in {args.settings}: MODEL_PATH")
    model_family = model_family_from_config(vlm_config)
    jobs = read_manifest(
        input_manifest,
        model_family=model_family,
        shot_interval=shot_interval_from_config(config),
        multimodal=multimodal_from_config(config),
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    requested_count = len(jobs)
    jobs = filter_processable_jobs(
        jobs,
        unprocessed_path=output_root / "unprocessed.txt",
    )
    terminal_failures = requested_count - len(jobs)
    if not jobs:
        print("Processed 0 on-device scene contexts")
        return 1 if terminal_failures else 0

    gpus = parse_gpus(args.gpus)
    if gpus:
        run_summary = run_parallel_scene_context_jobs(
            jobs=jobs,
            gpus=gpus,
            model_path=model_path,
            vlm_config=vlm_config,
            force=args.force,
        )
        terminal_failures += run_summary["failed"]
        print(
            f"Processed {run_summary['processed']} on-device scene contexts"
        )
        write_viewing_context_report(
            output_root,
            multimodal=jobs[0].multimodal,
            mode=jobs[0].shot_interval,
            source=model_family_postfix(model_family),
            payload={**run_summary, "terminal_failures": terminal_failures},
        )
        return 1 if terminal_failures else 0

    model, processor = init_vlm_model(model_path, model_family)
    processed_count = 0
    with scene_progress(jobs, force=args.force) as progress:
        for job in jobs:
            print_resume_message(progress, job.content_id)
            try:
                summary = run_scene_context_job(
                    job,
                    model=model,
                    processor=processor,
                    vlm_config=vlm_config,
                    force=args.force,
                    progress_callback=lambda count: update_scene_progress(
                        progress,
                        count,
                    ),
                    generation_callback=lambda tokens, seconds: update_generation_progress(
                        progress,
                        tokens,
                        seconds,
                    ),
                )
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                summary = {"success": 0, "failed": 1, "warnings": 0}
                tqdm.write(f"[error] {job.content_id}: {exc}")
            terminal_failures += summary["failed"]
            processed_count += 1
            print_done_message(job.content_id, summary)
    print(f"Processed {processed_count} on-device scene contexts")
    write_viewing_context_report(
        output_root,
        multimodal=jobs[0].multimodal,
        mode=jobs[0].shot_interval,
        source=model_family_postfix(model_family),
        payload={"processed": processed_count, "terminal_failures": terminal_failures},
    )
    return 1 if terminal_failures else 0


def resolve_output_root() -> Path:
    value = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not value:
        raise ValueError("OUTPUT_SAVE_PATH is required in config/.env or the environment")
    return custom_output_root(value)


def parse_gpus(value: str | None) -> list[str]:
    if not value:
        return []
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("--gpus must include at least one GPU id")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus contains duplicate GPU ids")
    return gpus


def filter_processable_jobs(jobs, unprocessed_path: Path):
    processable_jobs = []
    for job in jobs:
        errors = required_scene_context_input_errors(job)
        if not errors:
            processable_jobs.append(job)
            continue

        print(f"[SKIP] Missing required content for {job.content_id}: {'; '.join(errors)}")
        append_unprocessed_video_id(scene_context_job_video_id(job), unprocessed_path)
    return processable_jobs


def append_unprocessed_video_id(video_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(f"{video_id}\n")
        f.flush()


def run_parallel_scene_context_jobs(
    jobs,
    gpus: list[str],
    model_path: str,
    vlm_config: dict,
    force: bool = False,
) -> dict[str, int]:
    processed_count = 0
    failed_count = 0
    warning_count = 0
    with SceneContextWorkerPool(
        gpus=gpus,
        model_path=model_path,
        vlm_config=vlm_config,
    ) as worker_pool:
        with scene_progress(jobs, force=force) as progress:
            for job in jobs:
                print_resume_message(progress, job.content_id)
                summary = run_scene_context_job_parallel(
                    job,
                    worker_pool=worker_pool,
                    vlm_config=vlm_config,
                    force=force,
                    progress_callback=lambda count: update_scene_progress(
                        progress,
                        count,
                    ),
                    generation_callback=lambda tokens, seconds: update_generation_progress(
                        progress,
                        tokens,
                        seconds,
                    ),
                )
                processed_count += 1
                failed_count += summary["failed"]
                warning_count += summary["warnings"]
                print_done_message(job.content_id, summary)
    return {
        "processed": processed_count,
        "failed": failed_count,
        "warnings": warning_count,
    }


def scene_progress(jobs, force: bool = False):
    total = 0
    skipped = 0
    resumed_content_ids: set[str] = set()
    for job in jobs:
        scenes = load_scene_context_source(job)
        total += len(scenes)
        if force:
            continue
        successful_records, failure_records = load_ondevice_state(job, scenes)
        resumed_scenes = len(scenes) - len(
            pending_scenes_for_state(
                scenes,
                successful_records,
                failure_records,
            )
        )
        skipped += resumed_scenes
        if resumed_scenes:
            resumed_content_ids.add(job.content_id)
    pbar = tqdm(
        total=total,
        initial=skipped,
        desc="On-device scene contexts",
        unit="scene",
        bar_format=SCENE_PROGRESS_BAR_FORMAT,
    )
    return SceneProgress(
        pbar=pbar,
        started_at=perf_counter(),
        resumed_content_ids=frozenset(resumed_content_ids),
        pending_scenes=total - skipped,
    )


def update_scene_progress(progress: SceneProgress, count: int) -> None:
    progress.completed_scenes += count
    progress.pbar.update(count)
    if progress.completed_scenes <= 0:
        return
    elapsed_seconds = perf_counter() - progress.started_at
    average_seconds = elapsed_seconds / progress.completed_scenes
    remaining_scenes = max(
        progress.pending_scenes - progress.completed_scenes,
        0,
    )
    eta = tqdm.format_interval(average_seconds * remaining_scenes)
    tps = (
        f"{progress.generated_tokens / progress.generation_seconds:.2f}"
        if progress.generation_seconds > 0
        else "n/a"
    )
    progress.pbar.set_postfix_str(
        f"ETA {eta} | {average_seconds:.2f}s/scene | TPS {tps}",
        refresh=True,
    )


def update_generation_progress(
    progress: SceneProgress,
    generated_tokens: int,
    generation_seconds: float,
) -> None:
    progress.generated_tokens += generated_tokens
    progress.generation_seconds += generation_seconds


def print_resume_message(progress: SceneProgress, content_id: str) -> None:
    if content_id in progress.resumed_content_ids:
        tqdm.write(f"[Resume] Processing {content_id}")


def print_done_message(content_id: str, summary: dict[str, int]) -> None:
    tqdm.write(
        f"[Done] {content_id} - success: {summary['success']} | "
        f"warning: {summary['warnings']} | failed: {summary['failed']}"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
