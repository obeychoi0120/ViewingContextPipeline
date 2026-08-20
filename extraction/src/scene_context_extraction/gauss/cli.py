from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
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

from .api import (
    GaussApiClient,
    api_workers_from_config,
    require_nonempty_string,
    validate_http_url,
)
from .pipeline import (
    GaussApiWorkerPool,
    load_config,
    load_gauss_state,
    load_scene_context_source,
    pending_scenes_for_state,
    read_manifest,
    required_scene_context_input_errors,
    run_scene_context_job,
    scene_context_job_video_id,
)


GAUSS_CONFIG_PATH = "config/scene_context_extraction_gauss.json"
SCENE_PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
)


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
    parser = argparse.ArgumentParser(
        description=(
            "Extract scene visual graph JSONL with the remote AgentGauss API."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(CANONICAL_MANIFEST_PATH),
        help="Input video manifest CSV. Defaults to contracts/manifest.csv.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every selected scene instead of resuming existing outputs.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Unsupported for Gauss; set API_WORKERS in the config instead.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpus is not None:
        raise ValueError(
            "--gpus is not supported by the Gauss backend; "
            "set API_WORKERS in config/scene_context_extraction_gauss.json"
        )

    load_dotenv("config/.env")
    config = load_config(GAUSS_CONFIG_PATH)
    apply_gauss_network_config(config)
    output_root = resolve_output_root()
    client = GaussApiClient.from_config(config)
    worker_count = api_workers_from_config(config)
    jobs = read_manifest(
        Path(args.manifest),
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
        print("Processed 0 Gauss scene contexts")
        return 1 if terminal_failures else 0

    client.ensure_ready()
    processed_count = 0
    with GaussApiWorkerPool(
        client=client,
        worker_count=worker_count,
        config=config,
    ) as worker_pool:
        with scene_progress(jobs, force=args.force) as progress:
            for job in jobs:
                print_resume_message(progress, job.content_id)
                try:
                    summary = run_scene_context_job(
                        job,
                        worker_pool=worker_pool,
                        config=config,
                        force=args.force,
                        progress_callback=lambda count: update_scene_progress(
                            progress,
                            count,
                        ),
                        generation_callback=lambda tokens, seconds: (
                            update_generation_progress(
                                progress,
                                tokens,
                                seconds,
                            )
                        ),
                    )
                except (RuntimeError, ValueError, FileNotFoundError) as exc:
                    summary = {"success": 0, "failed": 1, "warnings": 0}
                    tqdm.write(f"[error] {job.content_id}: {exc}")
                terminal_failures += summary["failed"]
                processed_count += 1
                print_done_message(job.content_id, summary)

    print(f"Processed {processed_count} Gauss scene contexts")
    write_viewing_context_report(
        output_root,
        multimodal=jobs[0].multimodal,
        mode=jobs[0].shot_interval,
        source="gaussa_gemma4_e2b_v0_3",
        payload={"processed": processed_count, "terminal_failures": terminal_failures},
    )
    return 1 if terminal_failures else 0


def apply_gauss_network_config(config: dict[str, Any]) -> None:
    proxy_url = optional_config_string(config, "API_PROXY_URL")
    if proxy_url is not None:
        proxy_url = validate_http_url(proxy_url, field_name="API_PROXY_URL")

    cert_file = optional_config_string(config, "API_SSL_CERT_FILE")
    if cert_file is not None:
        cert_path = Path(cert_file).expanduser()
        if not cert_path.is_file():
            raise ValueError(
                f"API_SSL_CERT_FILE does not exist or is not a file: {cert_path}"
            )
        cert_file = str(cert_path)

    no_proxy = optional_config_string(config, "API_NO_PROXY")

    if proxy_url is not None:
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            os.environ[key] = proxy_url
    if cert_file is not None:
        os.environ["SSL_CERT_FILE"] = cert_file
    if no_proxy is not None:
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy


def optional_config_string(
    config: dict[str, Any],
    key: str,
) -> str | None:
    if key not in config:
        return None
    return require_nonempty_string(config[key], key)


def resolve_output_root() -> Path:
    value = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not value:
        raise ValueError(
            "OUTPUT_SAVE_PATH is required in config/.env or the environment"
        )
    return custom_output_root(value)


def filter_processable_jobs(jobs, unprocessed_path: Path):
    processable_jobs = []
    for job in jobs:
        errors = required_scene_context_input_errors(job)
        if not errors:
            processable_jobs.append(job)
            continue
        print(
            f"[SKIP] Missing required content for {job.content_id}: "
            f"{'; '.join(errors)}"
        )
        append_unprocessed_video_id(
            scene_context_job_video_id(job),
            unprocessed_path,
        )
    return processable_jobs


def append_unprocessed_video_id(video_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(f"{video_id}\n")
        file.flush()


def scene_progress(jobs, force: bool = False) -> SceneProgress:
    total = 0
    skipped = 0
    resumed_content_ids: set[str] = set()
    for job in jobs:
        scenes = load_scene_context_source(job)
        total += len(scenes)
        if force:
            continue
        successful_records, failure_records = load_gauss_state(job, scenes)
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
        desc="Gauss scene contexts",
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
