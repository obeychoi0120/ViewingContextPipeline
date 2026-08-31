from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from extraction.backends import QwenBackend
from extraction.backends.qwen_workers import (
    QwenGenerationTask,
    QwenStatusCallback,
    QwenWorkerPool,
)
from extraction.descriptions import DescriptionError
from extraction.errors import ExtractionStepError
from extraction.evidence import load_images
from extraction.semantic_graph import SemanticGraphError
from extraction.summary_validation import (
    SummaryContractError,
    parse_summary_sections,
    serialize_summary_sections,
)
from pipeline_runtime import read_json


GenerationCallback = Callable[[str, str], None]
ValidationFailureCallback = Callable[
    [str, int, int | None, str, Exception],
    None,
]
GenerationFunction = Callable[
    [list[QwenGenerationTask], GenerationCallback | None],
    dict[str, str],
]


def reuse_summary_document(
    output_path: Path,
    *,
    schema_version: str,
    content_id: str,
    arm: str,
    scene_count: int,
) -> dict[str, Any]:
    existing = read_json(output_path)
    try:
        sections = parse_summary_sections(
            json.dumps(existing.get("sections"), ensure_ascii=False)
        )
        text = serialize_summary_sections(sections)
    except (AttributeError, SummaryContractError, TypeError) as exc:
        raise ExtractionStepError(
            f"incompatible structured summary output: {output_path}; "
            "use --force or a new run_id"
        ) from exc
    expected = {
        "schema_version": schema_version,
        "content_id": content_id,
        "arm": arm,
        "status": "complete",
        "sections": sections,
        "text": text,
        "scene_count": scene_count,
    }
    if existing != expected:
        raise ExtractionStepError(
            f"incompatible structured summary output: {output_path}; "
            "use --force or a new run_id"
        )
    return expected


def generate_summaries_with_retry(
    generate: GenerationFunction,
    tasks: list[QwenGenerationTask],
    complete: GenerationCallback,
    retry_settings: dict[str, Any],
    on_validation_failure: ValidationFailureCallback | None = None,
    *,
    batch_size: int = 1,
) -> int:
    if batch_size <= 0:
        raise ValueError("summary retry batch_size must be positive")
    last_errors: dict[str, Exception] = {}

    def run_attempt(
        attempt_tasks: list[QwenGenerationTask],
        *,
        attempt: int,
        seed: int | None,
    ) -> list[QwenGenerationTask]:
        tasks_by_id = {task.task_id: task for task in attempt_tasks}
        handled: set[str] = set()
        failed: list[QwenGenerationTask] = []

        def handle_result(task_id: str, text: str) -> None:
            if task_id in handled:
                return
            task = tasks_by_id[task_id]
            handled.add(task_id)
            try:
                complete(task_id, text)
                last_errors.pop(task_id, None)
            except (DescriptionError, SemanticGraphError, SummaryContractError) as exc:
                failed.append(task)
                last_errors[task_id] = exc
                if on_validation_failure is not None:
                    on_validation_failure(task_id, attempt, seed, text, exc)

        results = generate(attempt_tasks, handle_result)
        # Local test generators may return a mapping without invoking the callback.
        for task in attempt_tasks:
            if task.task_id not in handled:
                handle_result(task.task_id, results[task.task_id])
        return failed

    retry_count = 0
    exhausted: list[QwenGenerationTask] = []
    for start in range(0, len(tasks), batch_size):
        pending = run_attempt(
            tasks[start : start + batch_size],
            attempt=1,
            seed=None,
        )
        for attempt, seed in enumerate(retry_settings["seeds"], start=2):
            if not pending:
                break
            retry_tasks = [
                replace(
                    task,
                    do_sample=True,
                    seed=int(seed),
                    temperature=float(retry_settings["temperature"]),
                    top_p=float(retry_settings["top_p"]),
                    top_k=int(retry_settings["top_k"]),
                )
                for task in pending
            ]
            retry_count += len(retry_tasks)
            pending = run_attempt(
                retry_tasks,
                attempt=attempt,
                seed=int(seed),
            )
        exhausted.extend(pending)

    if exhausted:
        task_ids = [task.task_id for task in exhausted]
        cause = last_errors[task_ids[0]]
        raise ExtractionStepError(
            f"structured summary failed after {len(retry_settings['seeds'])} retries: "
            f"task_ids={task_ids}"
        ) from cause
    return retry_count


@contextmanager
def qwen_generator(
    *,
    model_path: Path,
    gpus: int | None,
    on_status: QwenStatusCallback | None = None,
) -> Iterator[GenerationFunction]:
    if gpus is not None:
        with QwenWorkerPool(
            gpus,
            str(model_path),
            on_status=on_status,
        ) as worker_pool:
            yield worker_pool.generate
        return
    backend = QwenBackend.from_pretrained(str(model_path), use_fc_patch=True)
    if on_status is not None:
        on_status("worker_ready", {"worker_index": 0, "gpu_id": "default"})

    def generate(
        tasks: list[QwenGenerationTask],
        on_task_complete: GenerationCallback | None = None,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for task in tasks:
            if on_status is not None:
                on_status(
                    "task_started",
                    {
                        "worker_index": 0,
                        "gpu_id": "default",
                        "task_id": task.task_id,
                        "do_sample": task.do_sample,
                        "seed": task.seed,
                    },
                )
            text = backend.generate(
                load_images(list(task.image_paths)),
                task.prompt,
                task.max_new_tokens,
                do_sample=task.do_sample,
                seed=task.seed,
                temperature=task.temperature,
                top_p=task.top_p,
                top_k=task.top_k,
            )
            if on_task_complete is not None:
                on_task_complete(task.task_id, text)
            else:
                results[task.task_id] = text
        return results

    yield generate
