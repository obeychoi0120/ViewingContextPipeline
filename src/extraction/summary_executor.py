from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from extraction.backends import QwenBackend
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.descriptions import DescriptionError
from extraction.errors import ExtractionStepError
from extraction.evidence import load_images
from extraction.semantic_graph import SemanticGraphError
from extraction.summary_validation import (
    SUMMARY_SECTIONS,
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
        raw_sections = existing.get("sections")
        if not isinstance(raw_sections, dict):
            raise SummaryContractError("summary sections must be an object")
        sections = parse_summary_sections(
            "\n".join(
                f"{name}: {raw_sections[name]}"
                for name in SUMMARY_SECTIONS
            )
        )
        text = serialize_summary_sections(sections)
    except (AttributeError, KeyError, SummaryContractError, TypeError) as exc:
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


def generate_summaries_once(
    generate: GenerationFunction,
    tasks: list[QwenGenerationTask],
    complete: GenerationCallback,
    on_validation_failure: ValidationFailureCallback | None = None,
) -> None:
    last_errors: dict[str, Exception] = {}
    tasks_by_id = {task.task_id: task for task in tasks}
    handled: set[str] = set()

    def handle_result(task_id: str, text: str) -> None:
        if task_id in handled:
            return
        task = tasks_by_id[task_id]
        handled.add(task_id)
        try:
            complete(task_id, text)
        except (DescriptionError, SemanticGraphError, SummaryContractError) as exc:
            last_errors[task_id] = exc
            if on_validation_failure is not None:
                on_validation_failure(task_id, 1, task.seed, text, exc)

    results = generate(tasks, handle_result)
    # Local test generators may return a mapping without invoking the callback.
    for task in tasks:
        if task.task_id not in handled:
            handle_result(task.task_id, results[task.task_id])

    if last_errors:
        task_ids = [task.task_id for task in tasks if task.task_id in last_errors]
        cause = last_errors[task_ids[0]]
        raise ExtractionStepError(
            f"structured summary failed: task_ids={task_ids}"
        ) from cause


@contextmanager
def qwen_generator(
    *,
    model_path: Path,
    gpus: int | None,
) -> Iterator[GenerationFunction]:
    if gpus is not None:
        with QwenWorkerPool(gpus, str(model_path)) as worker_pool:
            yield worker_pool.generate
        return
    backend = QwenBackend.from_pretrained(str(model_path), use_fc_patch=True)

    def generate(
        tasks: list[QwenGenerationTask],
        on_task_complete: GenerationCallback | None = None,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for task in tasks:
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
