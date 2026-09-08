from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl
from extraction.step_support import result, write_progress


SUMMARY_FAILURE_SCHEMA_VERSION = "summary-generation-failure/v1"


@dataclass(frozen=True)
class SummaryBranch:
    stage: str
    arm: str
    label: str
    schema_version: str
    summary_dir: Path
    failure_dir: Path
    normalize_records: Callable
    content_id: Callable
    build_prompt: Callable
    validate: Callable


def _summary_failure_record(content_id, *, attempt, seed, failure_kind, error, raw_response):
    return {
        "schema_version": SUMMARY_FAILURE_SCHEMA_VERSION,
        "content_id": content_id,
        "attempt": attempt,
        "seed": seed,
        "failure_kind": failure_kind,
        "error": error,
        "raw_response": raw_response,
    }


def _prepare_summaries(branch, scene_paths, template, max_new_tokens, generation, force):
    documents = {}
    pending = {}
    tasks = []
    incompatible = {}
    empty = 0
    for scene_path in scene_paths:
        records = read_jsonl(scene_path)
        content_id = scene_path.stem
        failure_path = branch.failure_dir / f"{content_id}.jsonl"
        if not records:
            empty += 1
            write_jsonl(failure_path, [_summary_failure_record(
                content_id, attempt=None, seed=None, failure_kind="empty_scene_records",
                error=f"{branch.label} summary requires at least one successful scene",
                raw_response="",
            )])
            continue
        records = branch.normalize_records(records, scene_path)
        task_id = branch.content_id(records, scene_path)
        output_path = branch.summary_dir / f"{task_id}.json"
        if output_path.is_file() and not force:
            try:
                documents[task_id] = reuse_summary_document(
                    output_path, schema_version=branch.schema_version, content_id=task_id,
                    arm=branch.arm, scene_count=len(records),
                )
            except ExtractionStepError as exc:
                incompatible[task_id] = str(exc)
            else:
                failure_path.unlink(missing_ok=True)
                continue
        tasks.append(QwenGenerationTask(
            task_id=task_id, image_paths=(), prompt=branch.build_prompt(template, records),
            max_new_tokens=max_new_tokens, **generation,
        ))
        pending[task_id] = (records, output_path)
    return documents, pending, tasks, incompatible, empty


def run_summary_stage(
    branch: SummaryBranch, *, scene_paths, template, max_new_tokens, generation,
    model_path, gpus, force, names, generator_factory, progress_factory,
):
    documents, pending, tasks, incompatible, empty = _prepare_summaries(
        branch, scene_paths, template, max_new_tokens, generation, force,
    )
    description = (f"Graph summaries ({branch.arm.removeprefix('graph_')})"
                   if branch.label == "graph" else "Description summaries")
    failures_by_content = {task.task_id: [] for task in tasks}
    with progress_factory(total=len(scene_paths), initial=len(documents) + empty,
                          desc=description, unit="content") as progress:
        write_progress(progress,
            f"[SUMMARY] {branch.stage} | reused={len(documents)} "
            f"pending={len(tasks)} incompatible={len(incompatible)} "
            f"empty_scenes={empty} force={force} "
            f"repetition_penalty={generation['repetition_penalty']} output={branch.summary_dir}")
        for reason in incompatible.values():
            write_progress(progress, f"[SUMMARY RETRY] {reason}")

        def complete(task_id, text):
            records, output_path = pending[task_id]
            sections = branch.validate(text)
            document = {
                "schema_version": branch.schema_version, "content_id": task_id,
                "arm": branch.arm, "status": "complete", "sections": sections,
                "text": serialize_summary_sections(sections), "scene_count": len(records),
            }
            write_json(output_path, document)
            (branch.failure_dir / f"{task_id}.jsonl").unlink(missing_ok=True)
            documents[task_id] = document
            progress.update(1)

        def failed(task_id, attempt, seed, raw_response, error):
            failures = failures_by_content[task_id]
            failures.append(_summary_failure_record(
                task_id, attempt=attempt, seed=seed, failure_kind="schema_validation",
                error=str(error), raw_response=raw_response,
            ))
            write_jsonl(branch.failure_dir / f"{task_id}.jsonl", failures)
            message = " ".join(str(error).splitlines())
            write_progress(progress,
                f"[Qwen_summary_{branch.arm}_fail] "
                f"{names.get(task_id, f'{task_id}.mp4')} | {message}\n"
                f"Raw output:\n{raw_response or '<empty>'}")
            progress.update(1)

        if tasks:
            with generator_factory(model_path=model_path, gpus=gpus) as generate:
                generate_summaries_once(generate, tasks, complete, failed)
    return result(branch.stage, content_count=len(documents), failure_count=empty)


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
    try:
        existing = read_json(output_path)
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
    except (AttributeError, KeyError, ValueError, TypeError) as exc:
        raise ExtractionStepError(
            f"incompatible structured summary output: {output_path}; {exc}"
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
        mismatched = sorted(
            key for key in existing.keys() | expected.keys()
            if key not in existing or key not in expected or existing[key] != expected[key]
        )
        raise ExtractionStepError(
            f"incompatible structured summary output: {output_path}; "
            f"mismatched fields: {', '.join(mismatched)}"
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
                repetition_penalty=task.repetition_penalty,
            )
            if on_task_complete is not None:
                on_task_complete(task.task_id, text)
            else:
                results[task.task_id] = text
        return results

    yield generate
