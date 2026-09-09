from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.descriptions import DescriptionError
from extraction.errors import ExtractionStepError
from extraction.progress import InferenceProgress
from extraction.semantic_graph import SemanticGraphError
from extraction.summary_validation import (
    SUMMARY_SECTIONS,
    SummaryContractError,
    parse_summary_sections,
    parse_or_repair_summary,
    serialize_summary_sections,
)
from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl
from extraction.step_support import result, write_progress
from extraction.recovery import generate_with_recovery, has_pending_recovery
from extraction.structured_output import OutputValidationError
from extraction.raw_output import RAW_SUMMARY_SCHEMA, raw_summary_document
from extraction.input_tracking import inputs_match, record_inputs, mark_changed, invalidate_inputs


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
    allow_missing: bool = False


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
            invalidate_inputs(branch.summary_dir / f"{content_id}.json")
            write_jsonl(
                failure_path,
                [
                    _summary_failure_record(
                        content_id,
                        attempt=None,
                        seed=None,
                        failure_kind="empty_scene_records",
                        error=f"{branch.label} summary requires at least one successful scene",
                        raw_response="",
                    )
                ],
            )
            continue
        records = branch.normalize_records(records, scene_path)
        task_id = branch.content_id(records, scene_path)
        output_path = branch.summary_dir / f"{task_id}.json"
        if output_path.is_file() and not force:
            try:
                if has_pending_recovery(branch.summary_dir / ".recovery", task_id):
                    raise ExtractionStepError(f"summary has an unfinished recovery: {output_path}")
                if not inputs_match(output_path, records):
                    raise ExtractionStepError(f"summary scene input changed: {output_path}")
                documents[task_id] = reuse_summary_document(
                    output_path,
                    schema_version=branch.schema_version,
                    content_id=task_id,
                    arm=branch.arm,
                    scene_count=len(records),
                )
            except ExtractionStepError as exc:
                incompatible[task_id] = str(exc)
            else:
                failure_path.unlink(missing_ok=True)
                continue
        # Keep a failed regeneration pending even when a previous output is preserved.
        invalidate_inputs(output_path)
        tasks.append(
            QwenGenerationTask(
                task_id=task_id,
                image_paths=(),
                prompt=branch.build_prompt(template, records),
                max_new_tokens=max_new_tokens,
                **generation,
            )
        )
        pending[task_id] = (records, output_path)
    return documents, pending, tasks, incompatible, empty


def run_summary_stage(
    branch: SummaryBranch,
    *,
    scene_paths,
    template,
    max_new_tokens,
    generation,
    model_path,
    gpus,
    force,
    names,
    generator_factory,
    progress_factory,
    qwen_options=None,
    image_limit=6,
    runtime=None,
    penalties=None,
):
    documents, pending, tasks, incompatible, empty = _prepare_summaries(
        branch,
        scene_paths,
        template,
        max_new_tokens,
        generation,
        force,
    )
    description = (
        f"Graph summaries ({branch.arm.removeprefix('graph_')})"
        if branch.label == "graph"
        else "Description summaries"
    )
    failures_by_content = {task.task_id: [] for task in tasks}
    with InferenceProgress(
        total=len(tasks), reused=len(documents), empty=empty,
        desc=description, unit="summary", progress_factory=progress_factory,
    ) as progress:
        write_progress(
            progress,
            f"[SUMMARY] {branch.stage} | reused={len(documents)} "
            f"pending={len(tasks)} incompatible={len(incompatible)} "
            f"empty_scenes={empty} force={force} "
            f"repetition_penalty={generation['repetition_penalty']} output={branch.summary_dir}",
        )
        for reason in incompatible.values():
            write_progress(progress, f"[SUMMARY RETRY] {reason}")
        if runtime:
            unknown = sum(runtime.classification(task_id, document) == "legacy_unknown"
                          for task_id, document in documents.items())
            write_progress(progress, f"[Qwen] reused={len(documents)} legacy_unknown={unknown} pending={len(tasks)}")

        def validate(task_id, text):
            records, output_path = pending[task_id]
            try:
                sections, mode = parse_or_repair_summary(text)
            except SummaryContractError as exc:
                raise OutputValidationError(str(exc)) from exc
            document = {
                "schema_version": branch.schema_version,
                "content_id": task_id,
                "arm": branch.arm,
                "status": "complete",
                "sections": sections,
                "text": serialize_summary_sections(sections),
                "scene_count": len(records),
            }
            return document, mode

        def complete(task_id, document):
            records, output_path = pending[task_id]
            try:
                changed = not output_path.exists() or read_json(output_path) != document
            except ValueError:
                changed = True
            if changed:
                mark_changed(output_path)
            write_json(output_path, document)
            record_inputs(output_path, records)
            (branch.failure_dir / f"{task_id}.jsonl").unlink(missing_ok=True)
            documents[task_id] = document
            if runtime:
                runtime.record(task_id, document, status=document["status"])
            progress.complete()

        def failed(task_id, attempts):
            failures = failures_by_content[task_id]
            failures.extend(_summary_failure_record(
                task_id, attempt=row["attempt"], seed=row["seed"],
                failure_kind="schema_validation", error=row["error"],
                raw_response=row["raw_response"],
            ) for row in attempts)
            write_jsonl(branch.failure_dir / f"{task_id}.jsonl", failures)
            if runtime:
                runtime.record(task_id, failures[-1], status="failed")
            message = " ".join(str(attempts[-1]["error"]).splitlines())
            write_progress(
                progress,
                f"[Qwen_summary_{branch.arm}_fail] "
                f"{names.get(task_id, f'{task_id}.mp4')} | {message}",
            )
            progress.complete(failed=True)

        if tasks:
            allowed_failures = {
                task.task_id for task in tasks
                if branch.allow_missing and not pending[task.task_id][1].exists()
            }
            with generator_factory(
                model_path=model_path, gpus=gpus, settings=qwen_options,
                image_limit=image_limit, runtime=runtime,
                log=lambda message: write_progress(progress, message),
                on_progress=lambda stats: qwen_progress(progress, stats),
            ) as generate:
                generate_with_recovery(
                    generate, tasks, complete=complete, failed=failed, validate=validate,
                    penalties=penalties if penalties is not None else generation["repetition_penalty"],
                    directory=branch.summary_dir / ".recovery",
                    identity={"model": str(model_path), "settings": qwen_options,
                              "backend": "vllm-0.28.0", "arm": branch.arm},
                    raw_fallback=lambda task_id, raw: raw_summary_document(
                        task_id, branch.arm, len(pending[task_id][0]), raw),
                    force=force, runtime=runtime,
                    log=lambda message: write_progress(progress, message),
                )
            failed_ids = [key for key, rows in failures_by_content.items()
                          if rows and key not in allowed_failures]
            if failed_ids:
                raise ExtractionStepError(f"structured summary failed: task_ids={failed_ids}")
        failure_count = empty + sum(bool(rows) for rows in failures_by_content.values())
        if branch.allow_missing and failure_count:
            write_progress(
                progress,
                f"[SUMMARY FALLBACK] {branch.arm}: {failure_count} missing summaries; "
                "embed-representations will use Qwen Graph summaries",
            )
    return result(branch.stage, content_count=len(documents), failure_count=failure_count)


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
    scene_count: int | None,
) -> dict[str, Any]:
    """Validate the document; None checks its stored count before a refresh decision."""
    try:
        existing = read_json(output_path)
        stored_count = existing.get("scene_count")
        if type(stored_count) is not int or stored_count <= 0:
            raise SummaryContractError("summary scene_count must be a positive integer")
        if existing.get("schema_version") == RAW_SUMMARY_SCHEMA:
            text = existing.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SummaryContractError("raw summary text must not be empty")
            expected = raw_summary_document(
                content_id, arm, stored_count if scene_count is None else scene_count, text)
            if expected != existing:
                raise SummaryContractError("raw summary fields or identity do not match")
            return expected
        raw_sections = existing.get("sections")
        if not isinstance(raw_sections, dict):
            raise SummaryContractError("summary sections must be an object")
        sections = parse_summary_sections(
            "\n".join(f"{name}: {raw_sections[name]}" for name in SUMMARY_SECTIONS)
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
        "scene_count": stored_count if scene_count is None else scene_count,
    }
    if existing != expected:
        mismatched = sorted(
            key
            for key in existing.keys() | expected.keys()
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
    *,
    allowed_failures: set[str] | None = None,
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

    permitted = allowed_failures or set()
    task_ids = [
        task.task_id for task in tasks
        if task.task_id in last_errors and task.task_id not in permitted
    ]
    if task_ids:
        cause = last_errors[task_ids[0]]
        raise ExtractionStepError(f"structured summary failed: task_ids={task_ids}") from cause


def qwen_progress(progress, stats):
    progress.update_stats(stats)


@contextmanager
def qwen_generator(
    *, model_path: Path, gpus: int | None, settings=None, image_limit=6,
    runtime=None, log=None, on_progress=None,
) -> Iterator[GenerationFunction]:
    def ready(event):
        if runtime:
            runtime.engine_ready(event)
        if log:
            log(f"[Qwen] GPU {event['gpu_id']} ready | {event['gpu_name']} | "
                f"settings={event['settings']}")

    with ExitStack() as resources:
        pool = None
        def generate(tasks, on_task_complete=None):
            nonlocal pool
            if pool is None:
                pool = resources.enter_context(QwenWorkerPool(
                    1 if gpus is None else gpus, str(model_path), settings=settings,
                    image_limit=image_limit, on_runtime=ready, on_progress=on_progress,
                ))
            def complete(task_id, text):
                if runtime:
                    runtime.current_result = pool.last_result
                try:
                    on_task_complete(task_id, text)
                finally:
                    if runtime:
                        runtime.current_result = None
            return pool.generate(tasks, complete if on_task_complete else None)
        yield generate
