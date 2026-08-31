from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm import tqdm

from extraction.backends import GeminiGenerationOutcome, GeminiWorkerPool, QwenBackend
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.data_preparation.microlens import prepare_catalog
from extraction.descriptions import (
    DescriptionError,
    SCENE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    description_summary_prompt,
    validate_summary as validate_description_summary,
)
from extraction.evidence import (
    build_scene_evidence,
    load_images,
)
from extraction.monitoring import (
    graph_skip_message,
    scene_messages,
    summary_message,
    video_names,
)
from extraction.semantic_graph import (
    SCENE_EXTRACTION_PROMPT,
    SUMMARY_SCHEMA_VERSION as GRAPH_SUMMARY_SCHEMA_VERSION,
    SemanticGraphError,
    graph_semantic_warnings,
    graph_summary_prompt,
    parse_or_repair_graph,
    validate_summary as validate_graph_summary,
)
from extraction.summary_validation import (
    SummaryContractError,
    parse_summary_sections,
    serialize_summary_sections,
)
from pipeline_runtime import (
    RunContext,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


class ExtractionStepError(RuntimeError):
    pass


GRAPH_SOURCES = ("qwen", "gemini")


def graph_stage_name(stage: str, source: str) -> str:
    if source not in GRAPH_SOURCES:
        raise ValueError(f"unsupported graph source: {source}")
    return f"{stage}-{source}"


def _reuse_summary_document(
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
    if any(
        existing.get(key) != value
        for key, value in expected.items()
        if key != "text"
    ):
        raise ExtractionStepError(
            f"incompatible structured summary output: {output_path}; "
            "use --force or a new run_id"
        )
    if existing != expected:
        write_json(output_path, expected)
    return expected


def _generate_summaries_with_retry(
    generate: GenerationFunction,
    tasks: list[QwenGenerationTask],
    complete: GenerationCallback,
    retry_settings: dict[str, Any],
) -> int:
    last_errors: dict[str, Exception] = {}

    def run_attempt(attempt_tasks: list[QwenGenerationTask]) -> list[QwenGenerationTask]:
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

        results = generate(attempt_tasks, handle_result)
        # Keep compatibility with simple/local generators that return a result
        # mapping without invoking the optional completion callback.
        for task in attempt_tasks:
            if task.task_id not in handled:
                handle_result(task.task_id, results[task.task_id])
        return failed

    pending = run_attempt(tasks)

    retry_count = 0
    for seed in retry_settings["seeds"]:
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
        pending = run_attempt(retry_tasks)

    if pending:
        task_ids = [task.task_id for task in pending]
        cause = last_errors[task_ids[0]]
        raise ExtractionStepError(
            f"structured summary failed after {len(retry_settings['seeds'])} retries: "
            f"task_ids={task_ids}"
        ) from cause
    return retry_count


GenerationCallback = Callable[[str, str], None]
GenerationFunction = Callable[
    [list[QwenGenerationTask], GenerationCallback | None],
    dict[str, str],
]


@contextmanager
def _qwen_generator(
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
            results[task.task_id] = text
            if on_task_complete is not None:
                on_task_complete(task.task_id, text)
        return results

    yield generate


def _write_progress(progress: tqdm, message: str) -> None:
    tqdm.write(message, file=progress.fp)


def _complete_content_progress(progress: tqdm) -> None:
    progress.update(1)
    _write_progress(progress, "")


def _write_failure_jsonl(path: Path, failures: list[dict[str, Any]]) -> None:
    if failures:
        write_jsonl(path, failures)
    else:
        path.unlink(missing_ok=True)


def _video_name_map(context: RunContext) -> dict[str, str]:
    return video_names(read_jsonl(context.cohort_dir / "catalog.jsonl"))


def _scene_generation_rows(
    visual: dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    scenes = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in build_scene_evidence(
        scenes, visual["frames_dir"], visual["timestamp_json"]
    ):
        fallback_idx = scene["fallback_idx"]
        scene_idx = scene["scene_idx"]
        keyframes = scene["keyframes"]
        image_paths = scene["image_paths"]
        if not keyframes or len(image_paths) != len(keyframes):
            raise ExtractionStepError(
                f"{visual['content_id']} scene {scene_idx} has "
                f"{len(image_paths)} of {len(keyframes)} keyframes"
            )
        task_id = f"{visual['content_id']}:{fallback_idx}"
        rows.append({
            "task": QwenGenerationTask(
                task_id=task_id,
                image_paths=tuple(image_paths),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            ),
            "scene_idx": scene_idx,
            "scene_start_seconds": scene["scene_start_seconds"],
            "scene_end_seconds": scene["scene_end_seconds"],
            "keyframes": keyframes,
            "image_paths": image_paths,
        })
    if not rows:
        raise ExtractionStepError(f"{visual['content_id']} has no scenes")
    return rows


def _result(
    stage: str,
    *,
    content_count: int,
    failure_count: int = 0,
    retry_count: int | None = None,
) -> dict[str, Any]:
    result = {
        "stage": stage,
        "content_count": content_count,
        "failure_count": failure_count,
    }
    if retry_count is not None:
        result["retry_count"] = retry_count
    return result


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ExtractionStepError(f"missing {label}: {path}")
    return path


def _minimal_graph_records(
    records: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    try:
        minimal = [
            {
                "scene_idx": row["scene_idx"],
                "keyframes": row["keyframes"],
                "graph": row["graph"],
                "parse_mode": row.get("parse_mode", "unknown"),
                "semantic_warnings": row.get(
                    "semantic_warnings", graph_semantic_warnings(row["graph"])
                ),
            }
            for row in records
        ]
    except KeyError as exc:
        raise ExtractionStepError(
            f"invalid graph scene file, missing {exc.args[0]}: {path}"
        ) from exc
    if minimal != records:
        write_jsonl(path, minimal)
    return minimal


def _minimal_description_records(
    records: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    required = (
        "schema_version",
        "content_id",
        "scene_idx",
        "keyframes",
        "description",
    )
    try:
        minimal = [{key: row[key] for key in required} for row in records]
    except KeyError as exc:
        raise ExtractionStepError(
            f"invalid description scene file, missing {exc.args[0]}: {path}"
        ) from exc
    if minimal != records:
        write_jsonl(path, minimal)
    return minimal


def _minimal_graph_failures(
    failures: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    minimal = [
        {
            key: row[key]
            for key in ("scene_idx", "keyframes", "failure_kind", "error", "raw_response")
            if key in row
        }
        for row in failures
    ]
    if minimal != failures:
        _write_failure_jsonl(path, minimal)
    return minimal


def _visual_rows(context: RunContext) -> list[dict[str, Any]]:
    catalog_path = _require_file(
        context.cohort_dir / "catalog.jsonl",
        "cohort catalog",
    )
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(catalog_path):
        content_id = str(item["content_id"])
        frames_dir = context.evidence_dir / "resized_keyframes" / content_id
        timestamp = (
            context.cohort_dir
            / "source_assets"
            / content_id
            / "assets"
            / "timestamp_fixed_30s.json"
        )
        frames = (
            sorted(
                path
                for path in frames_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
            if frames_dir.is_dir()
            else []
        )
        if not frames or not timestamp.is_file():
            raise ExtractionStepError(f"missing visual evidence for {content_id}")
        rows.append({
            "content_id": content_id,
            "item_id": str(item["item_id"]),
            "frames_dir": str(frames_dir),
            "timestamp_json": str(timestamp),
        })
    if not rows:
        raise ExtractionStepError(f"empty cohort catalog: {catalog_path}")
    return rows


def prepare_input_data(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    if not force:
        try:
            rows = _visual_rows(context)
        except ExtractionStepError:
            pass
        else:
            return _result("prepare-input-data", content_count=len(rows))
    catalog_path = _require_file(context.cohort_dir / "catalog.jsonl", "cohort catalog")
    catalog = read_jsonl(catalog_path)
    settings = context.config["extraction"]["visual_evidence"]
    result = prepare_catalog(
        catalog,
        assets_root=context.cohort_dir / "source_assets",
        output_root=context.run_root,
        image_size=tuple(settings["image_resolution"]),
        force=force,
    )
    if result["failed"] or result["succeeded"] != len(catalog):
        raise ExtractionStepError(f"visual evidence preparation is incomplete: {result}")
    rows = _visual_rows(context)
    return _result("prepare-input-data", content_count=len(rows))


def extract_graph_scenes(
    context: RunContext,
    *,
    model: str,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    if model not in GRAPH_SOURCES:
        raise ValueError(f"unsupported graph extractor model: {model}")
    if model == "gemini" and gpus is not None:
        raise ValueError("--gpus cannot be used with --model gemini")
    stage = graph_stage_name("extract-graph-scenes", model)
    context.initialize()
    settings = context.config["extraction"]["graph"]
    prompt = SCENE_EXTRACTION_PROMPT
    model_path: Path | None = None
    if model == "qwen":
        model_path = context.path("models", "qwen")
    visual_rows = _visual_rows(context)
    names = _video_name_map(context)
    scene_dir = context.graph_scene_dir(model)
    failure_dir = context.graph_failure_dir(model)
    records_by_content: dict[str, list[dict[str, Any]]] = {}
    failures_by_content: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for visual in visual_rows:
        path = scene_dir / f"{visual['content_id']}.jsonl"
        failure_path = failure_dir / f"{visual['content_id']}.jsonl"
        scene_rows = _scene_generation_rows(
            visual,
            prompt=prompt,
            max_new_tokens=int(settings["scene_max_new_tokens"]),
        )
        expected_scene_indices = {int(row["scene_idx"]) for row in scene_rows}
        if path.is_file() and not force:
            existing = read_jsonl(path)
            failures = read_jsonl(failure_path) if failure_path.is_file() else []
            failures = _minimal_graph_failures(failures, failure_path)
            if not failures:
                failure_path.unlink(missing_ok=True)
            covered = {
                int(row["scene_idx"])
                for row in [*existing, *failures]
            }
            if covered == expected_scene_indices:
                content_id = str(visual["content_id"])
                existing = _minimal_graph_records(existing, path)
                records_by_content[content_id] = existing
                failures_by_content[content_id] = failures
                continue
        pending.append((visual, scene_rows))

    with tqdm(
        total=len(visual_rows),
        initial=len(records_by_content),
        desc=f"Graph scenes ({model})",
        unit="content",
    ) as progress:
        def complete_content(
            visual: dict[str, Any],
            scene_rows: list[dict[str, Any]],
            generated: dict[str, tuple[str, str | None]],
        ) -> None:
            records: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            for row in scene_rows:
                raw_response, generation_error = generated[row["task"].task_id]
                result = (
                    parse_or_repair_graph(raw_response)
                    if generation_error is None
                    else None
                )
                if result is not None and result.graph is not None:
                    records.append({
                        "scene_idx": row["scene_idx"],
                        "keyframes": row["keyframes"],
                        "graph": result.graph,
                        "parse_mode": result.parse_mode,
                        "semantic_warnings": graph_semantic_warnings(result.graph),
                    })
                else:
                    failures.append({
                        "scene_idx": row["scene_idx"],
                        "keyframes": row["keyframes"],
                        "failure_kind": (
                            "generation" if generation_error else "json_repair"
                        ),
                        "error": generation_error
                        or (result.error if result is not None else None)
                        or "JSON repair failed",
                        "raw_response": raw_response,
                    })
            path = scene_dir / f"{visual['content_id']}.jsonl"
            failure_path = failure_dir / f"{visual['content_id']}.jsonl"
            records.sort(key=lambda row: int(row["scene_idx"]))
            failures.sort(key=lambda row: int(row["scene_idx"]))
            write_jsonl(path, records)
            _write_failure_jsonl(failure_path, failures)
            content_id = str(visual["content_id"])
            records_by_content[content_id] = records
            failures_by_content[content_id] = failures
            video_name = names.get(content_id, f"{content_id}.mp4")
            for message in scene_messages(
                video_name,
                records,
                arm="graph",
                source=model,
            ):
                _write_progress(progress, message)
            for failure in failures:
                _write_progress(
                    progress,
                    graph_skip_message(video_name, failure, source=model),
                )
            _complete_content_progress(progress)

        if pending and model == "qwen":
            assert model_path is not None
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                for visual, scene_rows in pending:
                    responses = generate([row["task"] for row in scene_rows], None)
                    complete_content(
                        visual,
                        scene_rows,
                        {task_id: (text, None) for task_id, text in responses.items()},
                    )
        elif pending:
            gemini = context.config["models"]["gemini"]
            task_context: dict[
                str,
                tuple[str, dict[str, Any], list[dict[str, Any]]],
            ] = {}
            generated_by_content: dict[str, dict[str, tuple[str, str | None]]] = {}
            tasks: list[QwenGenerationTask] = []
            for visual, scene_rows in pending:
                content_id = str(visual["content_id"])
                generated_by_content[content_id] = {}
                for row in scene_rows:
                    task = row["task"]
                    tasks.append(task)
                    task_context[task.task_id] = (
                        content_id,
                        visual,
                        scene_rows,
                    )

            def complete_gemini_scene(outcome: GeminiGenerationOutcome) -> None:
                content_id, visual, scene_rows = task_context[outcome.task_id]
                responses = generated_by_content[content_id]
                responses[outcome.task_id] = (outcome.text, outcome.error)
                if len(responses) == len(scene_rows):
                    complete_content(visual, scene_rows, responses)

            pool = GeminiWorkerPool(
                int(settings["gemini_concurrency"]),
                project_id=str(gemini["project_id"]),
                location=str(gemini["location"]),
                model_id=str(gemini["model_id"]),
                temperature=float(gemini["temperature"]),
                max_output_tokens=int(gemini["max_output_tokens"]),
                thinking_level=str(gemini["thinking_level"]),
                media_resolution=str(gemini["media_resolution"]),
            )
            pool.generate(tasks, complete_gemini_scene)
    failures = [
        record
        for visual in visual_rows
        for record in failures_by_content[str(visual["content_id"])]
    ]
    return _result(
        stage,
        content_count=len(visual_rows),
        failure_count=len(failures),
    )


def summarize_graph(
    context: RunContext,
    *,
    source: str,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    if source not in GRAPH_SOURCES:
        raise ValueError(f"unsupported graph source: {source}")
    stage = graph_stage_name("summarize-graph", source)
    context.initialize()
    settings = context.config["extraction"]["graph"]
    retry_settings = context.config["extraction"]["summary_retry"]
    prompt_path = context.config_path("extraction", "graph", "summary_prompt")
    template = prompt_path.read_text(encoding="utf-8")
    model_path = context.path("models", "qwen")
    visual_rows = _visual_rows(context)
    names = _video_name_map(context)
    scene_dir = context.graph_scene_dir(source)
    summary_dir = context.graph_summary_dir(source)
    if not scene_dir.is_dir():
        raise ExtractionStepError(f"missing graph scene directory: {scene_dir}")
    scene_paths = [scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows]
    if not all(path.is_file() for path in scene_paths):
        missing = next(path for path in scene_paths if not path.is_file())
        raise ExtractionStepError(f"missing graph scene output: {missing}")
    documents_by_content: dict[str, dict[str, Any]] = {}
    pending: list[tuple[list[dict[str, Any]], Path, str]] = []
    tasks: list[QwenGenerationTask] = []
    empty_scene_files = 0
    for scene_path in scene_paths:
        records = read_jsonl(scene_path)
        if not records:
            empty_scene_files += 1
            continue
        records = _minimal_graph_records(records, scene_path)
        content_id = scene_path.stem
        output_path = summary_dir / f"{content_id}.json"
        if output_path.is_file() and not force:
            documents_by_content[content_id] = _reuse_summary_document(
                output_path,
                schema_version=GRAPH_SUMMARY_SCHEMA_VERSION,
                content_id=content_id,
                arm=f"graph_{source}",
                scene_count=len(records),
            )
            continue
        prompt = graph_summary_prompt(template, records)
        task_id = content_id
        tasks.append(
            QwenGenerationTask(
                task_id=task_id,
                image_paths=(),
                prompt=prompt,
                max_new_tokens=int(settings["summary_max_new_tokens"]),
            )
        )
        pending.append((records, output_path, task_id))

    pending_by_task = {
        task_id: (records, output_path)
        for records, output_path, task_id in pending
    }
    with tqdm(
        total=len(visual_rows),
        initial=len(documents_by_content),
        desc=f"Graph summaries ({source})",
        unit="content",
    ) as progress:
        def complete_graph_summary(task_id: str, text: str) -> None:
            records, output_path = pending_by_task[task_id]
            sections = validate_graph_summary(text)
            summary = serialize_summary_sections(sections)
            content_id = task_id
            document = {
                "schema_version": GRAPH_SUMMARY_SCHEMA_VERSION,
                "content_id": content_id,
                "arm": f"graph_{source}",
                "status": "complete",
                "sections": sections,
                "text": summary,
                "scene_count": len(records),
            }
            write_json(output_path, document)
            documents_by_content[content_id] = document
            _write_progress(
                progress,
                summary_message(
                    names.get(content_id, f"{content_id}.mp4"),
                    arm="graph",
                    scene_count=len(records),
                    text=summary,
                    source=source,
                ),
            )
            _complete_content_progress(progress)

        retry_count = 0
        if tasks:
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                retry_count = _generate_summaries_with_retry(
                    generate,
                    tasks,
                    complete_graph_summary,
                    retry_settings,
                )
    return _result(
        stage,
        content_count=len(documents_by_content),
        failure_count=empty_scene_files,
        retry_count=retry_count,
    )


def extract_description_scenes(
    context: RunContext,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    settings = context.config["extraction"]["description"]
    prompt_path = context.config_path("extraction", "description", "scene_prompt")
    prompt = prompt_path.read_text(encoding="utf-8")
    model_path = context.path("models", "qwen")
    visual_rows = _visual_rows(context)
    names = _video_name_map(context)
    records_by_content: dict[str, list[dict[str, Any]]] = {}
    failures_by_content: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for visual in visual_rows:
        path = context.description_scene_dir / f"{visual['content_id']}.jsonl"
        failure_path = context.description_failure_dir / f"{visual['content_id']}.jsonl"
        scene_rows = _scene_generation_rows(
            visual,
            prompt=prompt,
            max_new_tokens=int(settings["scene_max_new_tokens"]),
        )
        expected_scene_indices = {int(row["scene_idx"]) for row in scene_rows}
        if path.is_file() and not force:
            existing = read_jsonl(path)
            failures = read_jsonl(failure_path) if failure_path.is_file() else []
            if not failures:
                failure_path.unlink(missing_ok=True)
            covered = {
                int(row["scene_idx"])
                for row in [*existing, *failures]
            }
            if covered == expected_scene_indices:
                content_id = str(visual["content_id"])
                existing = _minimal_description_records(existing, path)
                records_by_content[content_id] = existing
                failures_by_content[content_id] = failures
                continue
        pending.append((visual, scene_rows))

    with tqdm(
        total=len(visual_rows),
        initial=len(records_by_content),
        desc="Description scenes",
        unit="content",
    ) as progress:
        if pending:
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                for visual, scene_rows in pending:
                    generated = generate([row["task"] for row in scene_rows], None)
                    records: list[dict[str, Any]] = []
                    failures: list[dict[str, Any]] = []
                    for row in scene_rows:
                        description = generated[row["task"].task_id].strip()
                        common = {
                            "content_id": visual["content_id"],
                            "scene_idx": row["scene_idx"],
                            "keyframes": row["keyframes"],
                        }
                        if not description:
                            failures.append({
                                "schema_version": "description-generation-failure/v1",
                                **common,
                                "failure_kind": "empty_response",
                                "error": "model produced an empty description",
                            })
                            continue
                        records.append({
                            "schema_version": SCENE_SCHEMA_VERSION,
                            **common,
                            "description": description,
                        })
                    path = context.description_scene_dir / f"{visual['content_id']}.jsonl"
                    failure_path = context.description_failure_dir / f"{visual['content_id']}.jsonl"
                    records.sort(key=lambda row: int(row["scene_idx"]))
                    failures.sort(key=lambda row: int(row["scene_idx"]))
                    write_jsonl(path, records)
                    _write_failure_jsonl(failure_path, failures)
                    content_id = str(visual["content_id"])
                    records_by_content[content_id] = records
                    failures_by_content[content_id] = failures
                    video_name = names.get(
                        content_id,
                        f"{visual['content_id']}.mp4",
                    )
                    for message in scene_messages(
                        video_name, records, arm="description"
                    ):
                        _write_progress(progress, message)
                    for failure in failures:
                        _write_progress(
                            progress,
                            f"[SKIPPED] {video_name} | description scene "
                            f"#{int(failure['scene_idx']):03d} | {failure['error']}",
                        )
                    _complete_content_progress(progress)
    failures = [
        record
        for visual in visual_rows
        for record in failures_by_content[str(visual["content_id"])]
    ]
    return _result(
        "extract-description-scenes",
        content_count=len(visual_rows),
        failure_count=len(failures),
    )


def summarize_description(
    context: RunContext,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    settings = context.config["extraction"]["description"]
    retry_settings = context.config["extraction"]["summary_retry"]
    prompt_path = context.config_path("extraction", "description", "summary_prompt")
    template = prompt_path.read_text(encoding="utf-8")
    model_path = context.path("models", "qwen")
    visual_rows = _visual_rows(context)
    names = _video_name_map(context)
    if not context.description_scene_dir.is_dir():
        raise ExtractionStepError(
            f"missing description scene directory: {context.description_scene_dir}"
        )
    scene_paths = [
        context.description_scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows
    ]
    for path in scene_paths:
        _require_file(path, "description scene output")
    documents_by_content: dict[str, dict[str, Any]] = {}
    pending: list[tuple[list[dict[str, Any]], Path, str]] = []
    tasks: list[QwenGenerationTask] = []
    empty_scene_files = 0
    for scene_path in scene_paths:
        records = read_jsonl(scene_path)
        if not records:
            empty_scene_files += 1
            continue
        records = _minimal_description_records(records, scene_path)
        if any(row.get("schema_version") != SCENE_SCHEMA_VERSION for row in records):
            raise ExtractionStepError(f"invalid description scene file: {scene_path}")
        output_path = context.description_summary_dir / f"{records[0]['content_id']}.json"
        if output_path.is_file() and not force:
            content_id = str(records[0]["content_id"])
            documents_by_content[content_id] = _reuse_summary_document(
                output_path,
                schema_version=SUMMARY_SCHEMA_VERSION,
                content_id=content_id,
                arm="description",
                scene_count=len(records),
            )
            continue
        prompt = description_summary_prompt(template, records)
        task_id = str(records[0]["content_id"])
        tasks.append(
            QwenGenerationTask(
                task_id=task_id,
                image_paths=(),
                prompt=prompt,
                max_new_tokens=int(settings["summary_max_new_tokens"]),
            )
        )
        pending.append((records, output_path, task_id))

    pending_by_task = {
        task_id: (records, output_path)
        for records, output_path, task_id in pending
    }
    with tqdm(
        total=len(visual_rows),
        initial=len(documents_by_content) + empty_scene_files,
        desc="Description summaries",
        unit="content",
    ) as progress:
        def complete_description_summary(task_id: str, text: str) -> None:
            records, output_path = pending_by_task[task_id]
            sections = validate_description_summary(text)
            summary = serialize_summary_sections(sections)
            content_id = str(records[0]["content_id"])
            document = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "content_id": content_id,
                "arm": "description",
                "status": "complete",
                "sections": sections,
                "text": summary,
                "scene_count": len(records),
            }
            write_json(output_path, document)
            documents_by_content[content_id] = document
            _write_progress(
                progress,
                summary_message(
                    names.get(content_id, f"{content_id}.mp4"),
                    arm="description",
                    scene_count=len(records),
                    text=summary,
                ),
            )
            _complete_content_progress(progress)

        retry_count = 0
        if tasks:
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                retry_count = _generate_summaries_with_retry(
                    generate,
                    tasks,
                    complete_description_summary,
                    retry_settings,
                )
    return _result(
        "summarize-description",
        content_count=len(documents_by_content),
        failure_count=empty_scene_files,
        retry_count=retry_count,
    )


STEP_HANDLERS: dict[str, Callable[[RunContext], dict[str, Any]]] = {
    "prepare-input-data": prepare_input_data,
    "extract-graph-scenes": extract_graph_scenes,
    "summarize-graph": summarize_graph,
    "extract-description-scenes": extract_description_scenes,
    "summarize-description": summarize_description,
}
