from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from extraction.backends import GeminiGenerationOutcome, GeminiWorkerPool
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.descriptions import (
    SCENE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    description_summary_prompt,
    validate_summary as validate_description_summary,
)
from extraction.errors import ExtractionStepError
from extraction.monitoring import (
    graph_skip_message,
    scene_messages,
    summary_message,
)
from extraction.preparation import prepare_input_data
from extraction.semantic_graph import (
    SCENE_EXTRACTION_PROMPT,
    SUMMARY_SCHEMA_VERSION as GRAPH_SUMMARY_SCHEMA_VERSION,
    graph_semantic_warnings,
    graph_summary_prompt,
    parse_or_repair_graph,
    validate_summary as validate_graph_summary,
)
from extraction.summary_validation import (
    serialize_summary_sections,
)
from extraction.summary_executor import (
    generate_summaries_with_retry,
    qwen_generator,
    reuse_summary_document,
)
from extraction.step_support import (
    complete_content_progress as _complete_content_progress,
    minimal_description_records as _minimal_description_records,
    minimal_graph_failures as _minimal_graph_failures,
    minimal_graph_records as _minimal_graph_records,
    require_file as _require_file,
    result as _result,
    scene_generation_rows as _scene_generation_rows,
    video_name_map as _video_name_map,
    visual_rows as _visual_rows,
    write_failure_jsonl as _write_failure_jsonl,
    write_progress as _write_progress,
)
from pipeline_runtime import (
    RunContext,
    read_jsonl,
    write_json,
    write_jsonl,
)


GRAPH_SOURCES = ("qwen", "gemini")


def graph_stage_name(stage: str, source: str) -> str:
    if source not in GRAPH_SOURCES:
        raise ValueError(f"unsupported graph source: {source}")
    return f"{stage}-{source}"


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
            with qwen_generator(model_path=model_path, gpus=gpus) as generate:
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
            documents_by_content[content_id] = reuse_summary_document(
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
            with qwen_generator(model_path=model_path, gpus=gpus) as generate:
                retry_count = generate_summaries_with_retry(
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
            with qwen_generator(model_path=model_path, gpus=gpus) as generate:
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
            documents_by_content[content_id] = reuse_summary_document(
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
            with qwen_generator(model_path=model_path, gpus=gpus) as generate:
                retry_count = generate_summaries_with_retry(
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
