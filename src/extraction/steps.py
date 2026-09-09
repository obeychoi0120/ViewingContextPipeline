from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Callable

from tqdm import tqdm

from extraction.backends import GeminiGenerationOutcome as GeminiGenerationOutcome, GeminiWorkerPool
from extraction.descriptions import (
    SCENE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    description_summary_prompt,
    validate_summary as validate_description_summary,
)
from extraction.errors import ExtractionStepError
from extraction.preparation import prepare_input_data
from extraction.qwen_runtime import QwenRuntimeLog
from extraction.recovery import has_pending_recovery, penalty_schedule
from extraction.structured_output import GRAPH_JSON_SCHEMA, SUMMARY_GRAMMAR
from extraction.progress import InferenceProgress
from extraction.scene_executor import run_qwen_scenes, run_gemini_scenes
from extraction.semantic_graph import (
    SUMMARY_SCHEMA_VERSION as GRAPH_SUMMARY_SCHEMA_VERSION,
    graph_summary_prompt,
    validate_summary as validate_graph_summary,
)
from extraction.summary_executor import (
    SummaryBranch,
    run_summary_stage,
    qwen_generator,
)
from extraction.step_support import (
    minimal_description_records as _minimal_description_records,
    minimal_graph_failures as _minimal_graph_failures,
    minimal_graph_records as _minimal_graph_records,
    require_file as _require_file,
    result as _result,
    scene_generation_rows as _scene_generation_rows,
    video_name_map as _video_name_map,
    visual_rows as _visual_rows,
    write_progress as _write_progress,
    write_failure_jsonl,
    restore_scene_checkpoint,
)
from pipeline_runtime import (
    RunContext,
    read_jsonl,
)


GRAPH_SOURCES = ("qwen", "gemini")


def graph_stage_name(stage: str, source: str) -> str:
    if source not in GRAPH_SOURCES:
        raise ValueError(f"unsupported graph source: {source}")
    return f"{stage}-{source}"


def _summary_generation_settings(context: RunContext) -> dict[str, Any]:
    extraction = context.config["extraction"]
    settings: dict[str, Any] = {
        "repetition_penalty": penalty_schedule(extraction["summary_repetition_penalty"])[0],
        "structured_output": {"grammar": SUMMARY_GRAMMAR},
    }
    if bool(extraction["greedy_decoding"]):
        return settings
    sampling = extraction["summary_sampling"]
    settings.update(
        {
            "do_sample": True,
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "top_k": int(sampling["top_k"]),
        }
    )
    return settings


def _graph_scene_work(context, visual_rows, prompt, settings, scene_dir, failure_dir, model, force):
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
            repetition_penalty=(
                penalty_schedule(context.config["extraction"]["graph_repetition_penalty"])[0]
                if model == "qwen"
                else 1.0
            ),
        )
        if model == "qwen":
            for row in scene_rows:
                row["task"] = replace(row["task"], structured_output={"json": GRAPH_JSON_SCHEMA})
        if not force:
            restore_scene_checkpoint(path, failure_path)
            existing = _minimal_graph_records(read_jsonl(path), path) if path.is_file() else []
            failures = read_jsonl(failure_path) if failure_path.is_file() else []
            if model == "qwen":
                normalized = _minimal_graph_failures(failures)
                if normalized != failures:
                    write_failure_jsonl(failure_path, normalized)
                failures = normalized
            expected = {int(row["scene_idx"]): row["keyframes"] for row in scene_rows}
            cached = [*existing, *failures]
            indices = [int(row["scene_idx"]) for row in cached]
            if len(indices) != len(set(indices)) or any(
                int(row["scene_idx"]) not in expected
                or row.get("keyframes") != expected[int(row["scene_idx"])]
                for row in cached
            ):
                raise ExtractionStepError(
                    f"incompatible cached scene indices/keyframes: {path}; "
                    "resuming requires unchanged inputs and settings; "
                    "use --force or a new run_id for changed inputs"
                )
            pending_indices = {int(row["scene_idx"]) for row in scene_rows
                               if has_pending_recovery(scene_dir / ".recovery", row["task"].task_id)}
            existing = [row for row in existing if int(row["scene_idx"]) not in pending_indices]
            content_id = str(visual["content_id"])
            records_by_content[content_id] = existing
            failures_by_content[content_id] = failures
            successful_indices = {int(row["scene_idx"]) for row in existing}
            pending_rows = [
                row for row in scene_rows if int(row["scene_idx"]) not in successful_indices
            ]
            if pending_rows:
                pending.append((visual, pending_rows))
            elif not failures:
                failure_path.unlink(missing_ok=True)
            continue
        pending.append((visual, scene_rows))

    return records_by_content, failures_by_content, pending


def _description_scene_work(context, visual_rows, prompt, settings, force):
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
            repetition_penalty=penalty_schedule(
                context.config["extraction"]["description_repetition_penalty"]
            )[0],
        )
        if not force:
            restore_scene_checkpoint(path, failure_path)
            existing = _minimal_description_records(read_jsonl(path), path) if path.is_file() else []
            failures = read_jsonl(failure_path) if failure_path.is_file() else []
            expected = {int(row["scene_idx"]): row["keyframes"] for row in scene_rows}
            cached = [*existing, *failures]
            indices = [int(row["scene_idx"]) for row in cached]
            if len(indices) != len(set(indices)) or any(
                int(row["scene_idx"]) not in expected
                or row.get("keyframes") != expected[int(row["scene_idx"])]
                for row in cached
            ):
                raise ExtractionStepError(f"incompatible cached scene indices/keyframes: {path}; use --force")
            pending_indices = {int(row["scene_idx"]) for row in scene_rows
                               if has_pending_recovery(context.description_scene_dir / ".recovery",
                                                       row["task"].task_id)}
            existing = [row for row in existing if int(row["scene_idx"]) not in pending_indices]
            content_id = str(visual["content_id"])
            records_by_content[content_id] = existing
            failures_by_content[content_id] = failures
            successful_indices = {int(row["scene_idx"]) for row in existing}
            pending_rows = [row for row in scene_rows if int(row["scene_idx"]) not in successful_indices]
            if pending_rows:
                pending.append((visual, pending_rows))
            if not failures:
                failure_path.unlink(missing_ok=True)
            continue
        pending.append((visual, scene_rows))

    return records_by_content, failures_by_content, pending


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
    prompt_path = context.config_path("extraction", "graph", "scene_prompt")
    prompt = prompt_path.read_text(encoding="utf-8")
    model_path: Path | None = None
    if model == "qwen":
        model_path = context.path("models", "qwen")
    visual_rows = _visual_rows(context)
    names = _video_name_map(context)
    scene_dir = context.graph_scene_dir(model)
    failure_dir = context.graph_failure_dir(model)
    records_by_content, failures_by_content, pending = _graph_scene_work(
        context,
        visual_rows,
        prompt,
        settings,
        scene_dir,
        failure_dir,
        model,
        force,
    )
    with InferenceProgress(
        total=sum(len(rows) for _, rows in pending),
        reused=sum(len(rows) for rows in records_by_content.values()),
        desc=f"Graph scenes ({model})",
        unit="scene", progress_factory=tqdm,
    ) as progress:
        if model == "gemini" and not force:
            _write_progress(
                progress,
                f"[Gemini] processing {sum(len(rows) for _, rows in pending)} failed or "
                f"missing scenes across {len(pending)} contents; successful scenes are reused",
            )
        if model == "qwen":
            completed, failed = run_qwen_scenes(
                pending,
                scene_dir=scene_dir,
                failure_dir=failure_dir,
                model_path=model_path,
                gpus=gpus,
                generator_factory=qwen_generator,
                names=names,
                progress=progress,
                arm="graph",
                source=model,
                existing_records=records_by_content,
                existing_failures=failures_by_content,
                qwen_options=context.config["extraction"].get("qwen"),
                image_limit=context.config["extraction"]["visual_evidence"]["num_keyframes"],
                runtime=QwenRuntimeLog(context.run_root, stage),
                penalties=context.config["extraction"]["graph_repetition_penalty"],
                force=force,
            )
            records_by_content.update(completed)
            failures_by_content.update(failed)
        elif pending:
            gemini = context.config["models"]["gemini"]
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
            run_gemini_scenes(
                pending,
                pool=pool,
                records_by_content=records_by_content,
                failures_by_content=failures_by_content,
                scene_dir=scene_dir,
                failure_dir=failure_dir,
                force=force,
                names=names,
                progress=progress,
                identity=gemini,
            )
    failures = [
        record
        for visual in visual_rows
        for record in failures_by_content[str(visual["content_id"])]
    ]
    return _result(stage, content_count=len(visual_rows), failure_count=len(failures))


def summarize_graph(
    context: RunContext,
    *,
    source: str,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    stage = graph_stage_name("summarize-graph", source)
    context.initialize()
    settings = context.config["extraction"]["graph"]
    generation = _summary_generation_settings(context)
    template = context.config_path("extraction", "graph", "summary_prompt").read_text(
        encoding="utf-8"
    )
    model_path = context.path("models", "qwen")
    visuals = _visual_rows(context)
    names = _video_name_map(context)
    scene_dir = context.graph_scene_dir(source)
    if not scene_dir.is_dir():
        raise ExtractionStepError(f"missing graph scene directory: {scene_dir}")
    paths = [scene_dir / f"{row['content_id']}.jsonl" for row in visuals]
    if not all(path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise ExtractionStepError(f"missing graph scene output: {missing}")
    if source == "gemini" and not force and context.config["schema_version"] == "viewing-context-config/v4":
        from extraction.summary_executor import reuse_summary_document
        for path in paths:
            output = context.graph_summary_dir(source) / f"{path.stem}.json"
            if output.is_file():
                reuse_summary_document(
                    output, schema_version=GRAPH_SUMMARY_SCHEMA_VERSION,
                    content_id=path.stem, arm="graph_gemini",
                    scene_count=None if read_jsonl(path) else 0,
                )
    return run_summary_stage(
        SummaryBranch(
            stage=stage,
            arm=f"graph_{source}",
            label="graph",
            schema_version=GRAPH_SUMMARY_SCHEMA_VERSION,
            summary_dir=context.graph_summary_dir(source),
            failure_dir=context.graph_summary_failure_dir(source),
            normalize_records=_minimal_graph_records,
            content_id=lambda records, path: path.stem,
            build_prompt=graph_summary_prompt,
            validate=validate_graph_summary,
            allow_missing=(source == "gemini" and context.config["schema_version"] == "viewing-context-config/v4"),
        ),
        scene_paths=paths,
        template=template,
        max_new_tokens=int(settings["summary_max_new_tokens"]),
        generation=generation,
        model_path=model_path,
        gpus=gpus,
        force=force,
        names=names,
        generator_factory=qwen_generator,
        progress_factory=tqdm,
        qwen_options=context.config["extraction"].get("qwen"),
        image_limit=context.config["extraction"]["visual_evidence"]["num_keyframes"],
        runtime=QwenRuntimeLog(context.run_root, stage),
        penalties=context.config["extraction"]["summary_repetition_penalty"],
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
    records_by_content, failures_by_content, pending = _description_scene_work(
        context,
        visual_rows,
        prompt,
        settings,
        force,
    )
    with InferenceProgress(
        total=sum(len(rows) for _, rows in pending),
        reused=sum(len(rows) for rows in records_by_content.values()),
        desc="Description scenes",
        unit="scene", progress_factory=tqdm,
    ) as progress:
        completed, failed = run_qwen_scenes(
            pending,
            scene_dir=context.description_scene_dir,
            failure_dir=context.description_failure_dir,
            model_path=model_path,
            gpus=gpus,
            generator_factory=qwen_generator,
            names=names,
            progress=progress,
            arm="description",
            existing_records=records_by_content,
            existing_failures=failures_by_content,
            qwen_options=context.config["extraction"].get("qwen"),
            image_limit=context.config["extraction"]["visual_evidence"]["num_keyframes"],
            runtime=QwenRuntimeLog(context.run_root, "extract-description-scenes"),
            penalties=context.config["extraction"]["description_repetition_penalty"],
            force=force,
        )
        records_by_content.update(completed)
        failures_by_content.update(failed)
    failures = [
        record
        for visual in visual_rows
        for record in failures_by_content[str(visual["content_id"])]
    ]
    return _result(
        "extract-description-scenes", content_count=len(visual_rows), failure_count=len(failures)
    )


def _description_summary_records(records, path):
    records = _minimal_description_records(records, path)
    if any(row.get("schema_version") != SCENE_SCHEMA_VERSION for row in records):
        raise ExtractionStepError(f"invalid description scene file: {path}")
    return records


def summarize_description(
    context: RunContext,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    settings = context.config["extraction"]["description"]
    generation = _summary_generation_settings(context)
    template = context.config_path("extraction", "description", "summary_prompt").read_text(
        encoding="utf-8"
    )
    model_path = context.path("models", "qwen")
    visuals = _visual_rows(context)
    names = _video_name_map(context)
    if not context.description_scene_dir.is_dir():
        raise ExtractionStepError(
            f"missing description scene directory: {context.description_scene_dir}"
        )
    paths = [context.description_scene_dir / f"{row['content_id']}.jsonl" for row in visuals]
    for path in paths:
        _require_file(path, "description scene output")
    return run_summary_stage(
        SummaryBranch(
            stage="summarize-description",
            arm="description",
            label="description",
            schema_version=SUMMARY_SCHEMA_VERSION,
            summary_dir=context.description_summary_dir,
            failure_dir=context.description_summary_failure_dir,
            normalize_records=_description_summary_records,
            content_id=lambda records, path: str(records[0]["content_id"]),
            build_prompt=description_summary_prompt,
            validate=validate_description_summary,
        ),
        scene_paths=paths,
        template=template,
        max_new_tokens=int(settings["summary_max_new_tokens"]),
        generation=generation,
        model_path=model_path,
        gpus=gpus,
        force=force,
        names=names,
        generator_factory=qwen_generator,
        progress_factory=tqdm,
        qwen_options=context.config["extraction"].get("qwen"),
        image_limit=context.config["extraction"]["visual_evidence"]["num_keyframes"],
        runtime=QwenRuntimeLog(context.run_root, "summarize-description"),
        penalties=context.config["extraction"]["summary_repetition_penalty"],
    )


STEP_HANDLERS: dict[str, Callable[[RunContext], dict[str, Any]]] = {
    "prepare-input-data": prepare_input_data,
    "extract-graph-scenes": extract_graph_scenes,
    "summarize-graph": summarize_graph,
    "extract-description-scenes": extract_description_scenes,
    "summarize-description": summarize_description,
}
