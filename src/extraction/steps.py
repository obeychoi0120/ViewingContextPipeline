from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm import tqdm

from extraction.backends import GeminiGenerationOutcome, GeminiWorkerPool, QwenBackend
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.data_preparation.microlens import prepare_catalog
from extraction.descriptions import (
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
    GRAPH_SOFT_VALIDATION_VERSION,
    GRAPH_SUMMARY_SCHEMA,
    JSON_REPAIR_VERSION,
    SCENE_EXTRACTION_PROMPT,
    SCENE_SCHEMA_VERSION as GRAPH_SCENE_SCHEMA,
    graph_summary_prompt,
    graph_soft_warnings,
    parse_or_repair_graph,
    taxonomy_contract,
    validate_summary as validate_graph_summary,
)
from viewing_context_pipeline.runtime import (
    RunContext,
    directory_fingerprint,
    file_fingerprint,
    fingerprint,
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
            )
            results[task.task_id] = text
            if on_task_complete is not None:
                on_task_complete(task.task_id, text)
        return results

    yield generate


def _completed_progress(description: str, total: int) -> None:
    with tqdm(total=total, initial=total, desc=description, unit="content"):
        pass


def _write_progress(progress: tqdm, message: str) -> None:
    tqdm.write(message, file=progress.fp)


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


def _current(
    context: RunContext,
    stage: str,
    sources: dict[str, str],
    required: list[Path],
) -> dict[str, Any] | None:
    path = context.stage_manifest(stage)
    if not path.is_file() or not all(
        item.is_file() or (item.is_dir() and any(item.iterdir())) for item in required
    ):
        return None
    value = read_json(path)
    if (
        value.get("schema_version") == "step-manifest/v1"
        and value.get("status") == "complete"
        and value.get("source_fingerprints") == sources
    ):
        return value
    return None


def _require_stage(context: RunContext, stage: str) -> dict[str, Any]:
    path = context.stage_manifest(stage)
    if not path.is_file():
        raise ExtractionStepError(f"required stage is incomplete: {stage}")
    value = read_json(path)
    if value.get("schema_version") != "step-manifest/v1" or value.get("status") != "complete":
        raise ExtractionStepError(f"invalid stage manifest: {path}")
    return value


def _write_stage(
    context: RunContext,
    stage: str,
    *,
    source_fingerprints: dict[str, str],
    output_fingerprint: str,
    content_count: int | None = None,
) -> dict[str, Any]:
    manifest_path = context.stage_manifest(stage)
    previous_fingerprint = (
        read_json(manifest_path).get("output_fingerprint") if manifest_path.is_file() else None
    )
    document: dict[str, Any] = {
        "schema_version": "step-manifest/v1",
        "run_id": context.run_id,
        "stage": stage,
        "status": "complete",
        "source_fingerprints": source_fingerprints,
        "output_fingerprint": output_fingerprint,
    }
    if content_count is not None:
        document["content_count"] = content_count
    write_json(manifest_path, document)
    if previous_fingerprint is not None and previous_fingerprint != output_fingerprint:
        from viewing_context_pipeline.pipeline import invalidate_descendants

        invalidate_descendants(context, {stage})
    return document


def prepare_input_data(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    cohort = _require_stage(context, "prepare-cohort")
    sources = {"prepare-cohort": cohort["output_fingerprint"]}
    sources.update({
        "titles": file_fingerprint(context.path("data", "titles_csv")),
        "tags": file_fingerprint(context.path("data", "tags_csv")),
        "settings": fingerprint(context.config["extraction"]["visual_evidence"]),
    })
    if not force:
        current = _current(context, "prepare-input-data", sources, [context.visual_manifest])
        if current is not None:
            return current
    catalog = read_jsonl(context.cohort_dir / "catalog.jsonl")
    settings = context.config["extraction"]["visual_evidence"]
    result = prepare_catalog(
        catalog,
        titles_csv=context.path("data", "titles_csv"),
        tags_csv=context.path("data", "tags_csv"),
        assets_root=context.cohort_dir / "source_assets",
        output_root=context.run_root,
        image_size=tuple(settings["image_resolution"]),
        force=force,
    )
    if result["failed"] or result["succeeded"] != len(catalog):
        raise ExtractionStepError(f"visual evidence preparation is incomplete: {result}")
    rows: list[dict[str, Any]] = []
    for item in catalog:
        content_id = str(item["content_id"])
        frames_dir = context.evidence_dir / "resized_keyframes" / content_id
        timestamp = context.cohort_dir / "source_assets" / content_id / "assets" / "timestamp_fixed_30s.json"
        frames = sorted(
            path for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ) if frames_dir.is_dir() else []
        if not frames or not timestamp.is_file():
            raise ExtractionStepError(f"missing visual evidence for {content_id}")
        scenes = json.loads(timestamp.read_text(encoding="utf-8"))
        evidence_fp = fingerprint({
            "timestamp": file_fingerprint(timestamp),
            "frames": [file_fingerprint(path) for path in frames],
        })
        rows.append({
            "schema_version": "visual-manifest/v1",
            "content_id": content_id,
            "item_id": str(item["item_id"]),
            "frames_dir": str(frames_dir),
            "timestamp_json": str(timestamp),
            "frame_count": len(frames),
            "scene_count": len(scenes),
            "evidence_fingerprint": evidence_fp,
        })
    write_jsonl(context.visual_manifest, rows)
    output_fp = fingerprint(rows)
    return _write_stage(
        context,
        "prepare-input-data",
        source_fingerprints=sources,
        output_fingerprint=output_fp,
        content_count=len(rows),
    )


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
    evidence = _require_stage(context, "prepare-input-data")
    settings = context.config["extraction"]["graph"]
    prompt = SCENE_EXTRACTION_PROMPT
    taxonomy_fp = fingerprint(taxonomy_contract())
    prompt_fp = fingerprint(prompt)
    repair_fp = fingerprint({"version": JSON_REPAIR_VERSION})
    validation_fp = fingerprint({"version": GRAPH_SOFT_VALIDATION_VERSION})
    model_path: Path | None = None
    if model == "qwen":
        model_path = context.path("models", "qwen")
        model_id = model_path.name
        model_fp = fingerprint({
            "source": model,
            "path": str(model_path),
            "files": directory_fingerprint(model_path),
            "max_new_tokens": settings["scene_max_new_tokens"],
            "do_sample": settings["do_sample"],
        })
    else:
        gemini = context.config["models"]["gemini"]
        model_id = str(gemini["model_id"])
        model_fp = fingerprint({
            "source": model,
            "project_id": gemini["project_id"],
            "location": gemini["location"],
            "model_id": model_id,
            "max_new_tokens": settings["scene_max_new_tokens"],
            "temperature": 0.0,
        })
    sources = {
        "prepare-input-data": evidence["output_fingerprint"],
        "taxonomy": taxonomy_fp,
        "prompt": prompt_fp,
        "extractor": model_fp,
        "repair": repair_fp,
        "soft_validation": validation_fp,
    }
    visual_rows = read_jsonl(context.visual_manifest)
    names = _video_name_map(context)
    scene_dir = context.graph_scene_dir(model)
    failure_dir = context.graph_failure_dir(model)
    expected_outputs = [
        path
        for row in visual_rows
        for path in (
            scene_dir / f"{row['content_id']}.jsonl",
            failure_dir / f"{row['content_id']}.jsonl",
        )
    ]
    if not force:
        current = _current(context, stage, sources, expected_outputs)
        if current is not None:
            _completed_progress(f"Graph scenes ({model})", len(visual_rows))
            return current
    context.stage_manifest(stage).unlink(missing_ok=True)
    from viewing_context_pipeline.pipeline import invalidate_descendants

    invalidate_descendants(context, {stage})
    records_by_content: dict[str, list[dict[str, Any]]] = {}
    failures_by_content: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    for visual in visual_rows:
        input_fp = fingerprint({
            "evidence": visual["evidence_fingerprint"],
            "taxonomy": taxonomy_fp,
            "prompt": prompt_fp,
            "extractor": model_fp,
            "graph_source": model,
        })
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
            covered = {
                int(row["scene_idx"])
                for row in [*existing, *failures]
                if row.get("input_fingerprint") == input_fp
            }
            if covered == expected_scene_indices and all(
                row.get("input_fingerprint") == input_fp
                for row in [*existing, *failures]
            ):
                migrated = [
                    {
                        **row,
                        "parse_status": row.get("parse_status", "legacy_parser"),
                        "repair_fingerprint": repair_fp,
                        "validation_fingerprint": validation_fp,
                        "validation_warnings": graph_soft_warnings(
                            row.get("graph", {})
                        ),
                    }
                    for row in existing
                ]
                write_jsonl(path, migrated)
                write_jsonl(failure_path, failures)
                content_id = str(visual["content_id"])
                records_by_content[content_id] = migrated
                failures_by_content[content_id] = failures
                continue
        pending.append((visual, input_fp, scene_rows))

    with tqdm(
        total=len(visual_rows),
        initial=len(records_by_content),
        desc=f"Graph scenes ({model})",
        unit="content",
    ) as progress:
        def complete_content(
            visual: dict[str, Any],
            input_fp: str,
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
                common = {
                    "content_id": visual["content_id"],
                    "scene_idx": row["scene_idx"],
                    "scene_start_seconds": row["scene_start_seconds"],
                    "scene_end_seconds": row["scene_end_seconds"],
                    "keyframes": row["keyframes"],
                    "image_paths": row["image_paths"],
                    "graph_source": model,
                    "extractor_model_id": model_id,
                    "extractor_model_fingerprint": model_fp,
                    "taxonomy_fingerprint": taxonomy_fp,
                    "prompt_fingerprint": prompt_fp,
                    "evidence_fingerprint": visual["evidence_fingerprint"],
                    "input_fingerprint": input_fp,
                    "repair_fingerprint": repair_fp,
                    "validation_fingerprint": validation_fp,
                }
                if result is not None and result.graph is not None:
                    records.append({
                        "schema_version": GRAPH_SCENE_SCHEMA,
                        **common,
                        "parse_status": result.status,
                        "validation_warnings": graph_soft_warnings(result.graph),
                        "graph": result.graph,
                    })
                else:
                    failures.append({
                        "schema_version": "semantic-graph-repair-failure/v1",
                        **common,
                        "parse_status": "failed",
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
            write_jsonl(failure_path, failures)
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
            progress.update(1)

        if pending and model == "qwen":
            assert model_path is not None
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                for visual, input_fp, scene_rows in pending:
                    responses = generate([row["task"] for row in scene_rows], None)
                    complete_content(
                        visual,
                        input_fp,
                        scene_rows,
                        {task_id: (text, None) for task_id, text in responses.items()},
                    )
        elif pending:
            gemini = context.config["models"]["gemini"]
            task_context: dict[str, tuple[str, dict[str, Any], str, list[dict[str, Any]]]] = {}
            generated_by_content: dict[str, dict[str, tuple[str, str | None]]] = {}
            tasks: list[QwenGenerationTask] = []
            for visual, input_fp, scene_rows in pending:
                content_id = str(visual["content_id"])
                generated_by_content[content_id] = {}
                for row in scene_rows:
                    task = row["task"]
                    tasks.append(task)
                    task_context[task.task_id] = (
                        content_id,
                        visual,
                        input_fp,
                        scene_rows,
                    )

            def complete_gemini_scene(outcome: GeminiGenerationOutcome) -> None:
                content_id, visual, input_fp, scene_rows = task_context[outcome.task_id]
                responses = generated_by_content[content_id]
                responses[outcome.task_id] = (outcome.text, outcome.error)
                if len(responses) == len(scene_rows):
                    complete_content(visual, input_fp, scene_rows, responses)

            pool = GeminiWorkerPool(
                int(settings["gemini_concurrency"]),
                project_id=str(gemini["project_id"]),
                location=str(gemini["location"]),
                model_id=str(gemini["model_id"]),
            )
            pool.generate(tasks, complete_gemini_scene)
    outputs = [
        record
        for visual in visual_rows
        for record in records_by_content[str(visual["content_id"])]
    ]
    failures = [
        record
        for visual in visual_rows
        for record in failures_by_content[str(visual["content_id"])]
    ]
    if failures:
        failed_contents = len({str(row["content_id"]) for row in failures})
        raise ExtractionStepError(
            f"Graph extraction failed for {len(failures)} scene(s) across "
            f"{failed_contents} content(s); see {failure_dir}"
        )
    output_fp = fingerprint({"scenes": outputs, "failures": failures})
    return _write_stage(
        context,
        stage,
        source_fingerprints=sources,
        output_fingerprint=output_fp,
        content_count=len(visual_rows),
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
    scene_stage = graph_stage_name("extract-graph-scenes", source)
    context.initialize()
    scenes_manifest = _require_stage(context, scene_stage)
    settings = context.config["extraction"]["graph"]
    prompt_path = context.config_path("extraction", "graph", "summary_prompt")
    template = prompt_path.read_text(encoding="utf-8")
    prompt_fp = file_fingerprint(prompt_path)
    model_path = context.path("models", "qwen")
    model_fp = fingerprint({
        "path": str(model_path),
        "files": directory_fingerprint(model_path),
        "max_new_tokens": settings["summary_max_new_tokens"],
        "do_sample": settings["do_sample"],
    })
    sources = {
        scene_stage: scenes_manifest["output_fingerprint"],
        "prompt": prompt_fp,
        "summary_model": model_fp,
    }
    visual_rows = read_jsonl(context.visual_manifest)
    names = _video_name_map(context)
    scene_dir = context.graph_scene_dir(source)
    summary_dir = context.graph_summary_dir(source)
    scene_paths = [scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows]
    if not all(path.is_file() for path in scene_paths):
        raise ExtractionStepError("graph scene outputs are incomplete")
    expected_outputs = [summary_dir / f"{row['content_id']}.json" for row in visual_rows]
    if not force:
        current = _current(context, stage, sources, expected_outputs)
        if current is not None:
            _completed_progress(f"Graph summaries ({source})", len(visual_rows))
            return current
    documents_by_content: dict[str, dict[str, Any]] = {}
    pending: list[tuple[list[dict[str, Any]], str, Path, Path, str]] = []
    tasks: list[QwenGenerationTask] = []
    for scene_path in scene_paths:
        records = read_jsonl(scene_path)
        if not records or any(row.get("schema_version") != GRAPH_SCENE_SCHEMA for row in records):
            raise ExtractionStepError(f"invalid graph scene file: {scene_path}")
        scene_fp = file_fingerprint(scene_path)
        input_fp = fingerprint({"scenes": scene_fp, "prompt": prompt_fp, "model": model_fp})
        if any(row.get("graph_source") != source for row in records):
            raise ExtractionStepError(f"graph source mismatch in {scene_path}")
        output_path = summary_dir / f"{records[0]['content_id']}.json"
        if output_path.is_file() and not force:
            existing = read_json(output_path)
            if existing.get("input_fingerprint") == input_fp:
                documents_by_content[str(records[0]["content_id"])] = existing
                continue
        prompt = graph_summary_prompt(template, records)
        task_id = str(records[0]["content_id"])
        tasks.append(
            QwenGenerationTask(
                task_id=task_id,
                image_paths=(),
                prompt=prompt,
                max_new_tokens=int(settings["summary_max_new_tokens"]),
            )
        )
        pending.append((records, input_fp, output_path, scene_path, task_id))

    pending_by_task = {
        task_id: (records, input_fp, output_path, scene_path)
        for records, input_fp, output_path, scene_path, task_id in pending
    }
    with tqdm(
        total=len(visual_rows),
        initial=len(documents_by_content),
        desc=f"Graph summaries ({source})",
        unit="content",
    ) as progress:
        def complete_graph_summary(task_id: str, text: str) -> None:
            records, input_fp, output_path, scene_path = pending_by_task[task_id]
            summary = validate_graph_summary(text)
            content_id = str(records[0]["content_id"])
            document = {
                "schema_version": GRAPH_SUMMARY_SCHEMA,
                "content_id": content_id,
                "arm": "graph",
                "graph_source": source,
                "status": "complete",
                "text": summary,
                "scene_count": len(records),
                "evidence_fingerprint": records[0]["evidence_fingerprint"],
                "taxonomy_fingerprint": records[0]["taxonomy_fingerprint"],
                "scene_prompt_fingerprint": records[0]["prompt_fingerprint"],
                "extractor_model_id": records[0]["extractor_model_id"],
                "extractor_model_fingerprint": records[0][
                    "extractor_model_fingerprint"
                ],
                "summary_model": "qwen",
                "summary_model_id": model_path.name,
                "scene_graph_path": scene_path.relative_to(context.run_root).as_posix(),
                "scene_graph_fingerprint": file_fingerprint(scene_path),
                "summary_prompt_fingerprint": prompt_fp,
                "summary_model_fingerprint": model_fp,
                "input_fingerprint": input_fp,
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
            progress.update(1)

        if tasks:
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                generate(tasks, complete_graph_summary)
    documents = [
        documents_by_content[str(row["content_id"])] for row in visual_rows
    ]
    output_fp = fingerprint(documents)
    return _write_stage(
        context,
        stage,
        source_fingerprints=sources,
        output_fingerprint=output_fp,
        content_count=len(documents),
    )


def extract_description_scenes(
    context: RunContext,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    evidence = _require_stage(context, "prepare-input-data")
    settings = context.config["extraction"]["description"]
    prompt_path = context.config_path("extraction", "description", "scene_prompt")
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_fp = file_fingerprint(prompt_path)
    model_path = context.path("models", "qwen")
    model_fp = fingerprint({
        "path": str(model_path),
        "files": directory_fingerprint(model_path),
        "max_new_tokens": settings["scene_max_new_tokens"],
        "do_sample": settings["do_sample"],
    })
    sources = {
        "prepare-input-data": evidence["output_fingerprint"],
        "prompt": prompt_fp,
        "model": model_fp,
    }
    visual_rows = read_jsonl(context.visual_manifest)
    names = _video_name_map(context)
    expected_outputs = [
        context.description_scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows
    ]
    if not force:
        current = _current(context, "extract-description-scenes", sources, expected_outputs)
        if current is not None:
            _completed_progress("Description scenes", len(visual_rows))
            return current
    records_by_content: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    for visual in visual_rows:
        input_fp = fingerprint({
            "evidence": visual["evidence_fingerprint"],
            "prompt": prompt_fp,
            "model": model_fp,
        })
        path = context.description_scene_dir / f"{visual['content_id']}.jsonl"
        if path.is_file() and not force:
            existing = read_jsonl(path)
            if existing and all(row.get("input_fingerprint") == input_fp for row in existing):
                records_by_content[str(visual["content_id"])] = existing
                continue
        scene_rows = _scene_generation_rows(
            visual,
            prompt=prompt,
            max_new_tokens=int(settings["scene_max_new_tokens"]),
        )
        pending.append((visual, input_fp, scene_rows))

    with tqdm(
        total=len(visual_rows),
        initial=len(records_by_content),
        desc="Description scenes",
        unit="content",
    ) as progress:
        if pending:
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                for visual, input_fp, scene_rows in pending:
                    generated = generate([row["task"] for row in scene_rows], None)
                    records = []
                    for row in scene_rows:
                        description = generated[row["task"].task_id].strip()
                        if not description:
                            raise ExtractionStepError(
                                f"{visual['content_id']} scene {row['scene_idx']} "
                                "produced an empty description"
                            )
                        records.append({
                            "schema_version": SCENE_SCHEMA_VERSION,
                            "content_id": visual["content_id"],
                            "scene_idx": row["scene_idx"],
                            "keyframes": row["keyframes"],
                            "image_paths": row["image_paths"],
                            "description": description,
                        })
                    records = [{
                        **row,
                        "prompt_fingerprint": prompt_fp,
                        "model_fingerprint": model_fp,
                        "evidence_fingerprint": visual["evidence_fingerprint"],
                        "input_fingerprint": input_fp,
                    } for row in records]
                    path = context.description_scene_dir / f"{visual['content_id']}.jsonl"
                    write_jsonl(path, records)
                    records_by_content[str(visual["content_id"])] = records
                    video_name = names.get(
                        str(visual["content_id"]),
                        f"{visual['content_id']}.mp4",
                    )
                    for message in scene_messages(
                        video_name, records, arm="description"
                    ):
                        _write_progress(progress, message)
                    progress.update(1)
    outputs = [
        record
        for visual in visual_rows
        for record in records_by_content[str(visual["content_id"])]
    ]
    output_fp = fingerprint(outputs)
    return _write_stage(
        context,
        "extract-description-scenes",
        source_fingerprints=sources,
        output_fingerprint=output_fp,
        content_count=len({row["content_id"] for row in outputs}),
    )


def summarize_description(
    context: RunContext,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    scenes_manifest = _require_stage(context, "extract-description-scenes")
    settings = context.config["extraction"]["description"]
    prompt_path = context.config_path("extraction", "description", "summary_prompt")
    template = prompt_path.read_text(encoding="utf-8")
    prompt_fp = file_fingerprint(prompt_path)
    model_path = context.path("models", "qwen")
    model_fp = fingerprint({
        "path": str(model_path),
        "files": directory_fingerprint(model_path),
        "max_new_tokens": settings["summary_max_new_tokens"],
        "do_sample": settings["do_sample"],
    })
    sources = {
        "extract-description-scenes": scenes_manifest["output_fingerprint"],
        "prompt": prompt_fp,
        "model": model_fp,
    }
    visual_rows = read_jsonl(context.visual_manifest)
    names = _video_name_map(context)
    scene_paths = [
        context.description_scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows
    ]
    if not all(path.is_file() for path in scene_paths):
        raise ExtractionStepError("description scene outputs are incomplete")
    expected_outputs = [
        context.description_summary_dir / f"{row['content_id']}.json" for row in visual_rows
    ]
    if not force:
        current = _current(context, "summarize-description", sources, expected_outputs)
        if current is not None:
            _completed_progress("Description summaries", len(visual_rows))
            return current
    documents_by_content: dict[str, dict[str, Any]] = {}
    pending: list[tuple[list[dict[str, Any]], str, Path, str]] = []
    tasks: list[QwenGenerationTask] = []
    for scene_path in scene_paths:
        records = read_jsonl(scene_path)
        if not records or any(row.get("schema_version") != SCENE_SCHEMA_VERSION for row in records):
            raise ExtractionStepError(f"invalid description scene file: {scene_path}")
        input_fp = fingerprint({"scenes": file_fingerprint(scene_path), "prompt": prompt_fp, "model": model_fp})
        output_path = context.description_summary_dir / f"{records[0]['content_id']}.json"
        if output_path.is_file() and not force:
            existing = read_json(output_path)
            if existing.get("input_fingerprint") == input_fp:
                documents_by_content[str(records[0]["content_id"])] = existing
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
        pending.append((records, input_fp, output_path, task_id))

    pending_by_task = {
        task_id: (records, input_fp, output_path)
        for records, input_fp, output_path, task_id in pending
    }
    with tqdm(
        total=len(visual_rows),
        initial=len(documents_by_content),
        desc="Description summaries",
        unit="content",
    ) as progress:
        def complete_description_summary(task_id: str, text: str) -> None:
            records, input_fp, output_path = pending_by_task[task_id]
            summary = validate_description_summary(text)
            content_id = str(records[0]["content_id"])
            document = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "content_id": content_id,
                "arm": "description",
                "status": "complete",
                "text": summary,
                "scene_count": len(records),
                "evidence_fingerprint": records[0]["evidence_fingerprint"],
                "scene_prompt_fingerprint": records[0]["prompt_fingerprint"],
                "summary_prompt_fingerprint": prompt_fp,
                "model_fingerprint": model_fp,
                "input_fingerprint": input_fp,
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
            progress.update(1)

        if tasks:
            with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
                generate(tasks, complete_description_summary)
    documents = [
        documents_by_content[str(row["content_id"])] for row in visual_rows
    ]
    output_fp = fingerprint(documents)
    return _write_stage(
        context,
        "summarize-description",
        source_fingerprints=sources,
        output_fingerprint=output_fp,
        content_count=len(documents),
    )


STEP_HANDLERS: dict[str, Callable[[RunContext], dict[str, Any]]] = {
    "prepare-input-data": prepare_input_data,
    "extract-graph-scenes": extract_graph_scenes,
    "summarize-graph": summarize_graph,
    "extract-description-scenes": extract_description_scenes,
    "summarize-description": summarize_description,
}
