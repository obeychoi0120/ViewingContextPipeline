from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from extraction.backends import QwenBackend
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
from extraction.relational_graph import (
    GRAPH_SCENE_SCHEMA,
    GRAPH_SUMMARY_SCHEMA,
    graph_summary_prompt,
    ontology_from_document,
    parse_graph_output,
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


@contextmanager
def _qwen_generator(
    *,
    model_path: Path,
    gpus: int | None,
) -> Iterator[Callable[[list[QwenGenerationTask]], dict[str, str]]]:
    if gpus is not None:
        with QwenWorkerPool(gpus, str(model_path)) as worker_pool:
            yield worker_pool.generate
        return
    backend = QwenBackend.from_pretrained(str(model_path), use_fc_patch=True)

    def generate(tasks: list[QwenGenerationTask]) -> dict[str, str]:
        return {
            task.task_id: backend.generate(
                load_images(list(task.image_paths)),
                task.prompt,
                task.max_new_tokens,
            )
            for task in tasks
        }

    yield generate


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
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    evidence = _require_stage(context, "prepare-input-data")
    settings = context.config["extraction"]["graph"]
    ontology_path = context.config_path("extraction", "graph", "ontology")
    prompt_path = context.config_path("extraction", "graph", "scene_prompt")
    ontology_document = read_json(ontology_path)
    ontology = ontology_from_document(ontology_document)
    prompt = prompt_path.read_text(encoding="utf-8")
    ontology_fp = file_fingerprint(ontology_path)
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
        "ontology": ontology_fp,
        "prompt": prompt_fp,
        "model": model_fp,
    }
    visual_rows = read_jsonl(context.visual_manifest)
    expected_outputs = [context.graph_scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows]
    if not force:
        current = _current(context, "extract-graph-scenes", sources, expected_outputs)
        if current is not None:
            return current
    records_by_content: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    for visual in visual_rows:
        input_fp = fingerprint({
            "evidence": visual["evidence_fingerprint"],
            "ontology": ontology_fp,
            "prompt": prompt_fp,
            "model": model_fp,
        })
        path = context.graph_scene_dir / f"{visual['content_id']}.jsonl"
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

    if pending:
        with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
            for visual, input_fp, scene_rows in pending:
                generated = generate([row["task"] for row in scene_rows])
                records = [{
                    "schema_version": GRAPH_SCENE_SCHEMA,
                    "content_id": visual["content_id"],
                    "scene_idx": row["scene_idx"],
                    "keyframes": row["keyframes"],
                    "image_paths": row["image_paths"],
                    "triples": parse_graph_output(
                        generated[row["task"].task_id], ontology
                    ),
                } for row in scene_rows]
                records = [{
                    **row,
                    "ontology_id": ontology.ontology_id,
                    "ontology_status": ontology.status,
                    "ontology_fingerprint": ontology_fp,
                    "prompt_fingerprint": prompt_fp,
                    "model_fingerprint": model_fp,
                    "evidence_fingerprint": visual["evidence_fingerprint"],
                    "input_fingerprint": input_fp,
                } for row in records]
                path = context.graph_scene_dir / f"{visual['content_id']}.jsonl"
                write_jsonl(path, records)
                records_by_content[str(visual["content_id"])] = records
    outputs = [
        record
        for visual in visual_rows
        for record in records_by_content[str(visual["content_id"])]
    ]
    output_fp = fingerprint(outputs)
    return _write_stage(
        context,
        "extract-graph-scenes",
        source_fingerprints=sources,
        output_fingerprint=output_fp,
        content_count=len({row["content_id"] for row in outputs}),
    )


def summarize_graph(
    context: RunContext,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    context.initialize()
    scenes_manifest = _require_stage(context, "extract-graph-scenes")
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
        "extract-graph-scenes": scenes_manifest["output_fingerprint"],
        "prompt": prompt_fp,
        "model": model_fp,
    }
    visual_rows = read_jsonl(context.visual_manifest)
    scene_paths = [context.graph_scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows]
    if not all(path.is_file() for path in scene_paths):
        raise ExtractionStepError("graph scene outputs are incomplete")
    expected_outputs = [context.graph_summary_dir / f"{row['content_id']}.json" for row in visual_rows]
    if not force:
        current = _current(context, "summarize-graph", sources, expected_outputs)
        if current is not None:
            return current
    documents_by_content: dict[str, dict[str, Any]] = {}
    pending: list[tuple[list[dict[str, Any]], str, Path, str]] = []
    tasks: list[QwenGenerationTask] = []
    for scene_path in scene_paths:
        records = read_jsonl(scene_path)
        if not records or any(row.get("schema_version") != GRAPH_SCENE_SCHEMA for row in records):
            raise ExtractionStepError(f"invalid graph scene file: {scene_path}")
        input_fp = fingerprint({"scenes": file_fingerprint(scene_path), "prompt": prompt_fp, "model": model_fp})
        output_path = context.graph_summary_dir / f"{records[0]['content_id']}.json"
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
        pending.append((records, input_fp, output_path, task_id))

    generated: dict[str, str] = {}
    if tasks:
        with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
            generated = generate(tasks)
    for records, input_fp, output_path, task_id in pending:
        summary = validate_graph_summary(generated[task_id])
        document = {
            "schema_version": GRAPH_SUMMARY_SCHEMA,
            "content_id": records[0]["content_id"],
            "arm": "graph",
            "status": "complete",
            "text": summary,
            "scene_count": len(records),
            "evidence_fingerprint": records[0]["evidence_fingerprint"],
            "ontology_fingerprint": records[0]["ontology_fingerprint"],
            "scene_prompt_fingerprint": records[0]["prompt_fingerprint"],
            "summary_prompt_fingerprint": prompt_fp,
            "model_fingerprint": model_fp,
            "input_fingerprint": input_fp,
        }
        write_json(output_path, document)
        documents_by_content[str(records[0]["content_id"])] = document
    documents = [
        documents_by_content[str(row["content_id"])] for row in visual_rows
    ]
    output_fp = fingerprint(documents)
    return _write_stage(
        context,
        "summarize-graph",
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
    expected_outputs = [
        context.description_scene_dir / f"{row['content_id']}.jsonl" for row in visual_rows
    ]
    if not force:
        current = _current(context, "extract-description-scenes", sources, expected_outputs)
        if current is not None:
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

    if pending:
        with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
            for visual, input_fp, scene_rows in pending:
                generated = generate([row["task"] for row in scene_rows])
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

    generated: dict[str, str] = {}
    if tasks:
        with _qwen_generator(model_path=model_path, gpus=gpus) as generate:
            generated = generate(tasks)
    for records, input_fp, output_path, task_id in pending:
        summary = validate_description_summary(generated[task_id])
        document = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "content_id": records[0]["content_id"],
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
        documents_by_content[str(records[0]["content_id"])] = document
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
