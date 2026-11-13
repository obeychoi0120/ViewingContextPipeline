from __future__ import annotations
from functools import partial
from tqdm import tqdm
from arm_registry import generated_arm, resolve_generation_arm
from model_provenance import local_model_identity
from extraction.backends import GeminiWorkerPool
from extraction.errors import ExtractionStepError
from extraction.qwen_runtime import QwenRuntime
from artifact_io import atomic_write_json, file_fingerprint
from extraction.qwen_config import QWEN_MODELS, generation_settings, qwen_settings, validate_checkpoint
from extraction.graph_warnings import DESCRIPTION_OUTCOMES, OUTCOMES, needs_generation, outcome
from extraction.progress import InferenceProgress
from extraction.semantic_graph.parser import GRAPH_PARSER_VERSION
from extraction.semantic_graph import GRAPH_SCHEMA_VERSION
from extraction.scene_executor import run_gemini_scenes, run_qwen_scenes
from extraction.step_support import (
    minimal_description_records,
    minimal_graph_records,
    result,
    scene_generation_rows,
    video_name_map,
    visual_rows,
)
from extraction.generation import qwen_generator
from extraction.scene_storage import read_scene_records, migrate_scene_failures, reclassify_scene_file
from pipeline_logging import log_step_start

LOCAL_MODELS = QWEN_MODELS
GRAPH_SOURCES = (*QWEN_MODELS, "gemini")


def prompt_provenance(
    context, schema, arm, *, repetition_penalty=None, presence_penalty=None, stop_repeated_lines=False,
):
    path = context.prompt_path(schema)
    source = arm.model
    if repetition_penalty is not None and source not in LOCAL_MODELS:
        raise ValueError("repetition penalty is supported only for local vLLM models")
    if presence_penalty is not None and source not in LOCAL_MODELS:
        raise ValueError("presence penalty is supported only for local vLLM models")
    if stop_repeated_lines and source not in LOCAL_MODELS:
        raise ValueError("stopping before a repeated line is supported only for local vLLM models")
    if source in LOCAL_MODELS:
        import json
        checkpoint = context.path("models", source)
        validate_checkpoint(source, json.loads((checkpoint / "config.json").read_text()), checkpoint)
    model = (
        local_model_identity(context.path("models", source))
        if source in LOCAL_MODELS
        else context.config["models"][source]
    )
    extraction = context.config["extraction"]
    kind = "graph" if arm.representation == "graph" else "description"
    settings = {
        "max_new_tokens": extraction[kind][source]["scene_max_new_tokens"]
    }
    if source in LOCAL_MODELS:
        generation = generation_settings(
            source, representation=kind, repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, stop_repeated_lines=stop_repeated_lines,
        )
        settings.update(
            backend="vllm",
            qwen=qwen_settings(extraction.get("qwen")),
            generation=generation,
            repetition_penalty=[generation["repetition_penalty"]],
        )
    else:
        settings["backend"] = "gemini"
    settings["visual_evidence"] = dict(extraction["visual_evidence"])
    if arm.representation == "graph":
        settings["response_parser"] = GRAPH_PARSER_VERSION
    return {
        "scene_arm": arm.name,
        "representation": arm.representation,
        "prompt_path": str(path),
        "prompt_hash": file_fingerprint(path),
        "model": model,
        "settings": settings,
        "schema_contract": GRAPH_SCHEMA_VERSION
        if arm.representation == "graph"
        else "description/v2",
    }


def prepare_scene_batch(visuals, scene_dir, prepare_rows, normalize, *, force):
    existing, planned = {}, []
    for visual in visuals:
        cid = visual["content_id"]
        output = scene_dir / f"{cid}.jsonl"
        rows = prepare_rows(visual)
        try:
            if output.is_file() and not force:
                # Stored rows follow the current rules before deciding what to generate.
                reclassify_scene_file(output)
            cached = (
                read_scene_records(output, include_failures=True)
                if output.is_file() and (not force)
                else []
            )
        except (ValueError, ExtractionStepError) as exc:
            raise ExtractionStepError(
                f"cannot safely identify existing scenes: {output}: {exc}"
            ) from exc
        normalize([
            row for row in cached if not needs_generation(row)
        ], output)
        by_index = {r["scene_idx"]: r for r in cached}
        if len(by_index) != len(cached):
            raise ExtractionStepError(f"duplicate cached scenes: {output}")
        missing = []
        for row in rows:
            saved = by_index.get(row["scene_idx"])
            if needs_generation(saved):
                missing.append(row)
        expected_indices = {row["scene_idx"] for row in rows}
        existing[cid] = [row for row in cached if row["scene_idx"] in expected_indices]
        if missing:
            planned.append(
                (
                    visual,
                    [
                        {
                            "scene_idx": row["scene_idx"],
                            "keyframes": row["keyframes"],
                            "scene_start": row.get("scene_start_seconds"),
                            "scene_end": row.get("scene_end_seconds"),
                        }
                        for row in missing
                    ],
                )
            )
    return existing, planned


def _extract(
    context, *, representation, model, schema, force=False, arm=None,
    repetition_penalty=None, presence_penalty=None, stop_repeated_lines=False,
):
    from pipeline_runtime import require_generation_config

    require_generation_config(context.config, representation, model)
    # --arm is optional: the step and model already determine the source.
    arm = (generated_arm(context.config, representation, model) if arm is None
           else resolve_generation_arm(context.config, arm, representation, model))
    stage = f"extract-{representation}-scenes"
    path = context.prompt_path(schema)
    log_step_start(
        context, stage, model=model, schema=path, force=force, arm=arm.name,
        **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
        **({"presence_penalty": presence_penalty} if presence_penalty is not None else {}),
        **({"stop_repeated_lines": True} if stop_repeated_lines else {}),
    )
    context.initialize()
    prompt = path.read_text(encoding="utf-8")
    print("[PREPARE] Reading prompt/model settings and cohort...", flush=True)
    settings = context.config["extraction"]
    provenance = prompt_provenance(
        context, path, arm, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
        stop_repeated_lines=stop_repeated_lines,
    )
    penalties = provenance["settings"].get("repetition_penalty", [1.0])
    scene_dir = context.extraction_dir(arm.representation, model, "scenes")
    from extraction.scene_storage import ensure_generation_identity

    ensure_generation_identity(scene_dir, provenance, force=force)
    (scene_dir / "generation_stats.json").unlink(missing_ok=True)
    migrate_scene_failures(scene_dir)
    normalize = minimal_graph_records if representation == "graph" else minimal_description_records
    visuals = visual_rows(context)
    prepare_rows = partial(
        scene_generation_rows,
        prompt=prompt,
        max_new_tokens=settings[representation][model]["scene_max_new_tokens"],
        generation=provenance["settings"].get("generation", {}),
    )
    print("[PREPARE] Counting pending scenes and checking cached outputs...", flush=True)
    existing, planned = prepare_scene_batch(
        visuals,
        scene_dir,
        prepare_rows,
        normalize,
        force=force,
    )
    total = sum((len(scenes) for _, scenes in planned))
    reused = sum(
        not needs_generation(row)
        for records in existing.values() for row in records
    )

    def pending_contents():
        for visual, scenes in planned:
            missing = prepare_rows(visual, scenes=scenes)
            yield (visual, missing)

    outcomes = OUTCOMES if representation == "graph" else DESCRIPTION_OUTCOMES
    with InferenceProgress(
        total=total,
        reused=reused,
        desc=arm.name,
        unit="scene",
        outcomes=outcomes,
        progress_factory=tqdm,
    ) as progress:
        if model in LOCAL_MODELS:
            run_qwen_scenes(
                pending_contents(),
                provenance=provenance,
                scene_dir=scene_dir,
                model_path=context.path("models", model),
                generator_factory=qwen_generator,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
                existing_records=existing,
                qwen_options=settings.get("qwen"),
                image_limit=settings["visual_evidence"]["num_keyframes"],
                runtime=QwenRuntime(),
                penalties=penalties,
            )
        else:
            pool = GeminiWorkerPool(
                settings["gemini"]["threads"], **context.config["models"]["gemini"]
            )
            run_gemini_scenes(
                pending_contents(),
                provenance=provenance,
                pool=pool,
                records_by_content=existing,
                scene_dir=scene_dir,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
            )
    failure_count = sum("error" in row for records in existing.values() for row in records)
    # Each scene is counted once; the outcomes sum to the total.
    stats = {"total": total + reused, **dict.fromkeys(outcomes, 0)}
    for records in existing.values():
        for record in records:
            stats[outcome(record)] += 1
    atomic_write_json(scene_dir / "generation_stats.json", stats, durable=True)
    return result(
        f"{stage}-{model}", content_count=len(visuals), failure_count=failure_count
    )


def extract_graph_scenes(
    context, *, model, schema, force=False, arm=None, repetition_penalty=None, presence_penalty=None,
    stop_repeated_lines=False,
):
    return _extract(
        context, representation="graph", model=model, schema=schema, force=force, arm=arm,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
        stop_repeated_lines=stop_repeated_lines,
    )


def extract_description_scenes(
    context, *, model, schema, force=False, arm=None, repetition_penalty=None, presence_penalty=None,
):
    return _extract(
        context, representation="description", model=model, schema=schema, force=force, arm=arm,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
    )


STEP_HANDLERS = {
    "extract-graph-scenes": extract_graph_scenes,
    "extract-description-scenes": extract_description_scenes,
}
