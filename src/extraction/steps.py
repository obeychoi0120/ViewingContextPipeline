from __future__ import annotations

from dataclasses import replace
from typing import Any

from tqdm import tqdm

from arm_registry import generated_arm
from model_provenance import local_model_identity
from extraction.backends import GeminiWorkerPool
from extraction.errors import ExtractionStepError
from extraction.qwen_runtime import QwenRuntime
from extraction.recovery import file_fingerprint, generation_key, penalty_schedule
from extraction.failures import FailureLog
from extraction.progress import InferenceProgress
from extraction.structured_output import GRAPH_JSON_SCHEMA
from extraction.scene_executor import run_gemini_scenes, run_qwen_scenes
from extraction.step_support import (
    minimal_description_records,
    minimal_graph_records,
    result,
    scene_generation_rows,
    video_name_map,
    visual_rows,
)
from extraction.summary_executor import run_summary_stage, qwen_generator
from extraction.scene_storage import read_scene_records, migrate_scene_schema
from extraction.raw_output import is_raw_graph
from pipeline_logging import log_step_start
from pipeline_runtime import RunContext

GRAPH_SOURCES = ("qwen", "gemini")


def prompt_provenance(context, schema, arm, *, summary=False):
    path = context.prompt_path(schema)
    source = "qwen" if summary else arm.model
    model = (
        local_model_identity(context.path("models", "qwen"))
        if source == "qwen"
        else context.config["models"][source]
    )
    extraction = context.config["extraction"]
    kind = "graph" if arm.representation == "graph" else "description"
    settings = {
        "max_new_tokens": extraction[kind][
            "summary_max_new_tokens" if summary else "scene_max_new_tokens"
        ]
    }
    if source == "qwen":
        from extraction.qwen_config import qwen_settings

        settings.update(
            backend="vllm-0.28.0",
            qwen=qwen_settings(extraction.get("qwen")),
            repetition_penalty=penalty_schedule(
                extraction[
                    "summary_repetition_penalty" if summary else f"{kind}_repetition_penalty"
                ]
            ),
        )
    else:
        settings["backend"] = "gemini"
    if summary:
        settings["greedy_decoding"] = extraction["greedy_decoding"]
        if not extraction["greedy_decoding"]:
            settings["sampling"] = dict(extraction["summary_sampling"])
    else:
        settings["visual_evidence"] = dict(extraction["visual_evidence"])
    return {
        "arm": arm.name,
        "representation": arm.representation,
        "prompt_path": str(path),
        "prompt_hash": file_fingerprint(path),
        "model": model,
        "settings": settings,
        "schema_contract": "summary/v4"
        if summary
        else ("graph/v3" if arm.representation == "graph" else "description/v2"),
    }


def _summary_generation_settings(context: RunContext) -> dict[str, Any]:
    settings = context.config["extraction"]
    generation = {"repetition_penalty": penalty_schedule(settings["summary_repetition_penalty"])[0]}
    if not settings["greedy_decoding"]:
        generation.update(do_sample=True, **settings["summary_sampling"])
    return generation


def _extract(context, *, representation, model, schema, force=False):
    arm = generated_arm(context.config, representation, model)
    stage = f"extract-{representation}-scenes"
    path = context.prompt_path(schema)
    log_step_start(context, stage, model=model, schema=path, force=force)
    context.initialize()
    prompt = path.read_text(encoding="utf-8")
    print("[PREPARE] Reading prompt/model settings and cohort...", flush=True)
    provenance = prompt_provenance(context, path, arm)
    settings = context.config["extraction"]
    penalties = settings[f"{representation}_repetition_penalty"] if model == "qwen" else [1.0]
    scene_dir = context.extraction_dir(arm.representation, model, "scenes")
    failures = FailureLog(scene_dir)
    normalize = minimal_graph_records if representation == "graph" else minimal_description_records
    visuals = visual_rows(context)
    existing = {}
    content_ids = [visual["content_id"] for visual in visuals]
    if force:
        failures.clear_contents(content_ids)
    initial_penalty = penalty_schedule(penalties)[0]

    def prepare_rows(visual, *, scenes=None):
        rows = scene_generation_rows(
            visual,
            prompt=prompt,
            max_new_tokens=settings[representation]["scene_max_new_tokens"],
            repetition_penalty=initial_penalty,
            scenes=scenes,
        )
        for row in rows:
            if representation == "graph" and model == "qwen":
                row["task"] = replace(row["task"], structured_output={"json": GRAPH_JSON_SCHEMA})
            row["provenance"] = provenance
            row["input_key"] = generation_key(row["task"], provenance, penalties)
        return rows

    def plan_contents():
        for visual in visuals:
            cid = visual["content_id"]
            output = scene_dir / f"{cid}.jsonl"
            rows = prepare_rows(visual)
            try:
                cached = normalize(read_scene_records(output), output) if output.is_file() and not force else []
            except (ValueError, ExtractionStepError):
                # No journal is kept: incomplete metadata/payload pairs are regenerated.
                cached = []
            by_index = {r["scene_idx"]: r for r in cached}
            if len(by_index) != len(cached):
                raise ExtractionStepError(f"duplicate cached scenes: {output}")
            retained, missing = [], []
            for row in rows:
                saved = by_index.get(row["scene_idx"])
                generation = saved.get("generation", {}) if saved else {}
                reusable = (
                    saved is not None
                    and (
                        (
                            bool(generation.get("input_key"))
                            and saved.get("provenance") == row["provenance"]
                            and saved.get("keyframes") == row["keyframes"]
                        )
                        or generation.get("input_key") == generation_key(
                            row["task"], provenance, penalties
                        )
                    )
                )
                if reusable:
                    retained.append(saved)
                    if is_raw_graph(saved) and not failures.contains(cid, row["scene_idx"]):
                        failures.record(cid, row["scene_idx"], "graph validation failed; raw response retained")
                elif not failures.contains(cid, row["scene_idx"]):
                    missing.append(row)
            existing[cid] = retained
            if missing:
                # Retain compact scene metadata, not tasks or image paths.
                # Inference can then build tasks lazily without rereading assets.
                yield visual, [
                    {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"],
                     "scene_start": row.get("scene_start_seconds"),
                     "scene_end": row.get("scene_end_seconds")}
                    for row in missing
                ]

    print("[PREPARE] Counting pending scenes and checking cached outputs...", flush=True)
    planned = list(plan_contents())
    total = sum(len(scenes) for _, scenes in planned)
    reused = sum(len(records) for records in existing.values())

    def pending_contents():
        for visual, scenes in planned:
            missing = prepare_rows(visual, scenes=scenes)
            yield visual, missing

    with InferenceProgress(
        total=total,
        reused=reused,
        desc=arm.name,
        unit="scene",
        progress_factory=tqdm,
    ) as progress:
        if model == "qwen":
            run_qwen_scenes(
                pending_contents(),
                scene_dir=scene_dir,
                failures=failures,
                model_path=context.path("models", "qwen"),
                generator_factory=qwen_generator,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
                source=model,
                existing_records=existing,
                qwen_options=settings.get("qwen"),
                image_limit=settings["visual_evidence"]["num_keyframes"],
                runtime=QwenRuntime(),
            )
        else:
            pool = GeminiWorkerPool(
                settings["gemini"]["threads"], **context.config["models"]["gemini"]
            )
            run_gemini_scenes(
                pending_contents(),
                pool=pool,
                records_by_content=existing,
                scene_dir=scene_dir,
                failures=failures,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
            )
    return result(f"{stage}-{model}", content_count=len(visuals),
                  failure_count=failures.count(content_ids))


def _summarize(context, *, representation, source, schema, force=False):
    arm = generated_arm(context.config, representation, source)
    path = context.prompt_path(schema)
    stage = f"summarize-{representation}"
    log_step_start(context, stage, source=source, schema=path, force=force)
    context.initialize()
    cohort = context.require_ready_cohort()
    return run_summary_stage(
        context,
        arm=arm,
        schema=path,
        catalog=cohort["catalog"],
        provenance=prompt_provenance(context, path, arm, summary=True),
        generation=_summary_generation_settings(context),
        force=force,
        generator_factory=qwen_generator,
    )


def extract_graph_scenes(context, *, model, schema, force=False):
    return _extract(
        context, representation="graph", model=model, schema=schema, force=force
    )


def extract_description_scenes(context, *, model, schema, force=False):
    return _extract(
        context, representation="description", model=model, schema=schema, force=force
    )


def summarize_graph(context, *, source, schema, force=False):
    return _summarize(
        context, representation="graph", source=source, schema=schema, force=force
    )


def summarize_description(context, *, source, schema, force=False):
    return _summarize(
        context, representation="description", source=source, schema=schema, force=force
    )


STEP_HANDLERS = {
    "migrate-scene-schema": migrate_scene_schema,
    "extract-graph-scenes": extract_graph_scenes,
    "extract-description-scenes": extract_description_scenes,
    "summarize-graph": summarize_graph,
    "summarize-description": summarize_description,
}
