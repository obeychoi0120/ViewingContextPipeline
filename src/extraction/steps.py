from __future__ import annotations
from typing import Any
from functools import partial
from tqdm import tqdm
from arm_registry import generation_registry, resolve_generation_arm
from model_provenance import local_model_identity
from extraction.backends import GeminiWorkerPool
from extraction.errors import ExtractionStepError
from extraction.qwen_runtime import QwenRuntime
from artifact_io import file_fingerprint
from extraction.qwen_config import penalty_schedule
from extraction.failures import FailureLog
from extraction.progress import InferenceProgress
from extraction.semantic_graph.parser import GRAPH_PARSER_VERSION
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
from extraction.scene_storage import read_scene_records
from extraction.raw_output import is_raw_graph
from pipeline_logging import log_step_start
from pipeline_runtime import RunContext

GRAPH_SOURCES = ("qwen", "gemini")


def prompt_provenance(context, schema, arm, *, summary=False, model=None):
    path = context.prompt_path(schema)
    source = model if summary else arm.model
    model = (
        local_model_identity(context.path("models", "qwen"))
        if source == "qwen"
        else context.config["models"][source]
    )
    extraction = context.config["extraction"]
    kind = "graph" if arm.representation == "graph" else "description"
    settings = {
        "max_new_tokens": extraction[kind][source][
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
    if summary and source == "qwen":
        settings["greedy_decoding"] = extraction["greedy_decoding"]
        if not extraction["greedy_decoding"]:
            settings["sampling"] = dict(extraction["summary_sampling"])
    elif not summary:
        settings["visual_evidence"] = dict(extraction["visual_evidence"])
        if arm.representation == "graph":
            settings["response_parser"] = GRAPH_PARSER_VERSION
    return {
        **(
            {"uses_title": arm.uses_title, "scene_arm": arm.scene_arm}
            if summary
            else {"scene_arm": arm.name}
        ),
        "representation": arm.representation,
        "prompt_path": str(path),
        "prompt_hash": file_fingerprint(path),
        "model": model,
        "settings": settings,
        **({"summary_model": source} if summary else {}),
        "schema_contract": "summary/v5"
        if summary
        else ("graph/v5" if "[Actions]" in path.read_text(encoding="utf-8") else "graph/v3")
        if arm.representation == "graph"
        else "description/v2",
    }


def _summary_generation_settings(context: RunContext) -> dict[str, Any]:
    settings = context.config["extraction"]
    generation = {"repetition_penalty": penalty_schedule(settings["summary_repetition_penalty"])[0]}
    if not settings["greedy_decoding"]:
        generation.update(do_sample=True, **settings["summary_sampling"])
    return generation


def prepare_scene_batch(visuals, scene_dir, failures, prepare_rows, normalize, *, force):
    existing, planned = {}, []
    for visual in visuals:
        cid = visual["content_id"]
        output = scene_dir / f"{cid}.jsonl"
        rows = prepare_rows(visual)
        try:
            cached = (
                normalize(read_scene_records(output), output)
                if output.is_file() and (not force)
                else []
            )
        except (ValueError, ExtractionStepError) as exc:
            raise ExtractionStepError(
                f"cannot safely identify existing scenes: {output}: {exc}"
            ) from exc
        by_index = {r["scene_idx"]: r for r in cached}
        if len(by_index) != len(cached):
            raise ExtractionStepError(f"duplicate cached scenes: {output}")
        retained, missing = ([], [])
        for row in rows:
            saved = by_index.get(row["scene_idx"])
            reusable = saved is not None
            if reusable and is_raw_graph(saved) and (not failures.contains(cid, row["scene_idx"])):
                failures.record(
                    cid,
                    row["scene_idx"],
                    "graph validation failed; raw response retained",
                    saved["raw_response"],
                )
            if failures.contains(cid, row["scene_idx"]):
                missing.append(row)
            elif reusable:
                retained.append(saved)
            elif not failures.contains(cid, row["scene_idx"]):
                missing.append(row)
        existing[cid] = retained
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


def _extract(context, *, representation, model, schema, force=False, arm=None):
    if arm is None:
        raise ValueError("extraction requires --arm")
    arm = resolve_generation_arm(context.config, arm, representation, model)
    stage = f"extract-{representation}-scenes"
    path = context.prompt_path(schema)
    log_step_start(context, stage, model=model, schema=path, force=force, arm=arm.name)
    context.initialize()
    prompt = path.read_text(encoding="utf-8")
    print("[PREPARE] Reading prompt/model settings and cohort...", flush=True)
    settings = context.config["extraction"]
    penalties = settings[f"{representation}_repetition_penalty"] if model == "qwen" else [1.0]
    scene_dir = context.extraction_dir(arm.representation, model, "scenes")
    failures = FailureLog(scene_dir, scenes=True)
    normalize = minimal_graph_records if representation == "graph" else minimal_description_records
    visuals = visual_rows(context)
    content_ids = [visual["content_id"] for visual in visuals]
    if force:
        failures.clear_contents(content_ids)
    prepare_rows = partial(
        scene_generation_rows,
        prompt=prompt,
        max_new_tokens=settings[representation][model]["scene_max_new_tokens"],
        repetition_penalty=penalty_schedule(penalties)[0],
    )
    print("[PREPARE] Counting pending scenes and checking cached outputs...", flush=True)
    existing, planned = prepare_scene_batch(
        visuals,
        scene_dir,
        failures,
        prepare_rows,
        normalize,
        force=force,
    )
    total = sum((len(scenes) for _, scenes in planned))
    reused = sum((len(records) for records in existing.values()))

    def pending_contents():
        for visual, scenes in planned:
            missing = prepare_rows(visual, scenes=scenes)
            yield (visual, missing)

    with InferenceProgress(
        total=total,
        reused=reused,
        desc=arm.name,
        unit="scene",
        progress_factory=tqdm,
        track_structured=representation == "graph",
    ) as progress:
        if model == "qwen":
            run_qwen_scenes(
                pending_contents(),
                provenance=prompt_provenance(context, path, arm),
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
                penalties=penalty_schedule(penalties),
            )
        else:
            pool = GeminiWorkerPool(
                settings["gemini"]["threads"], **context.config["models"]["gemini"]
            )
            run_gemini_scenes(
                pending_contents(),
                provenance=prompt_provenance(context, path, arm),
                pool=pool,
                records_by_content=existing,
                scene_dir=scene_dir,
                failures=failures,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
            )
    return result(
        f"{stage}-{model}", content_count=len(visuals), failure_count=failures.count(content_ids)
    )


def _summarize(context, *, representation, model, schema, force=False, arm=None):
    if model not in GRAPH_SOURCES:
        raise ValueError("summary model must be qwen or gemini")
    if arm is None:
        raise ValueError("summarization requires --arm")
    arm = resolve_generation_arm(context.config, arm, representation, model, summary=True)
    path = context.prompt_path(schema)
    stage = f"summarize-{representation}"
    log_step_start(
        context, stage, source=arm.model, model=model, schema=path, force=force, arm=arm.name
    )
    from extraction.summary_storage import validate_summary_template

    validate_summary_template(path.read_text(), arm.uses_title)
    context.initialize()
    cohort = context.require_ready_cohort()
    return run_summary_stage(
        context,
        arm=arm,
        schema=path,
        catalog=cohort["catalog"],
        provenance=prompt_provenance(context, path, arm, summary=True, model=model),
        generation=_summary_generation_settings(context) if model == "qwen" else {},
        model=model,
        gemini_pool_factory=GeminiWorkerPool,
        force=force,
        generator_factory=qwen_generator,
    )


def extract_graph_scenes(context, *, model, schema, force=False, arm=None):
    return _extract(
        context, representation="graph", model=model, schema=schema, force=force, arm=arm
    )


def extract_description_scenes(context, *, model, schema, force=False, arm=None):
    return _extract(
        context, representation="description", model=model, schema=schema, force=force, arm=arm
    )


def summarize(context, *, model, schema, arm=None, force=False):
    if arm is None:
        raise ValueError("summarize requires --arm")
    selected = generation_registry(context.config).get(arm)
    representation = selected.representation if selected else None
    return _summarize(
        context, representation=representation, model=model, schema=schema, force=force, arm=arm
    )


STEP_HANDLERS = {
    "extract-graph-scenes": extract_graph_scenes,
    "extract-description-scenes": extract_description_scenes,
    "summarize": summarize,
}
