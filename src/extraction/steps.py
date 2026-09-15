from __future__ import annotations

from dataclasses import replace
from typing import Any

from tqdm import tqdm

from arm_registry import generated_arm
from model_provenance import local_model_identity
from extraction.backends import GeminiWorkerPool
from extraction.errors import ExtractionStepError
from extraction.preparation import prepare_input_data
from extraction.qwen_runtime import QwenRuntime
from extraction.recovery import (
    active_force_run,
    clear_recovery,
    generation_key,
    has_pending_recovery,
)
from extraction.recovery import file_fingerprint, penalty_schedule
from extraction.progress import InferenceProgress
from extraction.structured_output import GRAPH_JSON_SCHEMA
from extraction.scene_executor import run_gemini_scenes, run_qwen_scenes
from extraction.step_support import (
    minimal_description_records,
    minimal_graph_records,
    restore_scene_checkpoint,
    result,
    scene_generation_rows,
    video_name_map,
    visual_rows,
)
from extraction.summary_executor import run_summary_stage, qwen_generator
from pipeline_logging import log_step_start
from pipeline_runtime import RunContext, read_json, read_jsonl

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


def _extract(context, *, representation, model, schema, force=False, gpus=None):
    arm = generated_arm(context.config, representation, model)
    if model == "gemini" and gpus is not None:
        raise ValueError("--gpus cannot be used with Gemini extraction")
    stage = f"extract-{representation}-scenes"
    path = context.prompt_path(schema)
    log_step_start(context, stage, model=model, schema=path, force=force, gpus=gpus)
    context.initialize()
    prompt = path.read_text(encoding="utf-8")
    provenance = prompt_provenance(context, path, arm)
    settings = context.config["extraction"]
    penalties = settings[f"{representation}_repetition_penalty"] if model == "qwen" else [1.0]
    scene_dir = context.extraction_dir(arm.representation, model, "scenes")
    failure_dir = scene_dir / "failures"
    normalize = minimal_graph_records if representation == "graph" else minimal_description_records
    visuals = visual_rows(context)
    existing, failures, pending = {}, {}, []
    force_run = active_force_run(scene_dir / ".recovery")
    interrupted = set()
    if model == "gemini" and (scene_dir / ".pending-contents.json").is_file():
        cursor = scene_dir / ".completed-contents.json"
        count = read_json(cursor)["count"] if cursor.is_file() else 0
        interrupted = set(read_json(scene_dir / ".pending-contents.json")["content_ids"][count:])
    for visual in visuals:
        cid = visual["content_id"]
        output = scene_dir / f"{cid}.jsonl"
        failure_path = failure_dir / f"{cid}.jsonl"
        restore_scene_checkpoint(output, failure_path)
        rows = scene_generation_rows(
            visual,
            prompt=prompt,
            max_new_tokens=settings[representation]["scene_max_new_tokens"],
            repetition_penalty=penalty_schedule(penalties)[0],
        )
        for row in rows:
            if representation == "graph" and model == "qwen":
                row["task"] = replace(row["task"], structured_output={"json": GRAPH_JSON_SCHEMA})
            row["provenance"] = provenance
            row["input_key"] = generation_key(row["task"], provenance, penalties)
        cached = normalize(read_jsonl(output), output) if output.is_file() and not force else []
        by_index = {r["scene_idx"]: r for r in cached}
        if len(by_index) != len(cached):
            raise ExtractionStepError(f"duplicate cached scenes: {output}")
        retained, missing = [], []
        for row in rows:
            saved = by_index.get(row["scene_idx"])
            generation = saved.get("generation", {}) if saved else {}
            reusable = (
                saved is not None
                and generation.get("input_key") == row["input_key"]
                and cid not in interrupted
                and not has_pending_recovery(
                    scene_dir / ".recovery", row["task"].task_id, force_run, generation
                )
            )
            (retained if reusable else missing).append(saved if reusable else row)
        existing[cid] = retained
        failures[cid] = []
        if missing:
            pending.append((visual, missing))
        else:
            failure_path.unlink(missing_ok=True)
    with InferenceProgress(
        total=sum(len(r) for _, r in pending),
        reused=sum(len(r) for r in existing.values()),
        desc=arm.name,
        unit="scene",
        progress_factory=tqdm,
    ) as progress:
        if model == "qwen":
            done, failed = run_qwen_scenes(
                pending,
                scene_dir=scene_dir,
                failure_dir=failure_dir,
                model_path=context.path("models", "qwen"),
                gpus=gpus,
                generator_factory=qwen_generator,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
                source=model,
                existing_records=existing,
                existing_failures=failures,
                qwen_options=settings.get("qwen"),
                image_limit=settings["visual_evidence"]["num_keyframes"],
                runtime=QwenRuntime(),
                penalties=penalties,
                force=force,
                identity=provenance,
            )
            existing.update(done)
            failures.update(failed)
        elif pending:
            pool = GeminiWorkerPool(
                settings["gemini"]["threads"], **context.config["models"]["gemini"]
            )
            run_gemini_scenes(
                pending,
                pool=pool,
                records_by_content=existing,
                failures_by_content=failures,
                scene_dir=scene_dir,
                failure_dir=failure_dir,
                force=force,
                names=video_name_map(context),
                progress=progress,
                arm=representation,
            )
    failure_count = sum(len(rows) for rows in failures.values())
    if not failure_count:
        clear_recovery(scene_dir)
    return result(f"{stage}-{model}", content_count=len(visuals), failure_count=failure_count)


def _summarize(context, *, representation, source, schema, force=False, gpus=None):
    arm = generated_arm(context.config, representation, source)
    path = context.prompt_path(schema)
    stage = f"summarize-{representation}"
    log_step_start(context, stage, source=source, schema=path, force=force, gpus=gpus)
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
        gpus=gpus,
        generator_factory=qwen_generator,
    )


def extract_graph_scenes(context, *, model, schema, force=False, gpus=None):
    return _extract(
        context, representation="graph", model=model, schema=schema, force=force, gpus=gpus
    )


def extract_description_scenes(context, *, model, schema, force=False, gpus=None):
    return _extract(
        context, representation="description", model=model, schema=schema, force=force, gpus=gpus
    )


def summarize_graph(context, *, source, schema, force=False, gpus=None):
    return _summarize(
        context, representation="graph", source=source, schema=schema, force=force, gpus=gpus
    )


def summarize_description(context, *, source, schema, force=False, gpus=None):
    return _summarize(
        context, representation="description", source=source, schema=schema, force=force, gpus=gpus
    )


STEP_HANDLERS = {
    "prepare-input-data": prepare_input_data,
    "extract-graph-scenes": extract_graph_scenes,
    "extract-description-scenes": extract_description_scenes,
    "summarize-graph": summarize_graph,
    "summarize-description": summarize_description,
}
