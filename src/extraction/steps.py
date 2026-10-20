from __future__ import annotations
from functools import partial
from tqdm import tqdm
from arm_registry import generation_registry, resolve_generation_arm
from model_provenance import local_model_identity
from extraction.backends import GeminiWorkerPool
from extraction.errors import ExtractionStepError
from extraction.qwen_runtime import QwenRuntime
from artifact_io import atomic_write_json, file_fingerprint
from extraction.qwen_config import QWEN_MODELS, generation_settings, qwen_settings
from extraction.minicpm_config import (
    NORMALIZATION_VERSION, PROCESSOR_SETTINGS, minicpm_generation_settings,
    validate_local_checkpoint,
)
from extraction.graph_warnings import WARNING_TAGS
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
from extraction.summary_executor import run_summary_stage, qwen_generator
from extraction.scene_storage import read_scene_records, migrate_scene_failures
from pipeline_logging import log_step_start

LOCAL_MODELS = (*QWEN_MODELS, "minicpm")
GRAPH_SOURCES = (*QWEN_MODELS, "gemini", "minicpm")


def prompt_provenance(
    context, schema, arm, *, summary=False, model=None, repetition_penalty=None, presence_penalty=None,
):
    path = context.prompt_path(schema)
    source = model if summary else arm.model
    if source == "minicpm" and (summary or arm.representation != "graph"):
        raise ValueError("MiniCPM supports only Graph scene extraction")
    if repetition_penalty is not None and source not in LOCAL_MODELS:
        raise ValueError("repetition penalty is supported only for local vLLM models")
    if presence_penalty is not None and source not in LOCAL_MODELS:
        raise ValueError("presence penalty is supported only for local vLLM models")
    if source in LOCAL_MODELS:
        import json
        checkpoint = context.path("models", source)
        validate_local_checkpoint(source, json.loads((checkpoint / "config.json").read_text()))
    model = (
        local_model_identity(context.path("models", source))
        if source in LOCAL_MODELS
        else context.config["models"][source]
    )
    extraction = context.config["extraction"]
    kind = "graph" if arm.representation == "graph" else "description"
    settings = {
        "max_new_tokens": extraction[kind][source][
            "summary_max_new_tokens" if summary else "scene_max_new_tokens"
        ]
    }
    if source in LOCAL_MODELS:
        generation = minicpm_generation_settings(
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
        ) if source == "minicpm" else generation_settings(
            source, representation=kind, summary=summary, repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        settings.update(
            backend="vllm",
            qwen=qwen_settings(extraction.get("qwen")),
            generation=generation,
            repetition_penalty=[generation["repetition_penalty"]],
        )
        if source == "minicpm":
            settings["processor"] = {
                **PROCESSOR_SETTINGS, "max_slice_nums": "processor_default",
                "chat_template_hash": (
                    file_fingerprint(checkpoint / "chat_template.jinja")
                    if (checkpoint / "chat_template.jinja").is_file() else None
                ),
            }
            settings["output_normalization"] = NORMALIZATION_VERSION
    else:
        settings["backend"] = "gemini"
    if not summary:
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
        else GRAPH_SCHEMA_VERSION
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
            row for row in cached if "error" not in row and not row.get("semantic_warnings")
        ], output)
        by_index = {r["scene_idx"]: r for r in cached}
        if len(by_index) != len(cached):
            raise ExtractionStepError(f"duplicate cached scenes: {output}")
        missing = []
        for row in rows:
            saved = by_index.get(row["scene_idx"])
            if saved is None or "error" in saved or saved.get("semantic_warnings"):
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
    repetition_penalty=None, presence_penalty=None,
):
    if model == "minicpm":
        if representation != "graph":
            raise ValueError("MiniCPM supports only Graph scene extraction")
        if arm is None:
            arm = "graph_minicpm"
    if arm is None:
        raise ValueError("extraction requires --arm")
    arm = resolve_generation_arm(context.config, arm, representation, model)
    stage = f"extract-{representation}-scenes"
    path = context.prompt_path(schema)
    log_step_start(
        context, stage, model=model, schema=path, force=force, arm=arm.name,
        **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
        **({"presence_penalty": presence_penalty} if presence_penalty is not None else {}),
    )
    context.initialize()
    prompt = path.read_text(encoding="utf-8")
    print("[PREPARE] Reading prompt/model settings and cohort...", flush=True)
    settings = context.config["extraction"]
    provenance = prompt_provenance(
        context, path, arm, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
    )
    penalties = provenance["settings"].get("repetition_penalty", [1.0])
    scene_dir = context.extraction_dir(arm.representation, model, "scenes")
    from extraction.scene_storage import ensure_generation_identity

    ensure_generation_identity(scene_dir, provenance, force=force, strict=model == "minicpm")
    if representation == "graph":
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
        "error" not in row and not row.get("semantic_warnings")
        for records in existing.values() for row in records
    )

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
                source=model,
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
    if representation == "graph":
        stats = {
            "total": total + reused,
            "success": 0,
            "warning": 0,
            "failed": failure_count,
            "warning_counts": dict.fromkeys(WARNING_TAGS, 0),
        }
        for records in existing.values():
            for record in records:
                if "error" in record:
                    continue
                tags = record["semantic_warnings"]
                stats["warning" if tags else "success"] += 1
                for tag in tags:
                    stats["warning_counts"][tag] += 1
        atomic_write_json(scene_dir / "generation_stats.json", stats, durable=True)
    return result(
        f"{stage}-{model}", content_count=len(visuals), failure_count=failure_count
    )


def _summarize(
    context, *, representation, model, schema, force=False, arm=None,
    repetition_penalty=None, presence_penalty=None,
):
    if model not in (*QWEN_MODELS, "gemini"):
        raise ValueError("summary model must be qwen3vl, qwen3.5 or gemini")
    if arm is None:
        raise ValueError("summarization requires --arm")
    arm = resolve_generation_arm(context.config, arm, representation, model, summary=True)
    path = context.prompt_path(schema)
    stage = f"summarize-{representation}"
    log_step_start(
        context, stage, source=arm.model, model=model, schema=path, force=force, arm=arm.name,
        **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
        **({"presence_penalty": presence_penalty} if presence_penalty is not None else {}),
    )
    from extraction.summary_storage import validate_summary_template

    validate_summary_template(path.read_text(), arm.uses_title)
    context.initialize()
    cohort = context.require_ready_cohort()
    provenance = prompt_provenance(
        context, path, arm, summary=True, model=model, repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
    )
    return run_summary_stage(
        context,
        arm=arm,
        schema=path,
        catalog=cohort["catalog"],
        provenance=provenance,
        generation=provenance["settings"].get("generation", {}),
        model=model,
        gemini_pool_factory=GeminiWorkerPool,
        force=force,
        generator_factory=qwen_generator,
    )


def extract_graph_scenes(
    context, *, model, schema, force=False, arm=None, repetition_penalty=None, presence_penalty=None,
):
    return _extract(
        context, representation="graph", model=model, schema=schema, force=force, arm=arm,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
    )


def extract_description_scenes(
    context, *, model, schema, force=False, arm=None, repetition_penalty=None, presence_penalty=None,
):
    return _extract(
        context, representation="description", model=model, schema=schema, force=force, arm=arm,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
    )


def summarize(
    context, *, model, schema, arm=None, force=False, repetition_penalty=None, presence_penalty=None,
):
    if arm is None:
        raise ValueError("summarize requires --arm")
    selected = generation_registry(context.config).get(arm)
    representation = selected.representation if selected else None
    return _summarize(
        context, representation=representation, model=model, schema=schema, force=force, arm=arm,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
    )


STEP_HANDLERS = {
    "extract-graph-scenes": extract_graph_scenes,
    "extract-description-scenes": extract_description_scenes,
    "summarize": summarize,
}
