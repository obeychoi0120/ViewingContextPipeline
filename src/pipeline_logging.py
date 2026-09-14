"""Console-only settings snapshots for pipeline step entry points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_runtime import CONFIG_PATH, RunContext

STEP_CONFIG_PATHS = {
    "prepare-cohort": (
        "protocol.cohort_sampling", "protocol.catalog_scope", "data", "validation.cohort",
    ),
    "prepare-input-data": ("protocol.sampling", "extraction.visual_evidence"),
    "embed-representations": (
        "models.bge", "protocol.arms", "validation.encoder",
        "validation.cohort.metadata_missing_policy",
    ),
    "run-recommendation": (
        "protocol.arms", "validation.model", "validation.cohort.evaluation_days",
        "validation.cohort.timezone", "validation.cohort.exclude_final_day",
        "validation.evaluation.cutoffs", "validation.evaluation.primary_cutoff",
    ),
    "run-diagnosis": (
        "validation.evaluation", "protocol.arms", "validation.model.seeds",
        "validation.cohort", "extraction.visual_evidence.scene_duration",
    ),
}
GENERATION_STEPS = {
    "extract-graph-scenes", "extract-description-scenes", "summarize-graph",
    "summarize-description",
}


def _flatten(values: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(values, f"{prefix}.{key}", child)
    else:
        values[prefix] = value


def _include(values: dict[str, Any], config: dict[str, Any], *paths: str) -> None:
    for path in paths:
        value = config
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                # v3 lacks the v4-only source and rolling cohort settings.
                break
            value = value[key]
        else:
            _flatten(values, path, value)


def step_settings(context: RunContext, step: str, **options: Any) -> dict[str, Any]:
    """Select current in-memory YAML settings without loading inputs or models."""
    if step not in STEP_CONFIG_PATHS and step not in GENERATION_STEPS:
        raise ValueError(f"unknown pipeline step: {step}")
    config = context.config
    values: dict[str, Any] = {
        "run_id": context.run_id,
        "config_file": context.root / CONFIG_PATH,
        "force": options.get("force", False),
    }
    _include(values, config, "schema_version", "protocol.dataset", "artifacts_root")
    if step in GENERATION_STEPS:
        graph = step in {"extract-graph-scenes", "summarize-graph"}
        summary = step.startswith("summarize-")
        arm = "graph" if graph else "description"
        model = options.get("model") if step == "extract-graph-scenes" else "qwen"
        if model not in {"qwen", "gemini"}:
            raise ValueError("extract-graph-scenes requires model=qwen|gemini")
        values["model"] = model
        if step == "summarize-graph":
            if options.get("source") not in {"qwen", "gemini"}:
                raise ValueError("summarize-graph requires source=qwen|gemini")
            values["source"] = options["source"]
        if model == "qwen":
            from extraction.qwen_config import qwen_settings

            values["gpus"] = 1 if options.get("gpus") is None else options["gpus"]
            _include(values, config, "models.qwen")
            _flatten(values, "extraction.qwen", qwen_settings(config["extraction"].get("qwen")))
        else:
            _include(values, config, "models.gemini", "extraction.gemini.threads")
        phase = "summary" if summary else "scene"
        _include(values, config, f"extraction.{arm}.{phase}_prompt")
        if model == "qwen":
            _include(values, config, f"extraction.{arm}.{phase}_max_new_tokens",
                     f"extraction.{'summary' if summary else arm}_repetition_penalty")
        if summary:
            _include(values, config, "extraction.greedy_decoding")
            if not config["extraction"]["greedy_decoding"]:
                _include(values, config, "extraction.summary_sampling")
        else:
            _include(values, config, "extraction.visual_evidence")
        if "scene_concurrency" in options:
            values["scene_concurrency"] = options["scene_concurrency"]
    else:
        _include(values, config, *STEP_CONFIG_PATHS[step])
        if step == "prepare-cohort":
            values["plan_only"] = options.get("plan_only", False)
        elif step == "prepare-input-data":
            values["reuse_run_id"] = options.get("reuse_run_id")
        elif step in {"run-recommendation", "run-diagnosis"}:
            from validation.recommendation_contracts import resolve_target_arms, target_scope

            scope = target_scope(resolve_target_arms(options.get("target")))
            values["target"] = scope["target_sources"]
            values["selected_arms"] = scope["selected_arms"]
            if step == "run-recommendation":
                # Automatic device selection can choose CPU; do not probe CUDA for logging.
                values["gpus"] = "auto" if options.get("gpus") is None else options["gpus"]
                values["workers_per_gpu"] = options.get("workers_per_gpu", 1)
    if step == "extract-graph-scenes":
        output_dir = context.graph_scene_dir(options["model"])
    elif step == "summarize-graph":
        output_dir = context.graph_summary_dir(options["source"])
    else:
        output_dir = {
            "prepare-cohort": context.cohort_dir,
            "prepare-input-data": context.evidence_dir,
            "extract-description-scenes": context.description_scene_dir,
            "summarize-description": context.description_summary_dir,
            "embed-representations": context.representations_dir,
            "run-recommendation": context.recommendations_dir,
            "run-diagnosis": context.diagnosis_path.parent,
        }[step]
    values["output_dir"] = output_dir
    return values


def log_step_start(context: RunContext, step: str, **options: Any) -> None:
    """Print one readable settings block before a step begins work."""
    lines = [f"[STEP] {step}"]
    for key, value in step_settings(context, step, **options).items():
        if isinstance(value, (str, Path)):
            formatted = str(value).replace("\r", "\\r").replace("\n", "\\n")
        else:
            formatted = json.dumps(value, ensure_ascii=False)
        lines.append(f"  {key}: {formatted}")
    print("\n".join(lines), flush=True)
