from __future__ import annotations

import importlib.util
from functools import partial
import shutil
from typing import Any, Callable

from viewing_context_pipeline.runtime import RunContext, read_json, write_json


STAGES = (
    "prepare-cohort",
    "prepare-input-data",
    "extract-graph-scenes-qwen",
    "summarize-graph-qwen",
    "extract-graph-scenes-gemini",
    "summarize-graph-gemini",
    "extract-description-scenes",
    "summarize-description",
    "embed-representations",
    "run-recommendation",
    "run-diagnosis",
)

DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "prepare-cohort": (),
    "prepare-input-data": ("prepare-cohort",),
    "extract-graph-scenes-qwen": ("prepare-input-data",),
    "summarize-graph-qwen": ("extract-graph-scenes-qwen",),
    "extract-graph-scenes-gemini": ("prepare-input-data",),
    "summarize-graph-gemini": ("extract-graph-scenes-gemini",),
    "extract-description-scenes": ("prepare-input-data",),
    "summarize-description": ("extract-description-scenes",),
    "embed-representations": (
        "summarize-graph-qwen",
        "summarize-graph-gemini",
        "summarize-description",
    ),
    "run-recommendation": ("embed-representations",),
    "run-diagnosis": ("run-recommendation",),
}

GPU_STAGES = {
    "extract-graph-scenes-qwen",
    "summarize-graph-qwen",
    "summarize-graph-gemini",
    "extract-description-scenes",
    "summarize-description",
}


def handlers() -> dict[str, Callable[..., dict[str, Any]]]:
    from extraction.steps import STEP_HANDLERS as extraction_handlers
    from validation.steps import STEP_HANDLERS as validation_handlers

    return {
        "prepare-cohort": validation_handlers["prepare-cohort"],
        "prepare-input-data": extraction_handlers["prepare-input-data"],
        "extract-graph-scenes-qwen": partial(
            extraction_handlers["extract-graph-scenes"], model="qwen"
        ),
        "summarize-graph-qwen": partial(
            extraction_handlers["summarize-graph"], source="qwen"
        ),
        "extract-graph-scenes-gemini": partial(
            extraction_handlers["extract-graph-scenes"], model="gemini"
        ),
        "summarize-graph-gemini": partial(
            extraction_handlers["summarize-graph"], source="gemini"
        ),
        "extract-description-scenes": extraction_handlers[
            "extract-description-scenes"
        ],
        "summarize-description": extraction_handlers["summarize-description"],
        "embed-representations": validation_handlers["embed-representations"],
        "run-recommendation": validation_handlers["run-recommendation"],
        "run-diagnosis": validation_handlers["run-diagnosis"],
    }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _vertex_adc_available() -> bool:
    if not _module_available("google.genai"):
        return False
    try:
        import google.auth

        credentials, _ = google.auth.default()
    except Exception:
        return False
    return credentials is not None


def descendants(stages: set[str]) -> set[str]:
    affected = set(stages)
    changed = True
    while changed:
        changed = False
        for stage, dependencies in DEPENDENCIES.items():
            if stage not in affected and any(dependency in affected for dependency in dependencies):
                affected.add(stage)
                changed = True
    return affected


def invalidate_descendants(
    context: RunContext,
    stages: set[str],
    *,
    include_roots: bool = False,
) -> set[str]:
    affected = descendants(stages)
    targets = affected if include_roots else affected - stages
    for stage in targets:
        context.stage_manifest(stage).unlink(missing_ok=True)
    if context.pipeline_manifest.is_file():
        manifest = read_json(context.pipeline_manifest)
        rows = manifest.get("stages", {})
        if isinstance(rows, dict):
            for stage in targets:
                if stage in rows:
                    rows[stage] = {"status": "stale"}
            write_json(context.pipeline_manifest, manifest)
    return targets


def preflight(context: RunContext) -> dict[str, Any]:
    checks = {
        "data.videos_dir": context.path("data", "videos_dir").is_dir(),
        "data.titles_csv": context.path("data", "titles_csv").is_file(),
        "data.tags_csv": context.path("data", "tags_csv").is_file(),
        "data.pairs_tsv": context.path("data", "pairs_tsv").is_file(),
        "models.qwen": context.path("models", "qwen").is_dir(),
        "models.bge": context.path("models", "bge").is_dir(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "python.torch": _module_available("torch"),
        "python.transformers": _module_available("transformers"),
        "python.google_genai": _module_available("google.genai"),
        "gemini.vertex_adc": _vertex_adc_available(),
    }
    return {
        "schema_version": "pipeline-preflight/v1",
        "run_id": context.run_id,
        "ready": all(checks.values()),
        "checks": checks,
    }


def _pipeline_document(context: RunContext) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for stage in STAGES:
        path = context.stage_manifest(stage)
        if path.is_file():
            value = read_json(path)
            rows[stage] = {
                "status": value.get("status"),
                "output_fingerprint": value.get("output_fingerprint"),
                "manifest": str(path),
            }
        else:
            rows[stage] = {"status": "pending"}
    return {
        "schema_version": "pipeline-run/v1",
        "run_id": context.run_id,
        "protocol": context.config["protocol"],
        "stages": rows,
        "complete": all(row["status"] == "complete" for row in rows.values()),
    }


def execute_stage(
    context: RunContext,
    stage: str,
    *,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    missing = [dependency for dependency in DEPENDENCIES[stage] if not context.stage_manifest(dependency).is_file()]
    if missing:
        raise RuntimeError(f"{stage} requires completed stages: {', '.join(missing)}")
    kwargs: dict[str, Any] = {"force": force}
    if stage in GPU_STAGES and gpus is not None:
        kwargs["gpus"] = gpus
    result = handlers()[stage](context, **kwargs)
    write_json(context.pipeline_manifest, _pipeline_document(context))
    return result


def run_pipeline(
    context: RunContext,
    *,
    dry_run: bool = False,
    force_stages: set[str] | None = None,
    gpus: int | None = None,
) -> int:
    force_stages = set(force_stages or ())
    unknown = force_stages - set(STAGES)
    if unknown:
        raise ValueError(f"unknown force stages: {sorted(unknown)}")
    check = preflight(context)
    if dry_run:
        print({
            "preflight": check,
            "stages": list(STAGES),
            "force_stages": sorted(force_stages),
            "gpus": gpus,
        })
        return 0 if check["ready"] else 1
    if not check["ready"]:
        failed = [name for name, ready in check["checks"].items() if not ready]
        raise RuntimeError("preflight failed: " + ", ".join(failed))
    context.initialize()
    if force_stages:
        invalidate_descendants(context, force_stages, include_roots=True)
    for stage in STAGES:
        execute_stage(context, stage, force=stage in force_stages, gpus=gpus)
    return 0
