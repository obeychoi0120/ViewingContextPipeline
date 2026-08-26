from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from validation.cohort import prepare_cohort
from validation.config import ValidationConfig
from validation.diagnosis import diagnose_recommendations
from validation.features import encode_bge_texts
from validation.recommendation import RECOMMENDATION_ARMS, train_recommendation_arms
from viewing_context_pipeline.runtime import RunContext, read_json, read_jsonl, write_json


class ValidationStepError(RuntimeError):
    pass


def validation_config(context: RunContext) -> ValidationConfig:
    settings = context.config["validation"]
    return ValidationConfig.model_validate({
        "schema_version": "validation-config/v1",
        "run_id": context.run_id,
        "dataset": {
            "pairs_tsv": context.path("data", "pairs_tsv"),
            "videos_dir": context.path("data", "videos_dir"),
        },
        "cohort": settings["cohort"],
        "encoder": {
            **settings["encoder"],
            "model_path": context.path("models", "bge"),
        },
        "model": settings["model"],
        "evaluation": settings["evaluation"],
        "output_dir": context.run_root,
    })


def _runtime(context: RunContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "run_root": str(context.run_root),
        "modality": "visual_only",
        "paths": {
            "cohort_dir": str(context.cohort_dir),
            "representations_dir": str(context.representations_dir),
            "recommendations_dir": str(context.recommendations_dir),
            "diagnosis": str(context.diagnosis_path),
        },
    }


def _result(stage: str, *, content_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": stage}
    if content_count is not None:
        result["content_count"] = content_count
    return result


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValidationStepError(f"missing {label}: {path}")
    return path


def prepare_cohort_step(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    catalog_path = context.cohort_dir / "catalog.jsonl"
    sequences_path = context.cohort_dir / "sequences.jsonl"
    if catalog_path.is_file() and sequences_path.is_file() and not force:
        return _result("prepare-cohort", content_count=len(read_jsonl(catalog_path)))
    result = prepare_cohort(validation_config(context), output_dir=context.cohort_dir)
    return _result("prepare-cohort", content_count=int(result["catalog_size"]))


def _embedding_path(context: RunContext, branch: str) -> Path:
    return context.representations_dir / f"{branch}_embeddings.npz"


def embed_representations(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    config = validation_config(context)
    catalog_path = _require_file(context.cohort_dir / "catalog.jsonl", "cohort catalog")
    catalog = read_jsonl(catalog_path)
    content_ids = [str(row["content_id"]) for row in catalog]
    sources = {
        "graph_qwen": (context.graph_summary_dir("qwen"), "graph-video-summary/v1", "qwen"),
        "graph_gemini": (context.graph_summary_dir("gemini"), "graph-video-summary/v1", "gemini"),
        "desc": (context.description_summary_dir, "description-video-summary/v1", None),
    }
    item_index_path = context.representations_dir / "item_index.json"
    outputs = [_embedding_path(context, branch) for branch in sources]
    if not force and item_index_path.is_file() and all(path.is_file() for path in outputs):
        return _result("embed-representations", content_count=len(catalog))

    context.representations_dir.mkdir(parents=True, exist_ok=True)
    for branch, (directory, schema, graph_source) in sources.items():
        if not directory.is_dir():
            raise ValidationStepError(f"missing {branch} summary directory: {directory}")
        documents = [
            read_json(_require_file(directory / f"{content_id}.json", f"{branch} summary"))
            for content_id in content_ids
        ]
        if any(row.get("schema_version") != schema or row.get("status") != "complete" for row in documents):
            raise ValidationStepError(f"invalid {branch} summaries in {directory}")
        if graph_source is not None and any(row.get("graph_source") != graph_source for row in documents):
            raise ValidationStepError(f"graph source mismatch for {branch}")
        matrix = np.asarray(
            encode_bge_texts(config.encoder, [str(row["text"]) for row in documents]),
            dtype=np.float32,
        )
        expected_shape = (len(catalog), config.encoder.embedding_dim)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValidationStepError(f"invalid embedding matrix for {branch}: {matrix.shape}")
        np.savez_compressed(_embedding_path(context, branch), values=matrix)

    write_json(
        item_index_path,
        {str(row["item_id"]): index for index, row in enumerate(catalog)},
    )
    return _result("embed-representations", content_count=len(catalog))


def _checkpoint_paths(context: RunContext) -> list[Path]:
    seeds = validation_config(context).model.seeds
    return [
        context.recommendations_dir
        / "checkpoints"
        / f"seed_{seed}"
        / arm.lower()
        / "sasrec.pt"
        for seed in seeds
        for arm in RECOMMENDATION_ARMS
    ]


def run_recommendation(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    for branch in ("graph_qwen", "graph_gemini", "desc"):
        _require_file(_embedding_path(context, branch), f"{branch} embeddings")
    _require_file(context.representations_dir / "item_index.json", "item index")
    metrics_path = context.recommendations_dir / "per_user_metrics.jsonl"
    if not force and metrics_path.is_file() and all(path.is_file() for path in _checkpoint_paths(context)):
        return _result("run-recommendation")
    train_recommendation_arms(validation_config(context), _runtime(context))
    return _result("run-recommendation")


def run_diagnosis(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    _require_file(context.recommendations_dir / "per_user_metrics.jsonl", "per-user metrics")
    for path in _checkpoint_paths(context):
        _require_file(path, "recommendation checkpoint")
    if context.diagnosis_path.is_file() and not force:
        existing = read_json(context.diagnosis_path)
        if existing.get("schema_version") == "diagnosis/v1" and existing.get("report_ready") is True:
            return _result("run-diagnosis")
    document = diagnose_recommendations(validation_config(context), _runtime(context))
    write_json(context.diagnosis_path, document)
    return _result("run-diagnosis")


STEP_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "prepare-cohort": prepare_cohort_step,
    "embed-representations": embed_representations,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
