from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from validation.cohort import prepare_cohort
from validation.config import ValidationConfig
from validation.diagnosis import diagnose_recommendations
from validation.features import encode_bge_texts, encoder_file_manifest
from validation.recommendation import train_recommendation_arms
from viewing_context_pipeline.runtime import (
    RunContext,
    directory_fingerprint,
    file_fingerprint,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
)


class ValidationStepError(RuntimeError):
    pass


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
            "representations_manifest": str(context.representations_manifest),
            "recommendations_manifest": str(context.recommendations_manifest),
            "diagnosis": str(context.diagnosis_path),
        },
    }


def _require_stage(context: RunContext, stage: str) -> dict[str, Any]:
    path = context.stage_manifest(stage)
    if not path.is_file():
        raise ValidationStepError(f"required stage is incomplete: {stage}")
    value = read_json(path)
    if value.get("schema_version") != "step-manifest/v1" or value.get("status") != "complete":
        raise ValidationStepError(f"invalid stage manifest: {path}")
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


def prepare_cohort_step(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    sources = {
        "cohort": fingerprint(context.config["validation"]["cohort"]),
        "pairs": file_fingerprint(context.path("data", "pairs_tsv")),
        "videos": directory_fingerprint(context.path("data", "videos_dir")),
    }
    if not force:
        current = _current(context, "prepare-cohort", sources, [context.cohort_dir / "cohort_manifest.json"])
        if current is not None:
            return current
    manifest = prepare_cohort(validation_config(context), output_dir=context.cohort_dir)
    return _write_stage(
        context,
        "prepare-cohort",
        source_fingerprints=sources,
        output_fingerprint=manifest["cohort_fingerprint"],
        content_count=manifest["catalog_size"],
    )


def embed_representations(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    graph_qwen = _require_stage(context, "summarize-graph-qwen")
    graph_gemini = _require_stage(context, "summarize-graph-gemini")
    description = _require_stage(context, "summarize-description")
    encoder_fp = directory_fingerprint(context.path("models", "bge"))
    sources_fp = {
        "summarize-graph-qwen": graph_qwen["output_fingerprint"],
        "summarize-graph-gemini": graph_gemini["output_fingerprint"],
        "summarize-description": description["output_fingerprint"],
        "encoder": encoder_fp,
    }
    if not force:
        required = [context.representations_manifest]
        if context.representations_manifest.is_file():
            existing = read_json(context.representations_manifest)
            branches = existing.get("branches")
            if isinstance(branches, dict):
                required.extend(Path(row["path"]) for row in branches.values() if isinstance(row, dict) and "path" in row)
                required.append(context.representations_manifest.parent / "item_index.json")
        current = _current(context, "embed-representations", sources_fp, required)
        if current is not None:
            return current
    config = validation_config(context)
    catalog = read_jsonl(context.cohort_dir / "catalog.jsonl")
    content_ids = [row["content_id"] for row in catalog]
    content_order_fp = fingerprint(content_ids)
    sources = {
        "graph_qwen": (
            context.graph_summary_dir("qwen"),
            "graph-video-summary/v1",
            "qwen",
        ),
        "graph_gemini": (
            context.graph_summary_dir("gemini"),
            "graph-video-summary/v1",
            "gemini",
        ),
        "desc": (
            context.description_summary_dir,
            "description-video-summary/v1",
            None,
        ),
    }
    output = context.representations_manifest.parent
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = (
        read_json(context.representations_manifest)
        if context.representations_manifest.is_file()
        else {}
    )
    previous_branches = previous_manifest.get("branches", {})
    if not isinstance(previous_branches, dict):
        previous_branches = {}
    branches: dict[str, Any] = {}
    evidence_by_content: dict[str, str] = {}
    for branch, (directory, schema, graph_source) in sources.items():
        documents = [read_json(directory / f"{content_id}.json") for content_id in content_ids]
        if any(row.get("schema_version") != schema or row.get("status") != "complete" for row in documents):
            raise ValidationStepError(f"invalid {branch} summaries")
        if graph_source is not None and any(
            row.get("graph_source") != graph_source
            or row.get("summary_model") != "qwen"
            for row in documents
        ):
            raise ValidationStepError(f"graph provenance mismatch for {branch}")
        for row in documents:
            previous = evidence_by_content.setdefault(row["content_id"], row["evidence_fingerprint"])
            if previous != row["evidence_fingerprint"]:
                raise ValidationStepError(f"evidence fingerprint mismatch for {row['content_id']}")
        branch_source_fp = fingerprint(documents)
        previous_branch = previous_branches.get(branch)
        if (
            not force
            and isinstance(previous_branch, dict)
            and previous_branch.get("source_fingerprint") == branch_source_fp
            and previous_branch.get("encoder_fingerprint") == encoder_fp
            and previous_branch.get("content_order_fingerprint") == content_order_fp
            and _valid_embedding_artifact(
                previous_branch,
                row_count=len(catalog),
                dimension=config.encoder.embedding_dim,
            )
        ):
            branches[branch] = previous_branch
            continue
        matrix = np.asarray(encode_bge_texts(config.encoder, [row["text"] for row in documents]), dtype=np.float32)
        if matrix.shape != (len(catalog), config.encoder.embedding_dim) or not np.isfinite(matrix).all():
            raise ValidationStepError(f"invalid embedding matrix for {branch}: {matrix.shape}")
        path = output / f"{branch}_embeddings.npz"
        np.savez_compressed(path, values=matrix)
        branches[branch] = {
            "path": str(path),
            "source_fingerprint": branch_source_fp,
            "encoder_fingerprint": encoder_fp,
            "content_order_fingerprint": content_order_fp,
            "artifact_fingerprint": file_fingerprint(path),
        }
    item_index = {row["item_id"]: index for index, row in enumerate(catalog)}
    write_json(output / "item_index.json", item_index)
    encoder = {
        "model_path": str(config.encoder.model_path),
        "dimension": config.encoder.embedding_dim,
        "files": encoder_file_manifest(config.encoder.model_path),
    }
    document = {
        "schema_version": "representations/v1",
        "run_id": context.run_id,
        "modality": "visual_only",
        "catalog_size": len(catalog),
        "dimension": config.encoder.embedding_dim,
        "branches": branches,
        "encoder": encoder,
        "complete": True,
    }
    document["content_order_fingerprint"] = content_order_fp
    document["fingerprint"] = fingerprint({
        "branches": branches,
        "encoder": encoder,
        "content_order_fingerprint": content_order_fp,
    })
    write_json(context.representations_manifest, document)
    return _write_stage(
        context,
        "embed-representations",
        source_fingerprints=sources_fp,
        output_fingerprint=document["fingerprint"],
        content_count=len(catalog),
    )


def _valid_embedding_artifact(
    branch: dict[str, Any],
    *,
    row_count: int,
    dimension: int,
) -> bool:
    path = Path(str(branch.get("path", "")))
    expected_fingerprint = branch.get("artifact_fingerprint")
    if not path.is_file() or not isinstance(expected_fingerprint, str):
        return False
    if file_fingerprint(path) != expected_fingerprint:
        return False
    try:
        with np.load(path) as payload:
            values = payload["values"]
            return values.shape == (row_count, dimension) and np.isfinite(values).all()
    except (OSError, KeyError, ValueError):
        return False


def run_recommendation(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    source = _require_stage(context, "embed-representations")
    sources_fp = {"embed-representations": source["output_fingerprint"]}
    if not force:
        current = _current(context, "run-recommendation", sources_fp, [context.recommendations_manifest])
        if current is not None:
            existing = read_json(context.recommendations_manifest)
            runs = existing.get("runs")
            metrics = Path(str(existing.get("per_user_metrics", "")))
            if (
                existing.get("schema_version") == "recommendations/v1"
                and isinstance(runs, list)
                and metrics.is_file()
                and all(Path(row["checkpoint"]).is_file() for row in runs)
            ):
                return current
    document = train_recommendation_arms(validation_config(context), _runtime(context))
    write_json(context.recommendations_manifest, document)
    return _write_stage(
        context,
        "run-recommendation",
        source_fingerprints=sources_fp,
        output_fingerprint=document["fingerprint"],
    )


def run_diagnosis(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    source = _require_stage(context, "run-recommendation")
    sources_fp = {"run-recommendation": source["output_fingerprint"]}
    if not force:
        current = _current(context, "run-diagnosis", sources_fp, [context.diagnosis_path])
        if current is not None:
            existing = read_json(context.diagnosis_path)
            if existing.get("schema_version") == "diagnosis/v1" and existing.get("report_ready") is True:
                return current
    document = diagnose_recommendations(validation_config(context), _runtime(context))
    write_json(context.diagnosis_path, document)
    return _write_stage(
        context,
        "run-diagnosis",
        source_fingerprints=sources_fp,
        output_fingerprint=document["fingerprint"],
    )


STEP_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "prepare-cohort": prepare_cohort_step,
    "embed-representations": embed_representations,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
