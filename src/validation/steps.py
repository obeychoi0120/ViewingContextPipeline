from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from validation.cohort import prepare_cohort
from validation.config import ValidationConfig
from validation.diagnosis import diagnose_recommendations
from validation.features import BGETextEncoder
from validation.recommendation import (
    RECOMMENDATION_ARMS,
    TRAINING_RUNS_FILENAME,
    TRAINING_RUN_SCHEMA_VERSION,
    train_recommendation_arms,
)
from pipeline_runtime import RunContext, read_json, read_jsonl, write_json


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


def _representations_match_catalog(
    item_index_path: Path,
    outputs: list[Path],
    catalog: list[dict[str, Any]],
    embedding_dim: int,
) -> bool:
    if not item_index_path.is_file() or not all(path.is_file() for path in outputs):
        return False
    expected_index = {str(row["item_id"]): index for index, row in enumerate(catalog)}
    try:
        if read_json(item_index_path) != expected_index:
            return False
        for path in outputs:
            values = np.load(path)["values"]
            if values.shape != (len(catalog), embedding_dim) or not np.isfinite(values).all():
                return False
    except (OSError, KeyError, ValueError):
        return False
    return True


def _representation_matches_catalog(
    item_index_path: Path,
    output: Path,
    catalog: list[dict[str, Any]],
    embedding_dim: int,
) -> bool:
    return _representations_match_catalog(
        item_index_path,
        [output],
        catalog,
        embedding_dim,
    )


def _write_embedding(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, values=matrix)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def embed_representations(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    config = validation_config(context)
    catalog_path = _require_file(context.cohort_dir / "catalog.jsonl", "cohort catalog")
    catalog = read_jsonl(catalog_path)
    content_ids = [str(row["content_id"]) for row in catalog]
    sources = {
        "graph_qwen": context.graph_summary_dir("qwen"),
        "graph_gemini": context.graph_summary_dir("gemini"),
        "desc": context.description_summary_dir,
    }
    item_index_path = context.representations_dir / "item_index.json"
    pending = [
        branch
        for branch in sources
        if force
        or not _representation_matches_catalog(
            item_index_path,
            _embedding_path(context, branch),
            catalog,
            config.encoder.embedding_dim,
        )
    ]
    if not pending:
        return _result("embed-representations", content_count=len(catalog))

    context.representations_dir.mkdir(parents=True, exist_ok=True)
    documents_by_branch: dict[str, list[dict[str, Any]]] = {}
    for branch in pending:
        directory = sources[branch]
        if not directory.is_dir():
            raise ValidationStepError(f"missing {branch} summary directory: {directory}")
        documents = [
            read_json(_require_file(directory / f"{content_id}.json", f"{branch} summary"))
            for content_id in content_ids
        ]
        if any(
            str(row.get("content_id")) != content_id
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
            for content_id, row in zip(content_ids, documents, strict=True)
        ):
            raise ValidationStepError(f"invalid {branch} summaries in {directory}")
        documents_by_branch[branch] = documents

    encoder = BGETextEncoder(config.encoder)
    matrices: dict[str, np.ndarray] = {}
    for branch in pending:
        documents = documents_by_branch[branch]
        matrix = np.asarray(
            encoder.encode([str(row["text"]) for row in documents]),
            dtype=np.float32,
        )
        expected_shape = (len(catalog), config.encoder.embedding_dim)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValidationStepError(f"invalid embedding matrix for {branch}: {matrix.shape}")
        matrices[branch] = matrix

    for branch, matrix in matrices.items():
        _write_embedding(_embedding_path(context, branch), matrix)

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


def _training_runs_complete(
    path: Path,
    *,
    run_id: str,
    seeds: list[int],
) -> bool:
    if not path.is_file():
        return False
    expected = {(seed, arm) for seed in seeds for arm in RECOMMENDATION_ARMS}
    try:
        rows = read_jsonl(path)
        actual = {(int(row["seed"]), str(row["arm"])) for row in rows}
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return (
        len(rows) == len(expected)
        and actual == expected
        and all(
            row.get("schema_version") == TRAINING_RUN_SCHEMA_VERSION
            and row.get("run_id") == run_id
            and isinstance(row.get("epochs"), list)
            and bool(row["epochs"])
            for row in rows
        )
    )


def run_recommendation(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    config = validation_config(context)
    for branch in ("graph_qwen", "graph_gemini", "desc"):
        _require_file(_embedding_path(context, branch), f"{branch} embeddings")
    _require_file(context.representations_dir / "item_index.json", "item index")
    metrics_path = context.recommendations_dir / "per_user_metrics.jsonl"
    training_runs_path = context.recommendations_dir / TRAINING_RUNS_FILENAME
    if (
        not force
        and metrics_path.is_file()
        and _training_runs_complete(
            training_runs_path,
            run_id=context.run_id,
            seeds=config.model.seeds,
        )
        and all(path.is_file() for path in _checkpoint_paths(context))
    ):
        return _result("run-recommendation")
    train_recommendation_arms(config, _runtime(context))
    return _result("run-recommendation")


def run_diagnosis(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    config = validation_config(context)
    decision_config = {
        "min_scene_coverage": config.evaluation.min_scene_coverage,
        "max_arm_coverage_gap": config.evaluation.max_arm_coverage_gap,
        "familywise_alpha": config.evaluation.familywise_alpha,
        "multiple_comparison_correction": (
            config.evaluation.multiple_comparison_correction
        ),
    }
    document = diagnose_recommendations(config, _runtime(context), decision_config)
    write_json(context.diagnosis_path, document)
    decision = document.get("runtime_decision", {})
    if decision.get("status") != "pass":
        errors = decision.get("errors", [])
        error_codes = [
            str(error.get("code", "unknown"))
            for error in errors
            if isinstance(error, dict)
        ]
        raise ValidationStepError(
            "runtime diagnosis failed: " + ", ".join(error_codes or ["unknown"])
        )
    analysis = document.get("statistical_analysis", {})
    if analysis.get("status") not in {"computed", "computed_with_warnings"}:
        raise ValidationStepError(
            "statistical diagnosis failed; inspect statistical_analysis.errors"
        )
    return _result("run-diagnosis")


STEP_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "prepare-cohort": prepare_cohort_step,
    "embed-representations": embed_representations,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
