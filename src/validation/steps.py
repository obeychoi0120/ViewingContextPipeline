from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from validation.cohort import prepare_cohort
from validation.config import ValidationConfig
from validation.recommendation_contracts import (
    ARCHITECTURE_VERSION,
    RECOMMENDATION_ARMS,
    TRAINING_RUNS_FILENAME,
    TRAINING_RUN_SCHEMA_VERSION,
)
from pipeline_runtime import RunContext, read_json, read_jsonl, write_json


class ValidationStepError(RuntimeError):
    pass


def validation_config(context: RunContext) -> ValidationConfig:
    settings = context.config["validation"]
    return ValidationConfig.model_validate(
        {
            "schema_version": "validation-config/v2",
            "run_id": context.run_id,
            "dataset": {
                "pairs_tsv": context.path("data", "pairs_tsv"),
                "videos_dir": context.path("data", "videos_dir"),
                "titles_csv": context.path("data", "titles_csv"),
            },
            "cohort": settings["cohort"],
            "encoder": {
                **settings["encoder"],
                "model_path": context.path("models", "bge"),
            },
            "model": settings["model"],
            "evaluation": settings["evaluation"],
            "output_dir": context.run_root,
        }
    )


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
    metadata_titles_path = context.cohort_dir / "metadata_titles.jsonl"
    if (
        catalog_path.is_file()
        and sequences_path.is_file()
        and metadata_titles_path.is_file()
        and not force
    ):
        catalog = read_jsonl(catalog_path)
        if _metadata_titles_match_catalog(metadata_titles_path, catalog):
            return _result("prepare-cohort", content_count=len(catalog))
    result = prepare_cohort(validation_config(context), output_dir=context.cohort_dir)
    return _result("prepare-cohort", content_count=int(result["catalog_size"]))


def _embedding_path(context: RunContext, branch: str) -> Path:
    return context.representations_dir / f"{branch}_embeddings.npz"


def _metadata_titles_match_catalog(
    path: Path,
    catalog: list[dict[str, Any]],
) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        return False
    return len(rows) == len(catalog) and all(
        set(title_row) == {"item_id", "content_id", "title"}
        and str(title_row["item_id"]) == str(catalog_row["item_id"])
        and str(title_row["content_id"]) == str(catalog_row["content_id"])
        and isinstance(title_row["title"], str)
        and bool(title_row["title"].strip())
        for title_row, catalog_row in zip(rows, catalog, strict=True)
    )


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
    from validation.features import BGETextEncoder

    context.initialize()
    config = validation_config(context)
    catalog_path = _require_file(context.cohort_dir / "catalog.jsonl", "cohort catalog")
    catalog = read_jsonl(catalog_path)
    content_ids = [str(row["content_id"]) for row in catalog]
    sources = {
        "metadata": None,
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
        if branch == "metadata":
            metadata_titles_path = context.cohort_dir / "metadata_titles.jsonl"
            if not _metadata_titles_match_catalog(metadata_titles_path, catalog):
                raise ValidationStepError(
                    "metadata titles do not match the cohort catalog; rerun prepare-cohort"
                )
            documents_by_branch[branch] = [
                {
                    "content_id": row["content_id"],
                    "text": row["title"],
                }
                for row in read_jsonl(metadata_titles_path)
            ]
            continue
        directory = sources[branch]
        assert directory is not None
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
        del documents_by_branch[branch]

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
        context.recommendations_dir / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
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
            and row.get("architecture_version") == ARCHITECTURE_VERSION
            and row.get("run_id") == run_id
            and isinstance(row.get("selection"), dict)
            and isinstance(row["selection"].get("epochs"), list)
            and bool(row["selection"]["epochs"])
            and isinstance(row.get("refit"), dict)
            and isinstance(row["refit"].get("epochs"), list)
            and bool(row["refit"]["epochs"])
            and row["refit"].get("epochs_completed")
            == row["selection"].get("best_validation", {}).get("epoch")
            for row in rows
        )
    )


def run_recommendation(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    from validation.recommendation import train_recommendation_arms

    context.initialize()
    config = validation_config(context)
    for branch in ("metadata", "graph_qwen", "graph_gemini", "desc"):
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
    from validation.diagnosis import diagnose_recommendations

    context.initialize()
    config = validation_config(context)
    decision_config = {
        "min_scene_coverage": config.evaluation.min_scene_coverage,
        "max_arm_coverage_gap": config.evaluation.max_arm_coverage_gap,
        "familywise_alpha": config.evaluation.familywise_alpha,
        "multiple_comparison_correction": (config.evaluation.multiple_comparison_correction),
    }
    document = diagnose_recommendations(config, _runtime(context), decision_config)
    write_json(context.diagnosis_path, document)
    decision = document.get("runtime_decision", {})
    if decision.get("status") != "pass":
        errors = decision.get("errors", [])
        error_codes = [
            str(error.get("code", "unknown")) for error in errors if isinstance(error, dict)
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
