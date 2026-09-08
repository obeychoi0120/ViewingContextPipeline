from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from validation.cohort import prepare_cohort
from validation.config import ValidationConfig, build_validation_config
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
    return build_validation_config(
        run_id=context.run_id,
        dataset={
            key: context.path("data", key) for key in context.config["data"]
        },
        settings=context.config["validation"],
        model_path=context.path("models", "bge"),
        output_dir=context.run_root,
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


def prepare_cohort_step(
    context: RunContext, *, force: bool = False, plan_only: bool = False
) -> dict[str, Any]:
    context.initialize()
    if context.config["schema_version"] == "viewing-context-config/v4":
        from validation.rolling_data import prepare_full_cohort
        return prepare_full_cohort(context, plan_only=plan_only)
    prepared = prepare_cohort(
        validation_config(context),
        output_dir=context.cohort_dir,
        plan_only=plan_only,
        force=force,
    )
    return {
        **_result("prepare-cohort", content_count=int(prepared["catalog_size"])),
        "status": prepared["status"],
        "user_count": prepared["user_count"],
        "catalog_scope": prepared["catalog_scope"],
    }


def _embedding_path(context: RunContext, branch: str) -> Path:
    return context.representations_dir / f"{branch}_embeddings.npz"


def _metadata_titles_match_catalog(
    path: Path,
    catalog: list[dict[str, Any]],
    *,
    allow_blank: bool = False,
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
        and (allow_blank or bool(title_row["title"].strip()))
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


def _gemini_fallbacks_match(path: Path, fallbacks: list[dict[str, str]]) -> bool:
    if not path.is_file():
        return not fallbacks
    try:
        return read_json(path).get("fallbacks") == fallbacks
    except (OSError, ValueError):
        return False


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


def _embedding_work(context, catalog, config, force):
    sources = {
        "metadata": None,
        "graph_qwen": context.graph_summary_dir("qwen"),
        "graph_gemini": context.graph_summary_dir("gemini"),
        "desc": context.description_summary_dir,
    }
    gemini_fallbacks = [
        {
            "item_id": str(row["item_id"]),
            "content_id": str(row["content_id"]),
            "source": "graph_qwen",
            "summary_path": (context.graph_summary_dir("qwen") / f"{row['content_id']}.json")
            .relative_to(context.run_root)
            .as_posix(),
        }
        for row in catalog
        if not (context.graph_summary_dir("gemini") / f"{row['content_id']}.json").is_file()
    ]
    fallback_path = context.representations_dir / "graph_gemini_fallbacks.json"
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
        or branch == "graph_gemini"
        and not _gemini_fallbacks_match(fallback_path, gemini_fallbacks)
    ]

    return sources, gemini_fallbacks, pending


def _embedding_documents(context, catalog, sources, pending, fallback_ids):
    content_ids = [str(row["content_id"]) for row in catalog]
    documents_by_branch: dict[str, list[dict[str, Any]]] = {}
    for branch in pending:
        if branch == "metadata":
            metadata_titles_path = context.cohort_dir / "metadata_titles.jsonl"
            if not _metadata_titles_match_catalog(
                metadata_titles_path, catalog,
                allow_blank=context.config["schema_version"] == "viewing-context-config/v4",
            ):
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
        if not directory.is_dir() and branch != "graph_gemini":
            raise ValidationStepError(f"missing {branch} summary directory: {directory}")
        documents = []
        for content_id in content_ids:
            summary_path = directory / f"{content_id}.json"
            label = f"{branch} summary"
            if branch == "graph_gemini" and content_id in fallback_ids:
                label = f"graph_qwen fallback summary (Gemini summary missing: {summary_path})"
                summary_path = context.graph_summary_dir("qwen") / f"{content_id}.json"
            document = read_json(_require_file(summary_path, label))
            if context.config["schema_version"] == "viewing-context-config/v4":
                from extraction.summary_executor import reuse_summary_document
                actual_branch = "graph_qwen" if content_id in fallback_ids and branch == "graph_gemini" else branch
                if actual_branch == "desc":
                    actual_branch = "description"
                scene_count = document.get("scene_count")
                if type(scene_count) is not int or scene_count <= 0:
                    raise ValidationStepError(f"invalid {label} scene count: {summary_path}")
                document = reuse_summary_document(
                    summary_path,
                    schema_version="description-video-summary/v3" if branch == "desc" else "graph-video-summary/v3",
                    content_id=content_id, arm=actual_branch, scene_count=scene_count,
                )
            if (
                str(document.get("content_id")) != content_id
                or not isinstance(document.get("text"), str)
                or not document["text"].strip()
            ):
                raise ValidationStepError(f"invalid {label}: {summary_path}")
            documents.append(document)
        documents_by_branch[branch] = documents

    return documents_by_branch


def _encode_representations(encoder, pending, documents_by_branch, catalog, config):
    matrices: dict[str, np.ndarray] = {}
    for branch in pending:
        documents = documents_by_branch[branch]
        expected_shape = (len(catalog), config.encoder.embedding_dim)
        if branch == "metadata" and config.schema_version == "validation-config/v4":
            # Do not send empty titles through BGE, even as part of a padded batch.
            nonempty = [i for i, row in enumerate(documents) if row["text"].strip()]
            matrix = np.zeros(expected_shape, dtype=np.float32)
            if nonempty:
                encoded = np.asarray(
                    encoder.encode([documents[i]["text"] for i in nonempty]), dtype=np.float32,
                )
                if encoded.shape != (len(nonempty), config.encoder.embedding_dim):
                    raise ValidationStepError(f"invalid metadata embedding matrix: {encoded.shape}")
                matrix[nonempty] = encoded
        else:
            matrix = np.asarray(
                encoder.encode([str(row["text"]) for row in documents]), dtype=np.float32,
            )
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValidationStepError(f"invalid embedding matrix for {branch}: {matrix.shape}")
        matrices[branch] = matrix
        del documents_by_branch[branch]

    return matrices


def _persist_representations(context, matrices, catalog, gemini_fallbacks):
    fallback_path = context.representations_dir / "graph_gemini_fallbacks.json"
    item_index_path = context.representations_dir / "item_index.json"
    for branch, matrix in matrices.items():
        _write_embedding(_embedding_path(context, branch), matrix)
        if branch == "graph_gemini":
            write_json(fallback_path, {"fallbacks": gemini_fallbacks})

    write_json(
        item_index_path,
        {str(row["item_id"]): index for index, row in enumerate(catalog)},
    )


def embed_representations(context: RunContext, *, force: bool = False) -> dict[str, Any]:
    context.initialize()
    cohort = context.require_ready_cohort()
    from validation.features import BGETextEncoder

    config = validation_config(context)
    catalog = cohort["catalog"]
    sources, gemini_fallbacks, pending = _embedding_work(context, catalog, config, force)
    full = context.config["schema_version"] == "viewing-context-config/v4"
    documents_by_branch = _embedding_documents(
        context,
        catalog,
        sources,
        list(sources) if full else pending,
        {row["content_id"] for row in gemini_fallbacks},
    )
    print(
        f"[Embedding_fallback] graph_gemini -> graph_qwen: {len(gemini_fallbacks)} items"
        + (" (cached embeddings)" if "graph_gemini" not in pending else ""),
        flush=True,
    )
    for row in gemini_fallbacks:
        print(
            f"  item_id={row['item_id']} | content_id={row['content_id']} | "
            f"summary={row['summary_path']}",
            flush=True,
        )
    if not pending:
        if full:
            from validation.representation_checks import verify_representations
            verify_representations(context, cohort)
        return _result("embed-representations", content_count=len(catalog))

    context.representations_dir.mkdir(parents=True, exist_ok=True)
    encoder = BGETextEncoder(config.encoder)
    matrices = _encode_representations(encoder, pending, documents_by_branch, catalog, config)
    _persist_representations(context, matrices, catalog, gemini_fallbacks)
    if full:
        from validation.representation_checks import verify_representations
        verify_representations(context, cohort)
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
    context.initialize()
    if context.config["schema_version"] == "viewing-context-config/v4":
        from validation.rolling_recommendation import run_rolling
        return run_rolling(context, force=force)
    context.require_ready_cohort()
    from validation.recommendation import train_recommendation_arms

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
    if context.config["schema_version"] == "viewing-context-config/v4":
        from validation.rolling_diagnosis import diagnose
        return diagnose(context)
    from validation.diagnosis import diagnose_recommendations

    context.initialize()
    config = validation_config(context)
    decision_config = {
        "min_scene_coverage": config.evaluation.min_scene_coverage,
        "max_arm_coverage_gap": config.evaluation.max_arm_coverage_gap,
        "familywise_alpha": config.evaluation.familywise_alpha,
        "multiple_comparison_correction": (config.evaluation.multiple_comparison_correction),
    }
    document = diagnose_recommendations(
        config,
        _runtime(context),
        decision_config,
        scene_duration=context.config["extraction"]["visual_evidence"]["scene_duration"],
    )
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
