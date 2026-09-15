from __future__ import annotations
import os
from pathlib import Path
import tempfile
import numpy as np

from arm_registry import select_arms
from pipeline_logging import log_step_start
from pipeline_runtime import read_json, read_jsonl, write_json
from validation.config import build_validation_config
from validation.representation_inputs import documents_for_arm, representation_signature
from validation.representation_provenance import (
    begin_write,
    finish_write,
    matrix_hash,
    pending_write,
    read_state,
)


class ValidationStepError(RuntimeError):
    pass


def validation_config(context):
    return build_validation_config(
        run_id=context.run_id,
        dataset={key: context.path("data", key) for key in context.config["data"]},
        settings=context.config["validation"],
        model_path=context.path("models", "bge"),
        output_dir=context.run_root,
    )


def _embedding_path(context, branch):
    return context.representations_dir / f"{branch}_embeddings.npz"


def _metadata_titles_match_catalog(path, catalog, *, allow_blank=False):
    try:
        titles = read_jsonl(path)
        return len(titles) == len(catalog) and all(
            set(t) == {"item_id", "content_id", "title"}
            and str(t["item_id"]) == str(c["item_id"])
            and str(t["content_id"]) == str(c["content_id"])
            and isinstance(t["title"], str)
            and (allow_blank or bool(t["title"].strip()))
            for t, c in zip(titles, catalog, strict=True)
        )
    except (OSError, ValueError):
        return False


def _representations_match_catalog(item_index_path, outputs, catalog, embedding_dim):
    try:
        if read_json(item_index_path) != {str(row["item_id"]): i for i, row in enumerate(catalog)}:
            return False
        for path in outputs:
            with np.load(path) as arrays:
                matrix = arrays["values"]
                if matrix.shape != (len(catalog), embedding_dim) or not np.isfinite(matrix).all():
                    return False
    except (OSError, ValueError, KeyError):
        return False
    return True


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


def embed_representations(context, *, force=False, target=None):
    from validation.features import BGETextEncoder
    from validation.representation_checks import verify_representations

    log_step_start(context, "embed-representations", force=force, target=target)
    context.initialize()
    cohort = context.require_ready_cohort()
    config = validation_config(context)
    arms = select_arms(context.config, target)
    catalog = cohort["catalog"]
    pending = []
    # Validate all selected input documents before loading the encoder or writing results.
    inputs = {}
    for name, arm in arms.items():
        docs = documents_for_arm(context, cohort, arm)
        signature = representation_signature(context, catalog, arm, docs)
        inputs[name] = (docs, signature)
        state = read_state(context, name)
        path = _embedding_path(context, name)
        matches = _representations_match_catalog(
            context.representations_dir / "item_index.json",
            [path],
            catalog,
            config.encoder.embedding_dim,
        )
        if (
            force
            or not matches
            or not state
            or pending_write(context, name).exists()
            or state.get("input_hash") != signature
            or state.get("embedding_hash") != matrix_hash(path)
        ):
            pending.append(name)
    if pending:
        encoder = BGETextEncoder(config.encoder)
        for name in pending:
            docs, signature = inputs[name]
            # Only missing metadata titles receive zero vectors.
            indices = [i for i, row in enumerate(docs) if row["text"].strip()]
            if name != "metadata" and len(indices) != len(catalog):
                raise ValidationStepError(f"empty visual summary: {name}")
            matrix = np.zeros((len(catalog), config.encoder.embedding_dim), dtype=np.float32)
            if indices:
                encoded = np.asarray(
                    encoder.encode([docs[i]["text"] for i in indices]), dtype=np.float32
                )
                if (
                    encoded.shape != (len(indices), config.encoder.embedding_dim)
                    or not np.isfinite(encoded).all()
                ):
                    raise ValidationStepError(f"invalid embedding values: {name}")
                matrix[indices] = encoded
            path = _embedding_path(context, name)
            previous = matrix_hash(path) if path.is_file() else None
            previous = begin_write(context, name, previous)
            _write_embedding(path, matrix)
            sources = [{k: v for k, v in doc.items() if k != "text"} for doc in docs]
            finish_write(
                context,
                name,
                signature,
                previous,
                sources=sources,
                truncation=getattr(encoder, "last_truncation", None)
                if indices
                else {"text_count": 0, "truncated_count": 0},
            )
        write_json(
            context.representations_dir / "item_index.json",
            {str(row["item_id"]): i for i, row in enumerate(catalog)},
        )
    verify_representations(context, cohort, arms={name: name for name in arms})
    return {
        "stage": "embed-representations",
        "content_count": len(catalog),
        "generated_arms": pending,
        "selected_arms": list(arms),
    }


def run_recommendation(context, *, force=False, workers_per_gpu=1, target=None):
    from validation.rolling_recommendation import run_rolling

    log_step_start(
        context,
        "run-recommendation",
        force=force,
        target=target,
        workers_per_gpu=workers_per_gpu,
    )
    context.initialize()
    return run_rolling(
        context, force=force, workers_per_gpu=workers_per_gpu, target=target
    )


def run_diagnosis(context, *, force=False, target=None, compare_run_id=None):
    from validation.rolling_diagnosis import diagnose

    log_step_start(context, "run-diagnosis", force=force, target=target)
    return diagnose(context, target=target, compare_run_id=compare_run_id)


STEP_HANDLERS = {
    "embed-representations": embed_representations,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
