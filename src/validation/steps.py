from __future__ import annotations
import os
from pathlib import Path
import tempfile
import numpy as np
from validation.selection import prepare_validation_cohort, validation_arms
from pipeline_logging import log_step_start
from artifact_io import read_json, write_json
from validation.config import build_validation_config
from validation.text_features import validate_arrays, encode_components
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


def _representations_match_catalog(item_index_path, outputs, catalog, embedding_dim):
    try:
        if read_json(item_index_path) != {str(row["item_id"]): i for i, row in enumerate(catalog)}:
            return False
        for path in outputs:
            with np.load(path) as arrays:
                validate_arrays(arrays, len(catalog), embedding_dim)
    except (OSError, ValueError, KeyError):
        return False
    return True


def _write_embedding(path: Path, matrix: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **matrix)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def embed_representations(context, *, force=False, target=None, representation_mode="text"):
    from validation.graph_context import graph_context, validate_mode

    validate_mode(representation_mode, None, "embed-representations", target)
    if representation_mode == "graph":
        from validation.graph_inputs import prepare

        if not target:
            raise ValueError("--target is required")
        return prepare(graph_context(context), target=target, force=force)
    if target is None:
        raise ValueError("--target is required; select arms explicitly")
    from validation.features import BGETextEncoder
    from validation.representation_checks import verify_representations

    log_step_start(context, "embed-representations", force=force, target=target)
    context.initialize()
    arms = validation_arms(context, target)
    cohort = prepare_validation_cohort(context)
    config = validation_config(context)
    catalog = cohort["catalog"]
    from validation.shared_cache import SharedCache
    from validation.cache_identity import shareable_document

    pending = []
    reused = {"local": [], "shared": []}
    encoder = None
    component_cache = {}
    inputs = {}
    for name, arm in arms.items():
        docs = documents_for_arm(context, cohort, arm)
        signature = representation_signature(
            context, catalog, arm, docs, selection_hash=cohort["manifest"]["selection_hash"]
        )
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
            or (not state)
            or pending_write(context, name).exists()
            or (state.get("input_hash") != signature)
            or (state.get("embedding_hash") != matrix_hash(path))
        ):
            pending.append(name)
        else:
            reused["local"].append(name)
    generated = []
    for name in arms:
        docs, signature = inputs[name]
        eligible = name in {"meta", "metadata"} or all((shareable_document(d) for d in docs))
        cache = SharedCache(context, "embeddings", signature)
        state = read_state(context, name)
        truncation = state.get("truncation")
        origin = {
            "run_id": context.run_id,
            "kind": "local",
            "key": signature,
            "reused_from": (state.get("cache") or {}).get("reused_from")
            or (state.get("cache") if (state.get("cache") or {}).get("kind") == "shared" else None),
        }
        if name in pending:
            restored = None
            if not force and eligible:
                with tempfile.TemporaryDirectory() as temporary:
                    temporary = Path(temporary)
                    restored = cache.restore(temporary)
                    if restored:
                        try:
                            with np.load(temporary / "values.npz") as arrays:
                                matrix = {key: arrays[key] for key in arrays.files}
                                validate_arrays(
                                    matrix, len(catalog), config.encoder.embedding_dim, docs
                                )
                            _write_embedding(_embedding_path(context, name), matrix)
                        except (OSError, ValueError, KeyError):
                            restored = None
            if restored:
                reused["shared"].append(name)
                origin = {"kind": "shared", "key": signature, **restored["origin"]}
                truncation = restored["origin"].get("truncation")
            else:
                generated.append(name)
                origin = {"run_id": context.run_id, "kind": "generated", "key": signature}
                if encoder is None and any((d["text"].strip() for d in docs)):
                    encoder = BGETextEncoder(config.encoder)
                truncation = _encode_arm(context, config, name, docs, encoder, component_cache)

        finish_write(
            context,
            name,
            signature,
            None,
            sources=[
                {**{k: v for k, v in d.items() if k != "text"}, "empty": not d["text"].strip()}
                for d in docs
            ],
            selection_hash=cohort["manifest"]["selection_hash"],
            cache=origin,
            shareable=eligible,
            truncation=truncation,
        )
        if eligible:
            cache.publish(
                context.representations_dir,
                {"values.npz": f"{name}_embeddings.npz"},
                origin={"run_id": context.run_id, "truncation": truncation},
                replace_corrupt=not force,
            )
    write_json(
        context.representations_dir / "item_index.json",
        {str(row["item_id"]): i for i, row in enumerate(catalog)},
    )
    verify_representations(context, cohort, arms={name: name for name in arms})
    return {
        "stage": "embed-representations",
        "content_count": len(catalog),
        "generated_arms": generated,
        "selected_arms": list(arms),
        "reuse": reused,
    }


def _encode_arm(context, config, name, docs, encoder, component_cache=None):
    matrix, truncation = encode_components(
        docs,
        config.encoder.embedding_dim,
        encoder,
        component_cache if component_cache is not None else {},
    )
    path = _embedding_path(context, name)
    try:
        previous = matrix_hash(path) if path.is_file() else None
    except (OSError, ValueError):
        previous = None
    begin_write(context, name, previous)
    _write_embedding(path, matrix)
    return truncation


def run_recommendation(
    context,
    *,
    force=False,
    workers_per_gpu=1,
    target=None,
    representation_mode="text",
    scene_aggregation=None,
    profile_every=None,
):
    from validation.graph_context import graph_context, validate_mode

    validate_mode(representation_mode, scene_aggregation, "run-recommendation", target)
    if representation_mode == "graph":
        context = graph_context(context, scene_aggregation)
    if target is None:
        raise ValueError("--target is required; select arms explicitly")
    from validation.rolling_recommendation import run_rolling

    log_step_start(
        context,
        "run-recommendation",
        force=force,
        target=target,
        workers_per_gpu=workers_per_gpu,
        profile_every=profile_every,
    )
    context.initialize()
    return run_rolling(
        context,
        force=force,
        workers_per_gpu=workers_per_gpu,
        target=target,
        **({"profile_every": profile_every} if profile_every is not None else {}),
    )


def run_diagnosis(
    context,
    *,
    force=False,
    target=None,
    compare_run_id=None,
    representation_mode="text",
    scene_aggregation=None,
):
    from validation.graph_context import graph_context, validate_mode

    validate_mode(representation_mode, scene_aggregation, "run-diagnosis", target)
    if representation_mode == "graph":
        from validation.graph_diagnosis import diagnose_graph

        if not target:
            raise ValueError("--target is required")
        return diagnose_graph(
            graph_context(context, scene_aggregation), target=target, compare_run_id=compare_run_id
        )
    if target is None:
        raise ValueError("--target is required; select arms explicitly")
    from validation.rolling_diagnosis import diagnose

    log_step_start(context, "run-diagnosis", force=force, target=target)
    return diagnose(context, target=target, compare_run_id=compare_run_id)


STEP_HANDLERS = {
    "embed-representations": embed_representations,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
