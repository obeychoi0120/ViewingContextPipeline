"""Bind regenerated embeddings and recommendation combinations to their actual inputs."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from artifact_io import atomic_write_json
from extraction.input_tracking import marker
from extraction.recovery import fingerprint


def state_path(context, branch):
    return context.representations_dir / ".inputs" / f"{branch}.json"


def read_state(context, branch):
    path = state_path(context, branch)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def matrix_hash(path):
    with np.load(path) as arrays:
        digest = hashlib.sha256()
        for name in sorted(arrays.files):
            value = np.ascontiguousarray(arrays[name])
            digest.update(f"{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def input_hash(documents, catalog, encoder):
    return fingerprint({
        "catalog": [str(row["item_id"]) for row in catalog], "encoder": encoder,
        "documents": [{key: row.get(key) for key in ("content_id", "text")}
                      for row in documents],
    })


def pending_write(context, branch):
    return state_path(context, branch).with_suffix(".pending")


def begin_write(context, branch, previous_hash):
    path = pending_write(context, branch)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["previous_hash"]
    atomic_write_json(path, {"previous_hash": previous_hash}, durable=True)
    return previous_hash


def finish_write(context, branch, signature, previous_hash):
    current = matrix_hash(context.representations_dir / f"{branch}_embeddings.npz")
    old = read_state(context, branch)
    generation = old.get("recommendation_hash")
    if previous_hash != current:
        generation = current
    atomic_write_json(state_path(context, branch), {
        "input_hash": signature, "embedding_hash": current,
        "recommendation_hash": generation,
    }, durable=True)
    pending_write(context, branch).unlink(missing_ok=True)


def recommendation_identity(context, branch):
    value = read_state(context, branch).get("recommendation_hash")
    return {"embedding_hash": value} if value is not None else {}


def summary_sources(context, branch, documents, fallback_ids):
    for row in documents:
        content = row["content_id"]
        if branch == "metadata":
            continue
        if branch == "desc":
            directory = context.description_summary_dir
        else:
            source = "qwen" if branch == "graph_qwen" or content in fallback_ids else "gemini"
            directory = context.graph_summary_dir(source)
        yield directory / f"{content}.json"


def source_changed(paths):
    return any(marker(path).exists() for path in paths)
