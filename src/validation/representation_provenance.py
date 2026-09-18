"""Bind regenerated embeddings and recommendation combinations to their actual inputs."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from artifact_io import atomic_write_json
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


def finish_write(context, branch, signature, previous_hash, *, sources=None, truncation=None, summary_source=None, selection_hash=None):
    current = matrix_hash(context.representations_dir / f"{branch}_embeddings.npz")
    atomic_write_json(state_path(context, branch), {
        "input_hash": signature, "embedding_hash": current,
        "selection_hash": selection_hash,
        "recommendation_hash": fingerprint({"input_hash": signature, "embedding_hash": current}),
        "sources": sources or [], "truncation": truncation,
        **({"summary_source": summary_source} if summary_source is not None else {}),
    }, durable=True)
    pending_write(context, branch).unlink(missing_ok=True)


def recommendation_identity(context, branch):
    value = read_state(context, branch).get("recommendation_hash")
    if value is None:
        raise ValueError(f"missing embedding provenance: {branch}; run embed-representations")
    return {"embedding_hash": value}
