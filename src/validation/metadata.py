"""Explicit missing-title provenance; no invented text or removed catalog rows."""

from __future__ import annotations

import numpy as np

from pipeline_runtime import read_json


def missing_metadata_report(titles):
    missing = [
        {"item_id": row["item_id"], "content_id": row["content_id"], "embedding_row": index}
        for index, row in enumerate(titles)
        if not row["title"].strip()
    ]
    return {
        "schema_version": "metadata-missing/v1",
        "policy": "zero_vector",
        "catalog_size": len(titles),
        "missing_count": len(missing),
        "items": missing,
    }


def verify_missing_metadata(context, cohort):
    expected = missing_metadata_report(cohort["metadata_titles"])
    actual = read_json(context.cohort_dir / "metadata_missing.json")
    if actual != expected:
        raise RuntimeError("metadata missing-title report does not match catalog titles")
    with np.load(context.representations_dir / "metadata_embeddings.npz") as data:
        values = data["values"]
        if values.shape != (
            len(cohort["catalog"]),
            context.config["validation"]["encoder"]["embedding_dim"],
        ):
            raise RuntimeError("metadata embedding shape does not match catalog")
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite metadata embeddings")
        for row in expected["items"]:
            if np.any(values[row["embedding_row"]] != 0):
                raise RuntimeError(
                    f"missing metadata requires a zero vector: item {row['item_id']}"
                )
    return actual
