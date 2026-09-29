"""Explicit missing-title provenance; no invented text or removed catalog rows."""

from __future__ import annotations
from preparation.titles import missing_metadata_report
import numpy as np


def verify_missing_metadata(context, cohort):
    expected = missing_metadata_report(cohort["metadata_titles"])
    name = "meta"
    with np.load(context.representations_dir / f"{name}_embeddings.npz") as data:
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
    return expected
