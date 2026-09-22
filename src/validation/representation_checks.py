"""Require both actual inputs and encoded values to match recorded provenance."""

from validation.selection import load_validation_cohort, validation_arms, LEGACY_POLICY
from validation.metadata import verify_missing_metadata
from validation.representation_inputs import documents_for_arm, representation_signature
from validation.representation_provenance import matrix_hash, pending_write, read_state


def verify_representations(context, cohort=None, *, arms=None):
    from validation.steps import _representations_match_catalog

    cohort = load_validation_cohort(context)
    selected = validation_arms(context, list(arms) if arms is not None else None)
    paths = [context.representations_dir / f"{name}_embeddings.npz" for name in selected]
    if not _representations_match_catalog(
        context.representations_dir / "item_index.json",
        paths,
        cohort["catalog"],
        context.config["validation"]["encoder"]["embedding_dim"],
    ):
        raise RuntimeError("invalid catalog embedding mapping or values")
    if any(name in selected for name in ("meta", "metadata")):
        verify_missing_metadata(context, cohort)
    verify_recorded_representations(context, cohort, arms=selected)


def verify_recorded_representations(context, cohort, *, arms=None):
    selected = validation_arms(context, list(arms) if arms is not None else None)
    legacy = cohort["manifest"]["policy"] == LEGACY_POLICY
    for name, arm in selected.items():
        state = read_state(context, name)
        if not state or pending_write(context, name).exists():
            raise RuntimeError(f"missing/unfinished embedding provenance: {name}; rerun embedding")
        if state.get("selection_hash") != cohort["manifest"]["selection_hash"]:
            raise RuntimeError(f"stale representation selection: {name}; rerun embedding")
        path = context.representations_dir / f"{name}_embeddings.npz"
        if state.get("embedding_hash") != matrix_hash(path):
            raise RuntimeError(f"embedding values changed without provenance: {name}")
        docs = documents_for_arm(context, cohort, arm,
                                 summary_source=state.get("summary_source", "qwen"), strict=True, legacy=legacy)
        if state.get("input_hash") != representation_signature(
            context, cohort["catalog"], arm, docs,
            selection_hash=cohort["manifest"]["selection_hash"], legacy=legacy
        ):
            raise RuntimeError(f"stale representation inputs: {name}; rerun embedding")

        if not legacy:
            import numpy as np
            with np.load(path) as arrays:
                values = arrays["values"]
                for i, doc in enumerate(docs):
                    if not doc["text"].strip() and np.any(values[i] != 0):
                        raise RuntimeError(f"empty expression must have a zero vector: {name}/{doc['content_id']}")
