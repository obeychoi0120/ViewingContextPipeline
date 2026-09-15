"""Require both actual inputs and encoded values to match recorded provenance."""

from arm_registry import select_arms
from validation.metadata import verify_missing_metadata
from validation.representation_inputs import documents_for_arm, representation_signature
from validation.representation_provenance import matrix_hash, pending_write, read_state


def verify_representations(context, cohort=None, *, arms=None):
    from validation.steps import _representations_match_catalog

    cohort = cohort or context.require_ready_cohort()
    selected = select_arms(context.config, list(arms) if arms is not None else None)
    paths = [context.representations_dir / f"{name}_embeddings.npz" for name in selected]
    if not _representations_match_catalog(
        context.representations_dir / "item_index.json",
        paths,
        cohort["catalog"],
        context.config["validation"]["encoder"]["embedding_dim"],
    ):
        raise RuntimeError("invalid catalog embedding mapping or values")
    if "metadata" in selected:
        verify_missing_metadata(context, cohort)
    verify_recorded_representations(context, cohort, arms=selected)


def verify_recorded_representations(context, cohort, *, arms=None):
    selected = select_arms(context.config, list(arms) if arms is not None else None)
    for name, arm in selected.items():
        state = read_state(context, name)
        if not state or pending_write(context, name).exists():
            raise RuntimeError(f"missing/unfinished embedding provenance: {name}; rerun embedding")
        path = context.representations_dir / f"{name}_embeddings.npz"
        if state.get("embedding_hash") != matrix_hash(path):
            raise RuntimeError(f"embedding values changed without provenance: {name}")
        docs = documents_for_arm(context, cohort, arm)
        if state.get("input_hash") != representation_signature(
            context, cohort["catalog"], arm, docs
        ):
            raise RuntimeError(f"stale representation inputs: {name}; rerun embedding")
