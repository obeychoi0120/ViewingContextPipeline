"""Validate representation contents without binding caches to their inputs."""

from validation.metadata import verify_missing_metadata
from validation.recommendation_contracts import RECOMMENDATION_ARMS


def verify_representations(context, cohort=None):
    from validation.steps import _representations_match_catalog

    if cohort is None:
        cohort = context.require_ready_cohort()
    outputs = [
        context.representations_dir / f"{branch}_embeddings.npz"
        for branch in RECOMMENDATION_ARMS.values()
    ]
    if not _representations_match_catalog(
        context.representations_dir / "item_index.json",
        outputs,
        cohort["catalog"],
        context.config["validation"]["encoder"]["embedding_dim"],
    ):
        raise RuntimeError("invalid full catalog embedding mapping or values")
    verify_missing_metadata(context, cohort)
