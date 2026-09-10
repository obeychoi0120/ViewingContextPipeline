"""Validate representation contents and any recorded source provenance."""

from validation.metadata import verify_missing_metadata
from validation.recommendation_contracts import RECOMMENDATION_ARMS


def verify_representations(context, cohort=None, *, arms=None):
    from validation.steps import _representations_match_catalog

    if cohort is None:
        cohort = context.require_ready_cohort()
    arms = RECOMMENDATION_ARMS if arms is None else arms
    outputs = [
        context.representations_dir / f"{branch}_embeddings.npz"
        for branch in arms.values()
    ]
    if not _representations_match_catalog(
        context.representations_dir / "item_index.json",
        outputs,
        cohort["catalog"],
        context.config["validation"]["encoder"]["embedding_dim"],
    ):
        raise RuntimeError("invalid full catalog embedding mapping or values")
    if "metadata" in arms.values():
        verify_missing_metadata(context, cohort)
    verify_recorded_representations(context, cohort, arms=arms)


def verify_recorded_representations(context, cohort, *, arms=None):
    """Check tracked dependencies without imposing v4 metadata rules on legacy runs."""
    from validation.representation_provenance import matrix_hash, pending_write, read_state
    arms = RECOMMENDATION_ARMS if arms is None else arms
    tracked = False
    for branch in arms.values():
        state = read_state(context, branch)
        tracked = tracked or bool(state)
        if pending_write(context, branch).exists():
            raise RuntimeError(f"unfinished embedding write: {branch}; rerun embed-representations")
        if state and state["embedding_hash"] != matrix_hash(
            context.representations_dir / f"{branch}_embeddings.npz"
        ):
            raise RuntimeError(f"embedding content changed without provenance: {branch}")
    if tracked:
        from extraction.input_tracking import input_state_path
        from validation.steps import _embedding_documents, _embedding_work, validation_config
        from validation.representation_provenance import input_hash, source_changed, summary_sources
        config = validation_config(context)
        sources, fallbacks, _ = _embedding_work(
            context, cohort["catalog"], config, False, branches=set(arms.values())
        )
        fallback_ids = {row["content_id"] for row in fallbacks}
        documents = _embedding_documents(context, cohort["catalog"], sources, list(sources), fallback_ids)
        for branch, docs in documents.items():
            paths = list(summary_sources(context, branch, docs, fallback_ids))
            signature = input_hash(docs, cohort["catalog"], {
                **config.encoder.model_dump(mode="json"), "model": str(context.path("models", "bge")),
            })
            state = read_state(context, branch)
            if (not state and source_changed(paths)
                    or any(input_state_path(path).with_suffix(".dirty").exists() for path in paths)
                    or state and state["input_hash"] != signature):
                raise RuntimeError(f"stale representation inputs: {branch}; rerun summary/embedding steps")
