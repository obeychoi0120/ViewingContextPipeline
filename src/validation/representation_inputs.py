"""Resolve catalog metadata and representation identities."""

from __future__ import annotations
from model_provenance import local_model_identity
from artifact_io import fingerprint


def metadata_documents(context, cohort):
    catalog = cohort["catalog"]
    titles = cohort["metadata_titles"]
    if len(titles) != len(catalog) or any(
        (
            str(t.get("item_id")) != str(c["item_id"])
            or str(t.get("content_id")) != str(c["content_id"])
            or (not isinstance(t.get("title"), str))
            for t, c in zip(titles, catalog, strict=True)
        )
    ):
        raise ValueError("metadata titles do not match catalog")
    return [
        {
            "content_id": str(t["content_id"]),
            "text": t["title"],
            "source_path": str(
                (
                    context.validation_dir / "cohort"
                    if "manifest" in cohort
                    else context.cohort_dir
                )
                / "metadata_titles.jsonl"
            ),
        }
        for t in titles
    ]


def representation_signature(context, catalog, arm, documents):
    from validation.cache_identity import REPRESENTATION_VERSION, semantic_documents
    from model_provenance import canonical
    from arm_registry import arm_contract

    return fingerprint(
        {
            "version": REPRESENTATION_VERSION,
            "arm": arm.name,
            **{
                "arm_contract": arm_contract(context.config),
                "uses_title": arm.uses_title,
                "scene_arm": arm.scene_arm,
            },
            "catalog": [{k: r[k] for k in ("item_id", "content_id")} for r in catalog],
            "encoder": context.config["validation"]["encoder"],
            "model": canonical(local_model_identity(context.path("models", "bge"))),
            "documents": semantic_documents(documents),
        }
    )
