"""Resolve catalog summaries, recording the actual model/path before hashing."""

from __future__ import annotations
from dataclasses import asdict

from arm_registry import registry
from model_provenance import local_model_identity
from extraction.input_tracking import input_state_path
from extraction.recovery import fingerprint
from extraction.summary_executor import reuse_summary_document
from pipeline_runtime import read_json, read_jsonl


def documents_for_arm(context, cohort, arm):
    catalog = cohort["catalog"]
    if arm.name == "metadata":
        titles = cohort["metadata_titles"]
        if len(titles) != len(catalog) or any(
            str(t.get("item_id")) != str(c["item_id"])
            or str(t.get("content_id")) != str(c["content_id"])
            or not isinstance(t.get("title"), str)
            for t, c in zip(titles, catalog, strict=True)
        ):
            raise ValueError("metadata titles do not match catalog")
        return [
            {
                "content_id": str(t["content_id"]),
                "text": t["title"],
                "source_path": str(context.cohort_dir / "metadata_titles.jsonl"),
            }
            for t in titles
        ]
    registered = registry(context.config)
    documents = []
    for row in catalog:
        cid = str(row["content_id"])
        actual = arm
        path = (
            context.extraction_dir(actual.representation, actual.model, "summaries") / f"{cid}.json"
        )
        # A malformed file is an error. Only absence permits fallback.
        if not path.exists() and arm.fallback:
            actual = registered[arm.fallback]
            path = (
                context.extraction_dir(actual.representation, actual.model, "summaries")
                / f"{cid}.json"
            )
        if not path.is_file():
            raise ValueError(f"missing {arm.name} catalog summary: {path}")
        doc = read_json(path)
        if (
            str(doc.get("content_id")) != cid
            or not isinstance(doc.get("text"), str)
            or not doc["text"].strip()
        ):
            raise ValueError(f"invalid summary identity or text: {path}")
        doc = reuse_summary_document(path, content_id=cid, arm=actual.name)
        prov = doc["provenance"]
        if prov.get("arm") != actual.name or prov.get("representation") != actual.representation:
            raise ValueError(f"summary source provenance mismatch: {path}")
        if input_state_path(path).exists():
            raise ValueError(f"scene inputs changed; regenerate summary: {path}")
        scene_path = (
            context.extraction_dir(actual.representation, actual.model, "scenes") / f"{cid}.jsonl"
        )
        if not scene_path.is_file() or prov.get("scene_input_hash") != fingerprint(
            read_jsonl(scene_path)
        ):
            raise ValueError(f"summary scene input hash mismatch: {path}")
        source_provenance = doc.get("provenance")
        if not isinstance(source_provenance, dict):
            source_provenance = {}
        documents.append(
            {
                "content_id": cid,
                "text": doc["text"],
                "source_path": str(path),
                "actual_arm": actual.name,
                "source_arm": doc.get("arm"),
                "document_hash": fingerprint(doc),
                "status": doc.get("status"),
                "summary_schema": doc.get("schema_version"),
                "summary_policy": source_provenance.get("prompt_hash"),
                "word_count": len(doc["text"].split()),
                "violations": doc.get("violations"),
                "correction_count": doc.get("correction_count"),
                "generation": doc.get("generation"),
                "source_provenance": {
                    key: source_provenance[key]
                    for key in (
                        "prompt_path",
                        "prompt_hash",
                        "model",
                        "schema_contract",
                    )
                    if key in source_provenance
                },
            }
        )
    return documents


def representation_signature(context, catalog, arm, documents):
    return fingerprint(
        {
            "arm": asdict(arm),
            "catalog": catalog,
            "encoder": context.config["validation"]["encoder"],
            "model": local_model_identity(context.path("models", "bge")),
            "documents": documents,
        }
    )
