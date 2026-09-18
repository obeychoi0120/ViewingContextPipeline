"""Resolve catalog summaries, recording the actual model/path before hashing."""

from __future__ import annotations

from dataclasses import asdict

from arm_registry import registry
from model_provenance import local_model_identity
from extraction.recovery import fingerprint
from extraction.summary_executor import reuse_summary_document, summary_failure_rows, summary_model_from_document
from pipeline_runtime import read_json


def documents_for_arm(context, cohort, arm, *, summary_source="qwen", strict=False, failure_rows=None):
    if summary_source not in {"qwen", "gemini"}:
        raise ValueError("summary_source must be qwen or gemini")
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
                "source_path": str((context.run_root / "validation" / "cohort"
                                    if "manifest" in cohort else context.cohort_dir) / "metadata_titles.jsonl"),
            }
            for t in titles
        ]
    registered = registry(context.config)
    documents = []
    summary_failures = {}

    def check_failed_summary(selected, cid):
        directory = context.summary_dir(selected.representation, selected.model, summary_source)
        if directory not in summary_failures:
            summary_failures[directory] = (failure_rows if failure_rows is not None
                                           else summary_failure_rows(directory))
        failure = summary_failures[directory].get(cid)
        if failure and (strict or failure.get("summary_model") == "gemini"):
            raise ValueError(f"unresolved {summary_source.capitalize()} summary failure for {cid} in {directory}")

    for row in catalog:
        cid = str(row["content_id"])
        actual = arm
        check_failed_summary(actual, cid)
        path = (
            context.summary_dir(actual.representation, actual.model, summary_source) / f"{cid}.json"
        )
        # A malformed file is an error. Only absence permits fallback.
        if not strict and not path.exists() and arm.fallback:
            actual = registered[arm.fallback]
            check_failed_summary(actual, cid)
            path = (
                context.summary_dir(actual.representation, actual.model, summary_source)
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
        if strict and doc["status"] != "complete":
            raise ValueError(f"summary is not complete: {path}")
        if summary_model_from_document(doc) != summary_source:
            raise ValueError(f"summary model provenance mismatch: {path}")
        prov = doc["provenance"]
        if prov.get("arm") != actual.name or prov.get("representation") != actual.representation:
            raise ValueError(f"summary source provenance mismatch: {path}")
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


def representation_signature(context, catalog, arm, documents, *, selection_hash=None):
    return fingerprint(
        {
            "arm": asdict(arm),
            "selection_hash": selection_hash,
            "catalog": catalog,
            "encoder": context.config["validation"]["encoder"],
            "model": local_model_identity(context.path("models", "bge")),
            "documents": documents,
        }
    )
