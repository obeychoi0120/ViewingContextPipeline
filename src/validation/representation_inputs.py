"""Resolve catalog summaries, recording the actual model/path before hashing."""

from __future__ import annotations
from model_provenance import local_model_identity
from artifact_io import fingerprint
from extraction.summary_storage import (
    reuse_summary_document,
    summary_failure_rows,
    summary_model_from_document,
)


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
                    context.run_root / "validation" / "cohort"
                    if "manifest" in cohort
                    else context.cohort_dir
                )
                / "metadata_titles.jsonl"
            ),
        }
        for t in titles
    ]


def _check_model_source(provenance, expected, path):
    backend = provenance.get("settings", {}).get("backend", "")
    model = provenance.get("model", {})
    inferred = (
        "gemini"
        if backend == "gemini" or "model_id" in model
        else "qwen"
        if backend.startswith("vllm")
        else None
    )
    if inferred is not None and inferred != expected:
        raise ValueError(f"generation model provenance mismatch: {path}")


def _source_documents_for_arm(context, cohort, arm, *, failure_rows=None):
    directory = context.summary_arm_dir(arm.name)
    failures = summary_failure_rows(directory) if failure_rows is None else failure_rows
    documents = []
    models = set()
    for item in cohort["catalog"]:
        cid = str(item["content_id"])
        path = directory / f"{cid}.json"
        failure = failures.get(cid)
        if path.exists():
            doc = reuse_summary_document(path, content_id=cid, arm=arm.name)
            prov = doc["provenance"]
            model = summary_model_from_document(doc)
            if model not in {"qwen", "gemini"}:
                raise ValueError(f"summary model provenance mismatch: {path}")
            if prov.get("representation") != arm.representation:
                raise ValueError(f"summary source provenance mismatch: {path}")
            if prov.get("uses_title") != arm.uses_title or prov.get("scene_arm") != arm.scene_arm:
                raise ValueError(f"Summary title/Scene policy mismatch: {path}")
            if not arm.uses_title and "english_title" in prov:
                raise ValueError(f"title present in title-free Summary: {path}")
            failed = failure is not None or doc["status"] in {"failed", "raw_fallback"}
            status = "failed" if failed else "complete"
            text = "" if failed else doc["text"]
            reason = (failure or {}).get("error") or (
                ", ".join(doc.get("violations", [])) if failed else None
            )
        else:
            doc, prov = ({}, (failure or {}).get("provenance", {}))
            model = (failure or {}).get("summary_model")
            status, text = ("failed" if failure else "missing", "")
            reason = (failure or {}).get("error", "summary not generated")
        if failure:
            failure_model = failure.get("summary_model", "qwen")
            if failure_model not in {"qwen", "gemini"}:
                raise ValueError(f"invalid summary failure model: {path}")
            if model and failure_model != model:
                raise ValueError(f"summary failure model mismatch: {path}")
            failure_prov = failure.get("provenance", {})
            if not isinstance(failure_prov, dict):
                raise ValueError(f"invalid summary failure provenance: {path}")
            for key, expected in (
                ("representation", arm.representation),
                ("summary_model", failure_model),
            ):
                if key in failure_prov and failure_prov[key] != expected:
                    raise ValueError(f"summary failure provenance mismatch: {path}")
            for key, expected in (("uses_title", arm.uses_title), ("scene_arm", arm.scene_arm)):
                if key in failure_prov and failure_prov[key] != expected:
                    raise ValueError(f"summary failure policy mismatch: {path}")
            if not arm.uses_title and "english_title" in failure_prov:
                raise ValueError(f"title present in title-free Summary failure: {path}")
            _check_model_source(failure_prov, failure_model, path)
            model = failure_model
        if model:
            models.add(model)
            _check_model_source(prov, model, path)
        for scene in prov.get("scene_provenance", []):
            if (
                not isinstance(scene, dict)
                or scene.get("scene_arm", arm.scene_arm) != arm.scene_arm
                or scene.get("representation", arm.representation) != arm.representation
            ):
                raise ValueError(f"scene source provenance mismatch: {path}")
            _check_model_source(scene, arm.model, path)
        documents.append(
            {
                "content_id": cid,
                "text": text,
                "source_path": str(path),
                "actual_arm": arm.name,
                "source_arm": arm.scene_arm,
                "document_hash": fingerprint(doc) if doc else None,
                "status": status,
                "reason": reason,
                "word_count": len(text.split()),
                "summary_schema": doc.get("schema_version"),
                "summary_policy": prov.get("prompt_hash"),
                "source_provenance": prov,
                "violations": doc.get("violations", []),
            }
        )
    if len(models) > 1:
        raise ValueError(f"multiple Summary models in one Arm: {arm.name}")
    return documents


def documents_for_arm(context, cohort, arm, *, failure_rows=None):
    from arm_registry import generation_registry, CONCAT_POLICY

    titles = None
    if arm.uses_title:
        titles = metadata_documents(context, cohort)
    if arm.model is None:
        docs = titles
    else:
        source = generation_registry(context.config)[arm.scene_arm]
        docs = _source_documents_for_arm(context, cohort, source, failure_rows=failure_rows)
    result = []
    for index, doc in enumerate(docs):
        title = titles[index]["text"].strip() if titles is not None else ""
        summary = doc["text"].strip() if arm.model else ""
        components = (
            "both"
            if title and summary
            else "title_only"
            if title
            else "summary_only"
            if summary
            else "neither"
        )
        text = "\n\n".join((part for part in (title, summary) if part))
        result.append(
            {
                **doc,
                "text": text,
                "actual_arm": arm.name,
                "source_arm": arm.scene_arm,
                "summary_source": arm.scene_arm,
                "composition_policy": CONCAT_POLICY,
                "components": components,
                "title_used": bool(title),
                "title_source_path": titles[index]["source_path"] if titles is not None else None,
                "summary_used": bool(summary),
                "summary_status": doc.get("status") if arm.model else "not_applicable",
                "status": doc.get("status", "complete"),
                "word_count": len(text.split()),
            }
        )
    return result


def representation_signature(context, catalog, arm, documents, *, selection_hash=None):
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
