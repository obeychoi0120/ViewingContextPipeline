"""Resolve catalog summaries, recording the actual model/path before hashing."""

from __future__ import annotations

from dataclasses import asdict

from arm_registry import registry, legacy_layout, ARM_CONTRACT
from model_provenance import local_model_identity
from extraction.recovery import fingerprint
from extraction.summary_executor import reuse_summary_document, summary_failure_rows, summary_model_from_document
from pipeline_runtime import read_json


def _legacy_documents_for_arm(context, cohort, arm, *, summary_source="qwen", strict=False, failure_rows=None):
    if summary_source not in {"qwen", "gemini"}:
        raise ValueError("summary_source must be qwen or gemini")
    catalog = cohort["catalog"]
    if arm.representation == "metadata":
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
        if prov.get("representation") != actual.representation:
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


def _legacy_representation_signature(context, catalog, arm, documents, *, selection_hash=None):
    return fingerprint(
        {
            "arm": {k: v for k, v in asdict(arm).items() if k in {"name", "representation", "model"}},
            "selection_hash": selection_hash,
            "catalog": catalog,
            "encoder": context.config["validation"]["encoder"],
            "model": local_model_identity(context.path("models", "bge")),
            "documents": documents,
        }
    )


def _check_model_source(provenance, expected, path):
    backend = provenance.get("settings", {}).get("backend", "")
    model = provenance.get("model", {})
    inferred = ("gemini" if backend == "gemini" or "model_id" in model
                else "qwen" if backend.startswith("vllm") else None)
    if inferred is not None and inferred != expected:
        raise ValueError(f"generation model provenance mismatch: {path}")


def documents_for_arm(context, cohort, arm, *, summary_source=None, strict=False, failure_rows=None, legacy=False):
    if legacy or arm.representation == "metadata":
        return _legacy_documents_for_arm(context, cohort, arm, summary_source=summary_source or "qwen",
                                         strict=strict, failure_rows=failure_rows)
    old = legacy_layout(context.config)
    if old:
        summary_source = summary_source or "qwen"
        if summary_source not in {"qwen", "gemini"}:
            raise ValueError("summary_source must be qwen or gemini")
    directory = (context.summary_dir(arm.representation, arm.model, summary_source)
                 if old else context.summary_arm_dir(arm.name))
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
            if model not in {"qwen", "gemini"} or (old and model != summary_source):
                raise ValueError(f"summary model provenance mismatch: {path}")
            binding = doc.get("migration", {})
            if prov.get("representation") != arm.representation:
                raise ValueError(f"summary source provenance mismatch: {path}")
            if not old:
                if binding:
                    if (binding.get("target_arm") != arm.name or binding.get("source_arm") != arm.scene_arm
                            or not arm.uses_title or "english_title" not in prov):
                        raise ValueError(f"invalid migrated Summary binding: {path}")
                elif prov.get("uses_title") != arm.uses_title or prov.get("scene_arm") != arm.scene_arm:
                    raise ValueError(f"Summary title/Scene policy mismatch: {path}")
                if not arm.uses_title and "english_title" in prov:
                    raise ValueError(f"title present in title-free Summary: {path}")
            failed = failure is not None or doc["status"] in {"failed", "raw_fallback"}
            status = "failed" if failed else "complete"
            text = "" if failed else doc["text"]
            reason = (failure or {}).get("error") or (", ".join(doc.get("violations", [])) if failed else None)
        else:
            doc, prov = {}, (failure or {}).get("provenance", {})
            model = (failure or {}).get("summary_model")
            status, text = ("failed" if failure else "missing"), ""
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
            for key, expected in (("representation", arm.representation),
                                  ("summary_model", failure_model)):
                if key in failure_prov and failure_prov[key] != expected:
                    raise ValueError(f"summary failure provenance mismatch: {path}")
            if not old:
                for key, expected in (("uses_title", arm.uses_title), ("scene_arm", arm.scene_arm)):
                    if key in failure_prov and failure_prov[key] != expected:
                        raise ValueError(f"summary failure policy mismatch: {path}")
                if not arm.uses_title and "english_title" in failure_prov:
                    raise ValueError(f"title present in title-free Summary failure: {path}")
            _check_model_source(failure_prov, failure_model, path)
            model = failure_model
        if model:
            models.add(model)
            if old and model != summary_source:
                raise ValueError(f"summary failure model mismatch: {path}")
            _check_model_source(prov, model, path)
        for scene in prov.get("scene_provenance", []):
            if (not isinstance(scene, dict) or scene.get("scene_arm", arm.scene_arm) != arm.scene_arm
                    or scene.get("representation", arm.representation) != arm.representation):
                raise ValueError(f"scene source provenance mismatch: {path}")
            _check_model_source(scene, arm.model, path)
        documents.append({"content_id": cid, "text": text, "source_path": str(path),
                          "actual_arm": arm.name, "source_arm": arm.scene_arm,
                          "document_hash": fingerprint(doc) if doc else None,
                          "status": status, "reason": reason, "word_count": len(text.split()),
                          "summary_schema": doc.get("schema_version"),
                          "summary_policy": prov.get("prompt_hash"),
                          "source_provenance": prov, "violations": doc.get("violations", [])})
    if not old and len(models) > 1:
        raise ValueError(f"multiple Summary models in one Arm: {arm.name}")
    return documents


def representation_signature(context, catalog, arm, documents, *, selection_hash=None, legacy=False):
    if legacy:
        return _legacy_representation_signature(context, catalog, arm, documents, selection_hash=selection_hash)
    from validation.cache_identity import REPRESENTATION_VERSION, canonical, semantic_documents
    return fingerprint({"version": "full-catalog-representation/v2" if legacy_layout(context.config) else REPRESENTATION_VERSION, "arm": arm.name,
                        **({"arm_contract": ARM_CONTRACT, "uses_title": arm.uses_title, "scene_arm": arm.scene_arm}
                           if not legacy_layout(context.config) else {}),
                        "catalog": [{k: r[k] for k in ("item_id", "content_id")} for r in catalog], "encoder": context.config["validation"]["encoder"],
                        "model": canonical(local_model_identity(context.path("models", "bge"))),
                        "documents": semantic_documents(documents)})
