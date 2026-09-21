"""Semantic identities exclude locations and run bookkeeping, never model settings."""
from extraction.recovery import fingerprint

REPRESENTATION_VERSION = "shared-scenes-representation/v3"


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()
                if k not in {"run_id", "source_run_id", "prompt_path", "source_path", "scene_path",
                             "path", "created_at", "generated_at", "reused_from"}}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def generation_identity(provenance):
    keys = ("arm", "representation", "prompt_hash", "model", "settings", "summary_model",
            "schema_contract", "uses_title", "scene_arm", "scene_provenance", "english_title", "scene_input_hash")
    return canonical({k: provenance[k] for k in keys if k in provenance})


def verified_generation(prov):
    return (isinstance(prov, dict) and bool(prov.get("prompt_hash"))
            and bool(prov.get("model")) and bool(prov.get("settings")))


def shareable_document(doc):
    prov = doc.get("source_provenance", {})
    scenes = prov.get("scene_provenance")
    return (verified_generation(prov) and isinstance(scenes, list) and bool(scenes)
            and all(verified_generation(p) for p in scenes))


def semantic_documents(docs):
    return [{"content_id": d["content_id"], "text": d["text"],
             "generation": generation_identity(d.get("source_provenance", {}))} for d in docs]


def semantic_document_hash(doc):
    return fingerprint(semantic_documents([doc])[0])
