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
    keys = ("representation", "prompt_hash", "model", "settings", "summary_model",
            "schema_contract", "uses_title", "scene_arm", "scene_provenance", "english_title", "scene_input_hash")
    return canonical(without_provenance_arm({k: provenance[k] for k in keys if k in provenance}))


def verified_generation(prov):
    return (isinstance(prov, dict) and bool(prov.get("prompt_hash"))
            and bool(prov.get("model")) and bool(prov.get("settings")))


def shareable_document(doc):
    from arm_registry import CONCAT_POLICY
    if doc.get("composition_policy") == CONCAT_POLICY and not doc.get("summary_used"):
        # Missing/failed visual inputs have a deterministic metadata-only or zero representation.
        return doc.get("summary_status") in {"missing", "failed", "not_applicable"}
    prov = doc.get("source_provenance", {})
    scenes = prov.get("scene_provenance")
    return (verified_generation(prov) and isinstance(scenes, list) and bool(scenes)
            and all(verified_generation(p) for p in scenes))


def semantic_documents(docs):
    return [{"content_id": d["content_id"], "text": d["text"],
             "generation": generation_identity(d.get("source_provenance", {})),
             **({"composition": {k: d.get(k) for k in ("composition_policy", "components", "summary_source", "summary_status")}}
                if "composition_policy" in d else {})} for d in docs]


def semantic_document_hash(doc):
    return fingerprint(semantic_documents([doc])[0])


def without_provenance_arm(value):
    """Remove the retired duplicate identity, including nested Scene provenance."""
    if isinstance(value, dict):
        return {key: without_provenance_arm(child) for key, child in value.items() if key != "arm"}
    if isinstance(value, list):
        return [without_provenance_arm(child) for child in value]
    return value
