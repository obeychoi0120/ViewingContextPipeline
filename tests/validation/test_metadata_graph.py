import numpy as np
import pytest

from pipeline_runtime import read_json, write_json, write_jsonl
from test_rolling import full_context as full_context
from test_four_arm_diagnosis import _write_success_scene_files
from validation.metadata_graph import ARM, BRANCH, TARGET, combined_text
from validation.recommendation_contracts import RECOMMENDATION_ARMS, resolve_target_arms


def test_fusion_is_explicit_and_preserves_both_texts():
    assert resolve_target_arms() == RECOMMENDATION_ARMS
    assert len(resolve_target_arms()) == 4
    assert resolve_target_arms([TARGET]) == {ARM: BRANCH}
    assert combined_text("A title", "A graph summary") == "Title: A title\n\nVisual context:\nA graph summary"
    assert combined_text("", "Summary").endswith("Summary")
    with pytest.raises(ValueError):
        combined_text("Title", "")


def test_public_fusion_embedding_cache_and_input_invalidation(full_context, monkeypatch):
    from extraction.input_tracking import input_state_path
    from extraction.summary_validation import SUMMARY_SECTIONS, serialize_summary_sections
    from validation.steps import embed_representations, ValidationStepError

    context = full_context
    catalog = context.require_ready_cohort()["catalog"]
    _write_success_scene_files(context.run_root, [row["content_id"] for row in catalog])
    for row in catalog:
        sections = {name: "graph evidence" for name in SUMMARY_SECTIONS}
        write_json(context.graph_summary_dir("qwen") / f"{row['content_id']}.json", {
            "schema_version": "graph-video-summary/v3", "arm": "graph_qwen",
            "content_id": row["content_id"], "status": "complete", "scene_count": 1,
            "sections": sections, "text": serialize_summary_sections(sections),
        })
    encoded = []

    class Encoder:
        def __init__(self, _config):
            pass

        def encode(self, texts):
            encoded.append(list(texts))
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr("validation.features.BGETextEncoder", Encoder)
    embed_representations(context, target=[TARGET])
    assert len(encoded) == 1
    assert encoded[0][0].startswith("Title: title 1\n\nVisual context:\n")
    assert "graph evidence" in encoded[0][0]
    assert [path.name for path in context.representations_dir.glob("*.npz")] == [
        f"{BRANCH}_embeddings.npz",
    ]
    embed_representations(context, target=[TARGET])
    assert len(encoded) == 1  # No other arm or summary is required, including on resume.
    titles = [dict(row) for row in context.require_ready_cohort()["metadata_titles"]]
    titles[0]["title"] = "New title"
    write_jsonl(context.cohort_dir / "metadata_titles.jsonl", titles)
    embed_representations(context, target=[TARGET])
    assert len(encoded) == 2 and encoded[-1][0].startswith("Title: New title\n")
    summary_path = context.graph_summary_dir("qwen") / f"{catalog[0]['content_id']}.json"
    dirty = input_state_path(summary_path).with_suffix(".dirty")
    dirty.parent.mkdir(parents=True, exist_ok=True)
    dirty.write_text("changed scenes")
    with pytest.raises(ValidationStepError, match="regenerate summary first"):
        embed_representations(context, target=[TARGET])
    assert len(encoded) == 2


@pytest.mark.parametrize("step", ["embed_representations", "run_recommendation", "run_diagnosis"])
def test_fusion_rejects_legacy_protocol(full_context, step):
    from validation import steps

    full_context.config["schema_version"] = "viewing-context-config/v3"
    with pytest.raises(ValueError, match="requires the v4 rolling protocol"):
        getattr(steps, step)(full_context, target=[TARGET])


@pytest.mark.torch
def test_fusion_training_diagnosis_and_both_input_dependencies(full_context, monkeypatch):
    import torch
    from extraction.summary_validation import SUMMARY_SECTIONS, serialize_summary_sections
    from validation.steps import (validation_config, _embedding_work, _embedding_documents,
                                  _persist_representations, run_recommendation, run_diagnosis)
    from validation.representation_provenance import input_hash
    from validation.representation_checks import verify_representations
    from validation.model import SASRec

    context = full_context
    cohort = context.require_ready_cohort()
    catalog = cohort["catalog"]
    _write_success_scene_files(context.run_root, [r["content_id"] for r in catalog])
    for row in catalog:
        sections = {name: "visible evidence" for name in SUMMARY_SECTIONS}
        write_json(context.graph_summary_dir("qwen") / f"{row['content_id']}.json", {
            "schema_version": "graph-video-summary/v3", "arm": "graph_qwen",
            "content_id": row["content_id"], "status": "complete", "scene_count": 1,
            "sections": sections, "text": serialize_summary_sections(sections),
        })
    config = validation_config(context)
    sources, fallbacks, pending = _embedding_work(context, catalog, config, False, branches={BRANCH})
    assert pending == [BRANCH] and not fallbacks
    docs = _embedding_documents(context, catalog, sources, [BRANCH], set())
    assert "title 1" in docs[BRANCH][0]["text"] and "visible evidence" in docs[BRANCH][0]["text"]
    signature = input_hash(docs[BRANCH], catalog, {
        **config.encoder.model_dump(mode="json"), "model": str(context.path("models", "bge")),
    })
    values = np.random.default_rng(4).normal(size=(4, 1024)).astype(np.float32)
    _persist_representations(context, {BRANCH: values}, catalog, [], {BRANCH: signature})
    verify_representations(context, cohort, arms={ARM: BRANCH})

    def tiny_model(config, *, item_count, branch, features, device):
        assert branch == BRANCH
        return SASRec(item_count, 10, 8, 1, 2, 0, arm="graph", item_features=features).to(device)

    monkeypatch.setattr("validation.rolling_recommendation._new_model", tiny_model)
    monkeypatch.setattr("validation.rolling_recommendation.worker_devices", lambda *_: ["cpu"])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_recommendation(context, target=[TARGET])["completed"] == 21
        assert run_recommendation(context, target=[TARGET])["skipped"] == 21
        assert run_diagnosis(context, target=[TARGET])["status"] == "pass"
    finally:
        torch.set_num_threads(threads)
    result = read_json(context.diagnosis_path)
    assert result["selected_arms"] == [ARM]
    assert set(result["scene_coverage"]["arms"]) == {"graph_qwen"}
    assert set(result["generation_recovery"]) == {"graph/qwen/scenes", "graph/qwen/summaries"}
    titles = [dict(row) for row in cohort["metadata_titles"]]
    titles[0]["title"] = "Changed title"
    write_jsonl(context.cohort_dir / "metadata_titles.jsonl", titles)
    with pytest.raises(RuntimeError, match="stale"):
        verify_representations(context, cohort, arms={ARM: BRANCH})
    write_jsonl(context.cohort_dir / "metadata_titles.jsonl", cohort["metadata_titles"])
    path = context.graph_summary_dir("qwen") / f"{catalog[0]['content_id']}.json"
    doc = read_json(path)
    doc["sections"] = {name: "different evidence" for name in SUMMARY_SECTIONS}
    doc["text"] = serialize_summary_sections(doc["sections"])
    write_json(path, doc)
    with pytest.raises(RuntimeError, match="stale"):
        verify_representations(context, cohort, arms={ARM: BRANCH})
