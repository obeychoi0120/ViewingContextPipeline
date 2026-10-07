"""Unsupported persisted artifacts fail before generation or cached input reuse."""
import copy
import json

import pytest

from arm_registry import registry
from artifact_io import read_json, write_json, write_jsonl
from extraction.errors import UnsupportedArtifactError, ExtractionStepError
from extraction.scene_storage import read_scene_records
from extraction import steps
from validation.representation_inputs import documents_for_arm


@pytest.mark.parametrize("legacy", ["Relations", "receiver", "missing_location", "graph_alias", "raw_response", "metadata"])
def test_unsupported_scene_is_not_rewritten_on_resume(ready_context, fake_models, generate_all, legacy):
    generate_all(ready_context)
    directory = ready_context.scene_arm_dir("graph_qwen")
    path = next(directory.glob("*.jsonl"))
    row = json.loads(path.read_text().splitlines()[0])
    if legacy == "Relations":
        row["scene_graph"] = {"entities": [], "relations": [], "context": []}
    elif legacy == "receiver":
        row["scene_graph"]["actions"][0]["receiver"] = None
    elif legacy == "missing_location":
        row["scene_graph"]["actions"][0].pop("location")
    elif legacy in {"graph_alias", "raw_response"}:
        row["graph" if legacy == "graph_alias" else "raw_response"] = row.pop("scene_graph")
    else:
        row["status"] = "complete"
    write_jsonl(path, [row])
    original, calls = path.read_bytes(), len(fake_models)
    with pytest.raises(UnsupportedArtifactError) as error:
        read_scene_records(path)
    assert str(path) in str(error.value)
    with pytest.raises(ExtractionStepError, match="unsupported"):
        steps.extract_graph_scenes(ready_context, model="qwen", arm="graph_qwen", schema="prompts/scene_graph_v8.md")
    assert path.read_bytes() == original and len(fake_models) == calls


@pytest.mark.parametrize("legacy", ["v4", "raw_fallback", "missing_model"])
def test_unsupported_summary_rejected_by_resume_and_embedding(ready_context, fake_models, generate_all, legacy):
    generate_all(ready_context)
    path = next(ready_context.summary_arm_dir("graph_qwen").glob("*.json"))
    doc = read_json(path)
    if legacy == "v4":
        doc["schema_version"] = "video-summary/v4"
    elif legacy == "raw_fallback":
        doc["status"] = "raw_fallback"
    else:
        doc["provenance"].pop("summary_model")
    write_json(path, doc)
    original, calls = path.read_bytes(), len(fake_models)
    with pytest.raises(UnsupportedArtifactError) as error:
        steps.summarize(ready_context, arm="graph_qwen", model="gemini", schema="prompts/summary_graph_v8.md")
    assert str(path) in str(error.value)
    with pytest.raises(UnsupportedArtifactError):
        documents_for_arm(ready_context, ready_context.require_ready_cohort(), registry(ready_context.config)["graph_qwen"])
    assert path.read_bytes() == original and len(fake_models) == calls


def test_unsupported_generated_json_keeps_raw_and_warning(tmp_path):
    from extraction.scene_executor import graph_scene_result
    from extraction.scene_storage import write_scene_records
    graph = {"medium": "live_action", "format": "demonstration", "topics": [],
             "entities": [], "actions": []}
    old = copy.deepcopy(graph)
    old["actions"] = [{"actor": "p", "action": "holding", "target": None,
                       "tool": None, "location": None, "receiver": None}]
    text = json.dumps(old)
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert failure is None and record["graph"] == text and record["semantic_warnings"]
    path = tmp_path / "video.jsonl"
    write_scene_records(path, [record])
    assert read_scene_records(path)[0]["graph"] == text
