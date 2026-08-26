from __future__ import annotations

import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from extraction.semantic_graph import (
    SCENE_EXTRACTION_PROMPT,
    SemanticGraphError,
    extract_scene_graphs,
    graph_summary_prompt,
    parse_graph_output,
    taxonomy_contract,
    validate_summary,
)


def valid_graph() -> dict:
    return {
        "setting_context": "indoor",
        "entities": [
            {"local_id": "e1", "name": "person", "role": "primary"},
            {"local_id": "e2", "name": "vegetable", "role": "secondary"},
            {"local_id": "e3", "name": "knife", "role": "secondary"},
            {"local_id": "e4", "name": "apron", "role": "context"},
        ],
        "events": [
            {
                "local_id": "ev1",
                "actor_id": "e1",
                "action": "slice",
                "target_id": "e2",
                "instrument_id": "e3",
                "location_id": None,
            }
        ],
        "static_relations": [
            {"subject_id": "e1", "relation": "WEARING", "object_id": "e4"}
        ],
        "semantic_topics": [
            {
                "label": "vegetable preparation",
                "evidence_entity_ids": ["e1", "e2"],
                "evidence_event_ids": ["ev1"],
            }
        ],
        "affect": {
            "subject_ids": ["e1"],
            "valence": "neutral",
            "arousal": "medium",
        },
    }


def test_parse_fenced_graph_and_normalize_harmless_formatting() -> None:
    value = valid_graph()
    value["setting_context"] = " Indoor "
    value["entities"][0]["name"] = " Person "
    value["static_relations"][0]["relation"] = "wearing"

    parsed = parse_graph_output(f"```json\n{json.dumps(value)}\n```")

    assert parsed["setting_context"] == "indoor"
    assert parsed["entities"][0]["name"] == "person"
    assert parsed["static_relations"][0]["relation"] == "WEARING"


def test_json_repair_accepts_trailing_comma_but_rejects_empty_output() -> None:
    text = json.dumps(valid_graph())
    repaired = text[:-1] + ",}"
    assert parse_graph_output(repaired)["entities"][0]["local_id"] == "e1"
    with pytest.raises(SemanticGraphError, match="does not contain"):
        parse_graph_output("")


def test_empty_scene_is_valid_only_with_empty_unknown_dependents() -> None:
    value = {
        "setting_context": "unknown",
        "entities": [],
        "events": [],
        "static_relations": [],
        "semantic_topics": [],
        "affect": {"subject_ids": [], "valence": "unknown", "arousal": "unknown"},
    }
    assert parse_graph_output(json.dumps(value)) == value

    invalid = copy.deepcopy(value)
    invalid["affect"]["valence"] = "neutral"
    with pytest.raises(SemanticGraphError, match="unknown/unknown|without subjects"):
        parse_graph_output(json.dumps(invalid))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "exactly"),
        (lambda value: value.update(setting_context="studio"), "setting_context"),
        (lambda value: value["entities"][1].update(local_id="e3"), "must be 'e2'"),
        (lambda value: value["events"][0].update(local_id="ev2"), "must be 'ev1'"),
        (lambda value: value["events"][0].update(target_id="e9"), "dangling"),
        (lambda value: value["events"][0].update(actor_id=None, target_id=None, instrument_id=None), "at least one"),
        (lambda value: value["static_relations"][0].update(relation="AT"), "static relation"),
        (lambda value: value["semantic_topics"][0].update(label="one two three four five"), "exceeds 4"),
        (lambda value: value["semantic_topics"][0].update(evidence_entity_ids=[], evidence_event_ids=[]), "cite evidence"),
        (lambda value: value["affect"].update(subject_ids=["e9"]), "dangling"),
        (lambda value: value["entities"][0].update(name="object"), "generic"),
    ],
)
def test_strict_schema_rejects_invalid_values(mutate, message: str) -> None:
    value = valid_graph()
    mutate(value)
    with pytest.raises(SemanticGraphError, match=message):
        parse_graph_output(json.dumps(value))


def test_limits_are_rejected_instead_of_truncated() -> None:
    value = valid_graph()
    value["entities"] = [
        {"local_id": f"e{index}", "name": f"entity {index}", "role": "context"}
        for index in range(1, 8)
    ]
    with pytest.raises(SemanticGraphError, match="exceeds maximum 6"):
        parse_graph_output(json.dumps(value))


def test_old_flat_triples_payload_is_rejected() -> None:
    with pytest.raises(SemanticGraphError, match="exactly"):
        parse_graph_output(json.dumps({"triples": []}))


def test_taxonomy_and_prompt_are_multi_image_and_consistent() -> None:
    taxonomy = taxonomy_contract()
    assert taxonomy["schema_version"] == "minimal-semantic-scene/v1"
    assert taxonomy["limits"]["entities"] == 6
    assert "chronological keyframes" in SCENE_EXTRACTION_PROMPT
    assert "If entities is empty" in SCENE_EXTRACTION_PROMPT
    assert "story, intent, identity" in SCENE_EXTRACTION_PROMPT
    for value in taxonomy["setting_contexts"]:
        assert value in SCENE_EXTRACTION_PROMPT
    assert "relation_family" not in SCENE_EXTRACTION_PROMPT
    assert "INTERACTS_WITH" not in SCENE_EXTRACTION_PROMPT


def test_graph_summary_resolves_ids_and_sorts_scenes() -> None:
    records = []
    for scene_idx, start in ((1, 30), (0, 0)):
        records.append(
            {
                "schema_version": "minimal-semantic-scene/v1",
                "scene_idx": scene_idx,
                "scene_start_seconds": start,
                "scene_end_seconds": start + 30,
                "keyframes": [start + 5, start + 15, start + 25],
                "image_paths": ["a.png", "b.png", "c.png"],
                "graph": valid_graph(),
            }
        )

    prompt = graph_summary_prompt("Graphs:\n{scenes}", records)

    assert prompt.index("Scene 0") < prompt.index("Scene 1")
    assert "actor=person" in prompt
    assert "target=vegetable" in prompt
    assert "person wears apron" in prompt


def test_graph_summary_rejects_incomplete_scene_evidence() -> None:
    record = {
        "schema_version": "minimal-semantic-scene/v1",
        "scene_idx": 0,
        "scene_start_seconds": 0,
        "scene_end_seconds": 30,
        "keyframes": [5, 15, 25],
        "image_paths": ["only-one.png"],
        "graph": valid_graph(),
    }
    with pytest.raises(SemanticGraphError, match="incomplete scene evidence"):
        graph_summary_prompt("{scenes}", [record])


def test_summary_requires_one_to_150_words() -> None:
    assert validate_summary("one") == "one"
    assert len(validate_summary("word " * 150).split()) == 150
    with pytest.raises(SemanticGraphError, match="1-150"):
        validate_summary("")
    with pytest.raises(SemanticGraphError, match="1-150"):
        validate_summary("word " * 151)


def test_scene_graph_uses_one_chronological_multi_image_call(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for timestamp in (5, 15, 25):
        Image.new("RGB", (8, 6), "white").save(frames / f"{timestamp:04d}.png")
    timestamps = tmp_path / "timestamps.json"
    timestamps.write_text("[]", encoding="utf-8")

    class FakeBackend:
        model_id = "fake"

        def __init__(self) -> None:
            self.calls = []

        def generate(self, images, prompt, max_new_tokens, references=()):
            self.calls.append((images, prompt, max_new_tokens, references))
            return json.dumps(valid_graph())

    backend = FakeBackend()
    records = extract_scene_graphs(
        content_id="demo",
        scenes=[
            {
                "scene_idx": 0,
                "scene_start": 0,
                "scene_end": 30,
                "keyframes": [25, 5, 15],
            }
        ],
        frames_dir=frames,
        timestamp_json_path=timestamps,
        backend=backend,
        prompt="extract",
        max_new_tokens=128,
    )

    assert records[0]["keyframes"] == [5, 15, 25]
    assert records[0]["scene_start_seconds"] == 0
    assert records[0]["scene_end_seconds"] == 30
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 3
    assert backend.calls[0][1:] == ("extract", 128, ())
