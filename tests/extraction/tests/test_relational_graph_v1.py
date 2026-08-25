from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from extraction.relational_graph import (
    RelationalGraphError,
    extract_scene_graphs,
    graph_summary_prompt,
    ontology_from_document,
    parse_graph_output,
    validate_summary,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def ontology():
    value = json.loads(
        (ROOT / "contracts/extraction/relational_graph_ontology_v1.json").read_text(encoding="utf-8")
    )
    return ontology_from_document(value)


def test_parse_fenced_structured_triples(ontology) -> None:
    triples = parse_graph_output(
        """```json
        {"triples":[
          {"subject_id":"e1","subject":"pitcher","relation":"DOING","object_id":null,"object":"throwing"},
          {"subject_id":"e2","subject":"batter","relation":"DOING","object_id":null,"object":"hitting"},
          {"subject_id":"e1","subject":"pitcher","relation":"INTERACTS_WITH","object_id":"e2","object":"batter"}
        ]}
        ```""",
        ontology,
    )
    assert triples[2]["object_id"] == "e2"
    assert triples[2]["relation"] == "INTERACTS_WITH"


def test_dangling_interaction_is_rejected(ontology) -> None:
    text = json.dumps({"triples": [{
        "subject_id": "e1", "subject": "pitcher", "relation": "INTERACTS_WITH",
        "object_id": "e9", "object": "missing entity",
    }]})
    with pytest.raises(RelationalGraphError, match="dangling"):
        parse_graph_output(text, ontology)


def test_duplicate_entity_id_with_conflicting_label_is_rejected(ontology) -> None:
    text = json.dumps({"triples": [
        {"subject_id": "e1", "subject": "pitcher", "relation": "DOING", "object_id": None, "object": "throwing"},
        {"subject_id": "e1", "subject": "batter", "relation": "ROLE", "object_id": None, "object": "main subject"},
    ]})
    with pytest.raises(RelationalGraphError, match="conflicting labels"):
        parse_graph_output(text, ontology)


def test_graph_summary_preserves_scene_order() -> None:
    template = "Graphs:\n{scenes}"
    records = [
        {"schema_version": "scene-relational-graph/v1", "scene_idx": 0, "triples": [{"subject": "pitcher", "relation": "DOING", "object": "throwing"}]},
        {"schema_version": "scene-relational-graph/v1", "scene_idx": 1, "triples": [{"subject": "crowd", "relation": "MOOD", "object": "cheerful"}]},
    ]
    prompt = graph_summary_prompt(template, records)
    assert prompt.index("Scene 0") < prompt.index("Scene 1")


def test_summary_requires_150_to_300_words() -> None:
    assert len(validate_summary("word " * 150).split()) == 150
    with pytest.raises(RelationalGraphError, match="150-300"):
        validate_summary("too short")


def test_scene_graph_uses_shared_backend_contract(tmp_path: Path, ontology) -> None:
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
            return json.dumps({"triples": [{
                "subject_id": "e1",
                "subject": "pitcher",
                "relation": "DOING",
                "object_id": None,
                "object": "throwing",
            }]})

    backend = FakeBackend()
    records = extract_scene_graphs(
        content_id="demo",
        scenes=[{"scene_idx": 0, "keyframes": [5, 15, 25]}],
        frames_dir=frames,
        timestamp_json_path=timestamps,
        backend=backend,
        prompt="extract",
        ontology=ontology,
        max_new_tokens=128,
    )

    assert records[0]["triples"][0]["relation"] == "DOING"
    assert len(backend.calls[0][0]) == 3
    assert backend.calls[0][1:] == ("extract", 128, ())
