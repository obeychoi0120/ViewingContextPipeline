from __future__ import annotations

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


def graph_with_dangling_reference() -> dict:
    return {
        "setting_context": "indoor",
        "entities": [
            {"local_id": "e1", "name": "person", "role": "primary"}
        ],
        "events": [],
        "static_relations": [
            {"subject_id": "e1", "relation": "WEARING", "object_id": "e4"}
        ],
        "semantic_topics": [],
        "affect": {
            "subject_ids": ["e1"],
            "valence": "neutral",
            "arousal": "medium",
        },
    }


def test_parse_fenced_graph_without_semantic_normalization() -> None:
    value = graph_with_dangling_reference()
    value["setting_context"] = " Indoor "
    value["static_relations"][0]["relation"] = "wearing"

    parsed = parse_graph_output(f"```json\n{json.dumps(value)}\n```")

    assert parsed == value


def test_json_repair_accepts_trailing_comma_but_rejects_empty_output() -> None:
    text = json.dumps(graph_with_dangling_reference())
    repaired = text[:-1] + ",}"
    assert parse_graph_output(repaired)["entities"][0]["local_id"] == "e1"
    with pytest.raises(SemanticGraphError, match="empty VLM output"):
        parse_graph_output("")


def test_semantic_schema_and_reference_errors_are_not_rejected() -> None:
    dangling = graph_with_dangling_reference()
    assert parse_graph_output(json.dumps(dangling)) == dangling

    arbitrary = {"triples": [], "extra": "preserved"}
    assert parse_graph_output(json.dumps(arbitrary)) == arbitrary


def test_taxonomy_and_prompt_are_multi_image_and_consistent() -> None:
    taxonomy = taxonomy_contract()
    assert taxonomy["schema_version"] == "minimal-semantic-scene/v1"
    assert taxonomy["limits"]["entities"] == 6
    assert "chronological keyframes" in SCENE_EXTRACTION_PROMPT
    assert "If entities is empty" in SCENE_EXTRACTION_PROMPT
    for value in taxonomy["setting_contexts"]:
        assert value in SCENE_EXTRACTION_PROMPT


def test_graph_summary_preserves_raw_graph_and_sorts_scenes() -> None:
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
                "graph": graph_with_dangling_reference(),
            }
        )

    prompt = graph_summary_prompt("Graphs:\n{scenes}", records)

    assert prompt.index("Scene 0") < prompt.index("Scene 1")
    assert '"object_id": "e4"' in prompt


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
            return json.dumps(graph_with_dangling_reference())

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

    assert records[0]["graph"]["static_relations"][0]["object_id"] == "e4"
    assert records[0]["keyframes"] == [5, 15, 25]
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 3
