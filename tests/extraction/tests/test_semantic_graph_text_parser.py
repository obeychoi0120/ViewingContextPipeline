import json
from pathlib import Path

import pytest

from extraction.semantic_graph import parse_or_repair_graph
from extraction.structured_output import validate_graph_structure


TEXT = """[Entities]
person-1: person; long-haired; black, white and red shirt
cup1: cup; blue-green
[Relations]
person-1 -> standing-next-to -> cup1
[Context]
A blue-green, softly lit room.
[End]"""
GRAPH = {
    "entities": [
        {
            "id": "person-1",
            "name": "person",
            "attributes": ["long-haired", "black, white and red shirt"],
        },
        {"id": "cup1", "name": "cup", "attributes": ["blue-green"]},
    ],
    "relations": [{"subject_id": "person-1", "predicate": "standing-next-to", "object_id": "cup1"}],
    "context": ["A blue-green, softly lit room."],
}


def test_line_graph_preserves_fields_and_hyphenated_values():
    text = TEXT
    result = parse_or_repair_graph(text)
    assert result.graph == GRAPH and result.parse_mode == "native"
    validate_graph_structure(result.graph)


def test_repairs_only_relation_delimiters():
    delimiters = (" → ", " ⇒ ")
    left, right = delimiters
    text = TEXT.replace(
        "person-1 -> standing-next-to -> cup1", f"person-1{left}standing-next-to{right}cup1"
    )
    result = parse_or_repair_graph(text)
    assert result.graph == GRAPH and result.parse_mode == "repaired"


@pytest.mark.parametrize(
    "text",
    [
        TEXT.replace("[Relations]", "[Entities]"),
        TEXT.replace("person-1: person", "person-1:"),
        TEXT.replace(" -> standing-next-to -> ", " - standing - next to - "),
    ],
)
def test_does_not_invent_discard_or_select_ambiguous_content(text):
    result = parse_or_repair_graph(text)
    assert result.graph is None and result.error


@pytest.mark.parametrize("form", ["native", "repaired", "json"])
def test_graph_without_context(form):
    graph = {key: value for key, value in GRAPH.items() if key != "context"}
    text = TEXT.replace("[Context]\nA blue-green, softly lit room.\n", "")
    if form == "repaired":
        text = text.replace("[Relations]", "Relations:").replace(" -> ", " → ")
    elif form == "json":
        text = json.dumps(graph)
    result = parse_or_repair_graph(text)
    assert result.graph == graph
    assert result.parse_mode == ("repaired" if form == "repaired" else "native")
    validate_graph_structure(result.graph)


def test_v4_example_passes_scene_validation_and_summary():
    from extraction.scene_executor import graph_scene_result
    from extraction.semantic_graph import graph_summary_prompt
    from extraction.step_support import minimal_graph_records

    root = Path(__file__).resolve().parents[3]
    prompt = (root / "prompts/scene_graph_v4.md").read_text()
    example = prompt.split("[Example]", 1)[1]
    example = example[example.index("[Entities]") :]
    assert "context" not in prompt.lower()
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, example)
    assert failure is None
    assert "context" not in record["graph"]
    assert record["semantic_warnings"] == []
    records = minimal_graph_records([record], Path("video.jsonl"))
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert observations[0]["observation"] == record["graph"]


@pytest.mark.parametrize(
    "text, missing",
    [
        ("", None),
        ("[Entities]\nnone", "Relations"),
        ("[Entities]\nnone\n[Context]\nnone\n[End]", None),
    ],
)
def test_optional_context_does_not_replace_required_sections(text, missing):
    result = parse_or_repair_graph(text)
    assert result.graph is None
    if missing:
        assert f"missing [{missing}]" in result.error
    else:
        assert result.error
