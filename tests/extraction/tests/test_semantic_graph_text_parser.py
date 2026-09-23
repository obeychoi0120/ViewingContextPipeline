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
        {"id": "person-1", "name": "person",
         "attributes": ["long-haired", "black, white and red shirt"]},
        {"id": "cup1", "name": "cup", "attributes": ["blue-green"]},
    ],
    "relations": [{"subject_id": "person-1", "predicate": "standing-next-to", "object_id": "cup1"}],
    "context": ["A blue-green, softly lit room."],
}


@pytest.mark.parametrize("text", [TEXT, TEXT.replace(" -> ", "->"), TEXT.replace("\n", "\r\n")])
def test_line_graph_preserves_fields_and_hyphenated_values(text):
    result = parse_or_repair_graph(text)
    assert result.graph == GRAPH and result.parse_mode == "native"
    validate_graph_structure(result.graph)


@pytest.mark.parametrize("delimiters", [
    (" - ", " - "), (" -> ", " - "), (" - ", " -> "),
    (" → ", " ⇒ "), ("-->", "=>"), ("－＞", "－＞"),
    (" — ", " – "), ("--->", "--->"),
])
def test_repairs_only_relation_delimiters(delimiters):
    left, right = delimiters
    text = TEXT.replace("person-1 -> standing-next-to -> cup1",
                        f"person-1{left}standing-next-to{right}cup1")
    result = parse_or_repair_graph(text)
    assert result.graph == GRAPH and result.parse_mode == "repaired"


@pytest.mark.parametrize("text", [
    "```text\n" + TEXT + "\n```",
    "\ufeff" + TEXT,
    TEXT.replace("[Entities]", "[entities]").replace("[Relations]", "Relations:")
        .replace("[Context]", "## Context").replace("[End]", "[end]"),
    TEXT.replace("[Entities]", "- [Entities]").replace("[Relations]", "2. Relations:"),
    TEXT.replace("person-1:", "- person-1:").replace("cup1:", "2. cup1:")
        .replace("person-1 ->", "• person-1 ->").replace("A blue-green", "* A blue-green"),
    TEXT.replace(":", "：").replace(";", "；"),
    TEXT.replace("cup1: cup; blue-green", "cup1: cup; blue-green;"),
])
def test_surface_repairs_preserve_every_field(text):
    result = parse_or_repair_graph(text)
    assert result.graph == GRAPH and result.parse_mode == "repaired"


@pytest.mark.parametrize("text", [
    TEXT.replace("[Relations]", "[Entities]"),
    TEXT.replace("[Context]", "[context]\n[Context]"),
    TEXT + "\n[Entities]\nnone",
    TEXT + "\nextra explanation",
    "Here is the graph:\n" + TEXT,
    TEXT.replace("[Entities]", "[Entities]\nnone"),
    TEXT.replace("[Context]", "[Context]\nNone"),
    TEXT.replace("[Context]", "[Context]\n[Other]"),
    TEXT.replace("[Context]", "[Context]\n- [Entities]"),
    TEXT.replace("[Context]", "[Context]\n- [Other]"),
    TEXT.replace("person-1: person", "person-1:"),
    TEXT.replace("person-1: person; long-haired", "person-1: person;; long-haired"),
    TEXT.replace(" -> standing-next-to -> ", " - "),
    TEXT.replace(" -> standing-next-to -> ", " -> -> "),
    TEXT.replace(" -> standing-next-to -> ", " <- standing-next-to -> "),
    TEXT.replace(" -> standing-next-to -> ", " -> standing-next-to ← "),
    TEXT.replace(" -> standing-next-to -> ", " - standing - next to - "),
    TEXT.replace(" -> standing-next-to -> ", "-standing-next-to-"),
    TEXT.replace("cup1: cup; blue-green", "cup1: cup;\nblue-green"),
    TEXT.replace("A blue-green, softly lit room.", ""),
    "```\n" + TEXT,
    "```\n" + TEXT + "\n```\nextra explanation",
])
def test_does_not_invent_discard_or_select_ambiguous_content(text):
    result = parse_or_repair_graph(text)
    assert result.graph is None and result.error


def test_no_semantic_id_validation_or_count_truncation():
    entities = "\n".join(["p1: person; smiling"] * 6)
    relations = "\n".join(["missing -> looking at -> unregistered"] * 6)
    text = f"[Entities]\n{entities}\n[Relations]\n{relations}\n[Context]\nnone\n[End]"
    graph = parse_or_repair_graph(text).graph
    validate_graph_structure(graph)
    assert len(graph["entities"]) == len(graph["relations"]) == 6
    assert {e["id"] for e in graph["entities"]} == {"p1"}
    assert graph["relations"][0]["subject_id"] == "missing"


@pytest.mark.parametrize("fenced", [False, True])
def test_complete_json_remains_supported(fenced):
    text = json.dumps(GRAPH)
    if fenced:
        text = "```json\n" + text + "\n```"
    result = parse_or_repair_graph(text)
    assert result.graph == GRAPH
    assert result.parse_mode == ("repaired" if fenced else "native")


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
    example = example[example.index("[Entities]"):]
    assert "context" not in prompt.lower()
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, example)
    assert failure is None
    assert "context" not in record["graph"]
    assert record["semantic_warnings"] == []
    records = minimal_graph_records([record], Path("video.jsonl"))
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert observations[0]["observation"] == record["graph"]


@pytest.mark.parametrize("text", [
    "[Entities]\nnone\n[End]",
    "[Entities]\nnone\n[Relations]\nnone\n[Relations]\nnone\n[End]",
    "[Entities]\nnone\n[Relations]\nnone\n[End]\nextra",
])
def test_contextless_graph_still_requires_complete_ordered_sections(text):
    assert parse_or_repair_graph(text).graph is None


def test_empty_contextless_graph_and_invalid_legacy_context():
    from extraction.structured_output import GRAPH_JSON_SCHEMA, OutputValidationError

    result = parse_or_repair_graph("[Entities]\nnone\n[Relations]\nnone\n[End]")
    assert result.graph == {"entities": [], "relations": []}
    validate_graph_structure(result.graph)
    assert "context" not in GRAPH_JSON_SCHEMA["properties"]
    with pytest.raises(OutputValidationError):
        validate_graph_structure({**result.graph, "context": "invalid"})


@pytest.mark.parametrize('legacy_context', [False, True])
def test_missing_end_is_accepted_without_inventing_context(legacy_context):
    text = '[Entities]\nperson1: person; blue hair\n[Relations]\nnone'
    if legacy_context:
        text += '\n[Context]\nA room.'
    result = parse_or_repair_graph(text)
    assert result.error is None
    assert result.graph['entities'][0]['id'] == 'person1'
    assert result.graph['relations'] == []
    assert ('context' in result.graph) == legacy_context
    validate_graph_structure(result.graph)


@pytest.mark.parametrize('text, missing', [
    ('', None),
    ('[Entities]\nnone', 'Relations'),
    ('[Entities]\nnone\n[Context]\nnone\n[End]', None),
])
def test_optional_context_does_not_replace_required_sections(text, missing):
    result = parse_or_repair_graph(text)
    assert result.graph is None
    if missing:
        assert f'missing [{missing}]' in result.error
    else:
        assert result.error
