import json

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


@pytest.mark.parametrize("marker", ["none", "NONE", "None", "[]"])
def test_explicit_empty_sections(marker):
    text = f"[Entities]\n{marker}\n[Relations]\n{marker}\n[Context]\n{marker}\n[End]"
    assert parse_or_repair_graph(text).graph == {"entities": [], "relations": [], "context": []}


@pytest.mark.parametrize("text", [
    TEXT.replace("[End]", ""),
    TEXT.replace("[Context]\nA blue-green, softly lit room.\n", ""),
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
