"""Role defaults, explicit uncertainty, and old/new graph round trips."""
import copy
import json
import re
from pathlib import Path

import pytest

from extraction.scene_executor import graph_scene_result
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import graph_summary_prompt, parse_or_repair_graph
from extraction.step_support import minimal_graph_records
from extraction.structured_output import OutputValidationError, validate_graph_structure

PREFIX = """[Context]
medium: live_action
format: demonstration
topics: food preparation
[Entities]
p1: person; wearing gloves
c1: carrot
k1: knife
p2: person; wearing apron
l1: kitchen
[Actions]
"""


def parse(action):
    result = parse_or_repair_graph(PREFIX + action)
    assert result.graph is not None, result.error
    validate_graph_structure(result.graph)
    return result


@pytest.mark.parametrize("suffix,location,mode", [
    ("", "unknown", "repaired"),
    ("; p2", "p2", "native"),  # Five-slot v8: last slot always means location.
    ("; ; l1", "l1", "repaired"),
    ("; ;", "unknown", "repaired"),
    ("; p2;", "unknown", "repaired"),
    ("; none; unknown", "unknown", "repaired"),
    ("; unknown; none", None, "repaired"),
    ("; p2; l1", "l1", "repaired"),
    ("; l1", "l1", "native"),
])
def test_positional_defaults_and_explicit_values(suffix, location, mode):
    result = parse("p1 - cutting - c1; k1" + suffix)
    action = result.graph["actions"][0]
    assert "receiver" not in action
    assert action["location"] == location
    assert result.parse_mode == mode
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, PREFIX + "p1 - cutting - c1; k1" + suffix)
    assert failure is None and record["semantic_warnings"] == []


@pytest.mark.parametrize("missing", [(), ("receiver",), ("location",), ("receiver", "location")])
@pytest.mark.parametrize("explicit", [None, "unknown", "p2"])
def test_json_missing_keys_only(missing, explicit):
    graph = parse("p1 - handing - c1; none; p2; l1").graph
    action = graph["actions"][0]
    action.update(receiver=explicit, location=explicit)
    for key in missing:
        action.pop(key)
    before = copy.deepcopy(graph)
    validate_graph_structure(graph)
    assert graph == before  # Validation is non-mutating, including old data.
    result = parse_or_repair_graph(json.dumps(graph))
    validate_graph_structure(result.graph)
    assert "receiver" not in result.graph["actions"][0]
    assert result.graph["actions"][0]["location"] == ("unknown" if "location" in missing else explicit)
    assert result.parse_mode == ("repaired" if "receiver" not in missing or "location" in missing else "native")


@pytest.mark.parametrize("field", ["actor", "target", "tool", "receiver", "location"])
def test_unknown_allowed_without_entity(field):
    graph = parse("p1 - cutting - c1; k1; none; unknown").graph
    graph["actions"][0][field] = "unknown"
    validate_graph_structure(graph)


@pytest.mark.parametrize("actor,target", [(None, None), ("unknown", None),
                                          (None, "unknown"), ("unknown", "unknown")])
def test_unknown_is_not_a_known_endpoint(actor, target):
    graph = parse("p1 - cutting - c1; k1; none; unknown").graph
    graph["actions"][0].update(actor=actor, target=target)
    with pytest.raises(OutputValidationError):
        validate_graph_structure(graph)


@pytest.mark.parametrize("field", ["location"])
@pytest.mark.parametrize("value", ["", "absent", "none", 5, [], {}])
def test_invalid_optional_values_preserve_raw(field, value):
    graph = parse("p1 - cutting - c1; k1; none; unknown").graph
    graph["actions"][0][field] = value
    text = json.dumps(graph)
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert failure is None and record["graph"] == text
    assert record["semantic_warnings"]


@pytest.mark.parametrize("identifier", ["none", "unknown", "UNKNOWN"])
def test_reserved_entity_ids(identifier):
    graph = parse("p1 - cutting - c1; k1; none; unknown").graph
    graph["entities"].append({"id": identifier, "name": "person", "attributes": []})
    with pytest.raises(OutputValidationError):
        validate_graph_structure(graph)


@pytest.mark.parametrize("action", [
    "p1 - cutting - c1", "p1 - cutting - c1;", " - cutting - c1; k1",
    "p1 - cutting - ; k1", "p1 -  - c1; k1",
    "p1 - cutting - c1; k1; none; unknown; p2",
])
def test_required_slots_and_extra_slots_are_not_repaired(action):
    text = PREFIX + action
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert failure is None and record["graph"] == text
    assert record["semantic_warnings"]


def test_v7_examples_and_storage_summary_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[2]
    prompt = (root / "prompts/scene_graph_v7.md").read_text()
    examples = re.findall(r"\[Context\]\n.*?(?=\n\s*\n|\Z)", prompt.split("[Examples]", 1)[1], re.S)
    assert len(examples) == 3
    graphs = []
    for example in examples:
        result = parse_or_repair_graph(example)
        validate_graph_structure(result.graph)
        graphs.append(result.graph)
    assert "receiver" not in graphs[1]["actions"][0]
    assert graphs[1]["actions"][0]["location"] == "unknown"
    assert graphs[2]["actions"] == []
    # Keep a structured pre-v7 graph and a legacy relational graph unchanged on disk/read.
    old = copy.deepcopy(graphs[0])
    for field in ("receiver", "location"):
        old["actions"][0].pop(field, None)
    legacy = {"entities": [], "relations": []}
    graphs.extend([old, legacy])
    path = tmp_path / "video.jsonl"
    write_scene_records(path, [{"scene_idx": i, "graph": graph} for i, graph in enumerate(graphs)])
    records = minimal_graph_records(read_scene_records(path), path)
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert [row["observation"] for row in observations] == graphs
    summary = graph_summary_prompt((root / "prompts/summary_graph_v7.md").read_text(), records)
    assert "{scenes}" not in summary
    assert '"receiver":' not in summary and '"location": "unknown"' in summary


def test_mixed_optional_slots_and_json_repairs():
    text = PREFIX + "p1 - cutting - c1; k1\np1 - handing - c1; none; p2; l1"
    result = parse_or_repair_graph(text)
    validate_graph_structure(result.graph)
    assert result.parse_mode == "repaired"
    assert all("receiver" not in a for a in result.graph["actions"])
    graph = result.graph
    graph["actions"][0].pop("location")
    repaired = parse_or_repair_graph("```json\n" + json.dumps(graph)[:-1] + ",}\n```")
    assert repaired.graph["actions"][0]["location"] == "unknown"
    assert repaired.parse_mode == "repaired"
