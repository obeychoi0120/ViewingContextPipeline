"""New contract round trips, conservative repairs, and legacy read compatibility."""
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


TEXT = """[Context]
medium: live_action
format: demonstration
topics: seafood preparation; cooking
[Entities]
person-1: person; left-handed
food-1: sea urchin
knife-1: knife
[Actions]
person-1 - cross-cutting - food-1; knife-1
[End]"""


def parsed(text=TEXT):
    result = parse_or_repair_graph(text)
    assert result.error is None
    validate_graph_structure(result.graph)
    return result


@pytest.mark.parametrize("separator,mode", [(" - ", "native"), (" -> ", "repaired"),
                                            (" → ", "repaired"), (" ⇒ ", "repaired")])
def test_direction_and_internal_hyphens(separator, mode):
    result = parsed(TEXT.replace(" - ", separator))
    assert result.parse_mode == mode
    assert result.graph["topics"] == ["seafood preparation", "cooking"]
    assert result.graph["actions"] == [{"actor": "person-1", "action": "cross-cutting",
                                       "target": "food-1", "tool": "knife-1"}]
    assert "context" not in result.graph  # Context fields are top-level in storage.


@pytest.mark.parametrize("action", ["none - upgrading - food-1; none",
                                    "person-1 - walking - none; none", "none"])
def test_empty_references_and_actions(action):
    result = parsed(TEXT.replace("person-1 - cross-cutting - food-1; knife-1", action))
    assert "none" not in json.dumps(result.graph["actions"])


@pytest.mark.parametrize("end", ["", "\n[End]"])
def test_empty_sections(end):
    result = parsed("[Context]\nmedium: unknown\nformat: unknown\ntopics: none\n"
                    "[Entities]\nnone\n[Actions]\nnone" + end)
    assert result.graph == {"medium": "unknown", "format": "unknown", "topics": [],
                            "entities": [], "actions": []}


@pytest.mark.parametrize("old,new", [
    ("; knife-1", ""),
    ("cross-cutting", "cutting - slicing"),
    ("[Actions]", "[Relations]"),
    ("topics: seafood preparation; cooking", "topics: none; cooking"),
    ("[Entities]", "[Entities]\nnone"),
    ("[End]", "[End]\nignored extra content"),
])
def test_ambiguous_or_incomplete_lines_are_preserved_as_text(old, new):
    text = TEXT.replace(old, new)
    record, failure = graph_scene_result({"scene_idx": 2, "keyframes": []}, text)
    assert failure is None
    assert record["parse_mode"] == "text"
    assert record["graph"] == text


@pytest.mark.parametrize("mutation", [
    lambda g: g.update(topics="cooking"),
    lambda g: g["entities"].append(copy.deepcopy(g["entities"][0])),
    lambda g: g["entities"][0].update(id="none"),
    lambda g: g["actions"][0].update(actor="missing"),
    lambda g: g["actions"][0].update(target="missing"),
    lambda g: g["actions"][0].update(tool="missing"),
    lambda g: g["actions"][0].update(actor=None, target=None),
    lambda g: g["actions"][0].pop("tool"),
    lambda g: g["actions"][0].update(actor=1),
    lambda g: g["actions"][0].update(action="cutting - slicing"),
    lambda g: g.update(entities=[{"id": f"x{i}", "name": "cup", "attributes": []}
                                 for i in range(7)]),
])
def test_schema_and_reference_integrity(mutation):
    graph = parsed().graph
    mutation(graph)
    with pytest.raises(OutputValidationError):
        validate_graph_structure(graph)
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, json.dumps(graph))
    assert failure is None and record["parse_mode"] == "text"


def test_storage_and_summary_preserve_structured_raw_and_legacy(tmp_path):
    legacy = {"entities": [], "relations": [{"subject_id": "old", "predicate": "killed by",
                                             "object_id": "other"}], "context": ["gameplay"]}
    validate_graph_structure(legacy)  # No invented Actions or new restrictions on historical data.
    path = tmp_path / "video.jsonl"
    values = [parsed().graph, legacy, "unstructured observation"]
    write_scene_records(path, [{"scene_idx": i, "graph": value} for i, value in enumerate(values)])
    records = minimal_graph_records(read_scene_records(path), path)
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert [row["observation"] for row in observations] == values
    assert records[-1]["parse_mode"] == "text"


def test_prompt_examples_and_summary_template():
    root = Path(__file__).resolve().parents[3]
    prompt = (root / "prompts/scene_graph_v5.md").read_text()
    examples = re.findall(r"\[Context\]\n.*?(?=\n\s*\n|\Z)", prompt.split("[Examples]", 1)[1], re.S)
    assert len(examples) == 3
    graphs = [parsed(example).graph for example in examples]
    assert [len(graph["entities"]) for graph in graphs] == [3, 1, 1]
    summary = graph_summary_prompt((root / "prompts/summary_graph_v6.md").read_text(),
                                  [{"scene_idx": i, "graph": g} for i, g in enumerate(graphs)])
    assert '"tool": "t1"' in summary
    assert "{scenes}" not in summary


def test_end_is_optional_for_native_and_repaired_actions():
    expected = parsed().graph
    for separator in (" - ", " -> ", " → "):
        text = TEXT.removesuffix("[End]").replace(" - ", separator)
        assert parsed(text).graph == expected
        record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
        assert failure is None and record["graph"] == expected


@pytest.mark.parametrize("section", ["Context", "Entities", "Actions"])
def test_all_three_sections_remain_required_without_end(section):
    text = TEXT.removesuffix("[End]").replace(f"[{section}]", "")
    assert parse_or_repair_graph(text).graph is None


@pytest.mark.parametrize("text", [TEXT, TEXT.removesuffix("[End]")])
def test_token_termination_is_failure_even_with_complete_graph(text):
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text,
                                         error="graph: output truncated at token limit")
    assert record is None
    assert failure["raw_response"] == text
    assert failure["failure_kind"] == "generation"


def test_summary_receives_raw_but_does_not_count_it_as_structured(ready_context, monkeypatch):
    from extraction import steps
    from extraction.backends import GeminiGenerationOutcome
    from pipeline_runtime import read_json

    context = ready_context
    ids = [row["content_id"] for row in context.require_ready_cohort()["catalog"]]
    for cid in ids:
        write_scene_records(context.scene_arm_dir("graph_qwen") / f"{cid}.jsonl", [
            {"scene_idx": 0, "graph": parsed().graph},
            {"scene_idx": 1, "graph": "unstructured observation"},
        ])

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                assert task.max_new_tokens == 1024
                assert '"target": "food-1"' in task.prompt
                assert "unstructured observation" in task.prompt
                callback(GeminiGenerationOutcome(task.task_id, "A seafood preparation demonstration."))

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    root = Path(__file__).resolve().parents[3]
    result = steps.summarize_graph(context, model="gemini", arm="graph_qwen",
                                   schema=root / "prompts/summary_graph_v6.md")
    assert result["failure_count"] == 0
    for cid in ids:
        doc = read_json(context.summary_arm_dir("graph_qwen") / f"{cid}.json")
        assert doc["scene_count"] == 2
        assert doc["provenance"]["normal_scene_count"] == 1
        assert doc["provenance"]["raw_scene_count"] == 1
        assert doc["provenance"]["text_scene_count"] == 1


@pytest.mark.parametrize("mutation", [
    lambda g: g.update(medium="video"),
    lambda g: g.update(format="tutorial"),
    lambda g: g.update(topics=["a b c d e"]),
    lambda g: g.update(topics=["a"] * 4),
    lambda g: g["entities"][0].update(attributes=["a b c d e f g"]),
    lambda g: g["entities"][0].update(attributes=["a", "b", "c"]),
    lambda g: g.update(actions=g["actions"] * 5),
    lambda g: g["entities"].extend({"id": f"x{i}", "name": "cup", "attributes": []}
                                  for i in range(7)),
])
def test_prompt_limits_do_not_reject_or_trim_graph(mutation):
    graph = parsed().graph
    mutation(graph)
    validate_graph_structure(graph)
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, json.dumps(graph))
    assert failure is None and record["graph"] == graph
    assert record["semantic_warnings"] == []


@pytest.mark.parametrize("text,tag", [
    (TEXT.replace("[Context]", ""), "MISSING_REQUIRED"),
    (TEXT.replace("[Entities]", ""), "MISSING_REQUIRED"),
    (TEXT.replace("[Actions]", ""), "MISSING_REQUIRED"),
    (TEXT.replace("format: demonstration\n", ""), "MISSING_REQUIRED"),
    (TEXT.replace("food-1: sea urchin", "food-1:"), "MISSING_REQUIRED"),
    (TEXT.replace("; knife-1", ""), "INVALID_ACTION_SYNTAX"),
    (TEXT.replace("cross-cutting", "cutting - slicing"), "INVALID_ACTION_SYNTAX"),
    (TEXT.replace("; knife-1", "; absent"), "INVALID_REFERENCE"),
    (TEXT.replace("[Actions]", "person-1: person\n[Actions]"), "DUPLICATE_ENTITY_ID"),
    ("{broken JSON", "PARSE_ERROR"),
    (TEXT + "\nUnexpected commentary", "PARSE_ERROR"),
])
def test_warning_tags_are_saved_in_order_and_survive_summary_roundtrip(tmp_path, text, tag):
    from pipeline_runtime import read_jsonl

    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert failure is None and record["semantic_warnings"] == [tag]
    path = tmp_path / "video.jsonl"
    write_scene_records(path, [record])
    stored = read_jsonl(path)[0]
    assert list(stored) == ["content_id", "scene_idx", "tokens", "warning", "scene_graph"]
    assert stored["warning"] == [tag] and stored["scene_graph"] == text
    records = minimal_graph_records(read_scene_records(path), path)
    assert records[0]["semantic_warnings"] == [tag]
    assert json.loads(graph_summary_prompt("{scenes}", records))[0]["observation"] == text


def test_duplicate_and_reference_tags_are_collected_without_duplicates():
    graph = parsed().graph
    graph["entities"].append(copy.deepcopy(graph["entities"][0]))
    graph["actions"][0].update(actor="missing", tool="missing")
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, json.dumps(graph))
    assert failure is None
    assert record["semantic_warnings"] == ["DUPLICATE_ENTITY_ID", "INVALID_REFERENCE"]


@pytest.mark.parametrize("mutation,tag", [
    (lambda g: g.pop("medium"), "MISSING_REQUIRED"),
    (lambda g: g["actions"][0].pop("tool"), "INVALID_ACTION_SYNTAX"),
    (lambda g: g["actions"][0].update(action=""), "INVALID_ACTION_SYNTAX"),
    (lambda g: g["actions"][0].update(actor=None, target=None), "INVALID_ACTION_SYNTAX"),
    (lambda g: g.update(topics="wrong type"), "PARSE_ERROR"),
])
def test_json_structure_uses_specific_tags(mutation, tag):
    graph = parsed().graph
    mutation(graph)
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, json.dumps(graph))
    assert failure is None and record["semantic_warnings"] == [tag]
