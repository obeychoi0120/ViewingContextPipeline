"""New contract round trips, conservative repairs, and unsupported format rejection."""
import json

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
person-1 - cross-cutting - food-1; knife-1; unknown
[End]"""


def parsed(text=TEXT):
    result = parse_or_repair_graph(text)
    assert result.error is None
    validate_graph_structure(result.graph)
    return result


@pytest.mark.parametrize("mutation", [
    lambda g: g.update(topics="cooking"),
    lambda g: g["actions"][0].pop("tool"),
    lambda g: g["actions"][0].update(actor=1),
    lambda g: g["actions"][0].update(action=""),
])
def test_required_shape_integrity(mutation):
    graph = parsed().graph
    mutation(graph)
    with pytest.raises(OutputValidationError):
        validate_graph_structure(graph)
    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, json.dumps(graph))
    assert failure is None and record["parse_mode"] == "text"


def test_storage_and_summary_preserve_current_structured_and_raw(tmp_path):
    path = tmp_path / "video.jsonl"
    values = [parsed().graph, "unstructured observation"]
    write_scene_records(path, [{"scene_idx": i, "graph": value} for i, value in enumerate(values)])
    records = minimal_graph_records(read_scene_records(path), path)
    observations = json.loads(graph_summary_prompt("{scenes}", records))
    assert [row["observation"] for row in observations] == values
    assert records[-1]["parse_mode"] == "text"


@pytest.mark.parametrize("section", ["Context", "Entities", "Actions"])
def test_all_three_sections_remain_required_without_end(section):
    text = TEXT.removesuffix("[End]").replace(f"[{section}]", "")
    assert parse_or_repair_graph(text).graph is None


@pytest.mark.parametrize("text,tag", [
    (TEXT.replace("[Context]", ""), "MISSING_REQUIRED"),
    (TEXT.replace("[Entities]", ""), "MISSING_REQUIRED"),
    (TEXT.replace("[Actions]", ""), "MISSING_REQUIRED"),
    (TEXT.replace("format: demonstration\n", ""), "MISSING_REQUIRED"),
    (TEXT.replace("food-1: sea urchin", "food-1:"), "MISSING_REQUIRED"),
    (TEXT.replace("; knife-1", ""), "PARSE_FAILED"),
    (TEXT.replace("cross-cutting", "cutting - slicing"), "PARSE_FAILED"),
    ("{broken JSON", "PARSE_FAILED"),
    (TEXT + "\nUnexpected commentary", "PARSE_FAILED"),
])
def test_warning_tags_are_saved_in_order_and_survive_summary_roundtrip(tmp_path, text, tag):
    from artifact_io import read_jsonl

    record, failure = graph_scene_result({"scene_idx": 0, "keyframes": []}, text)
    assert failure is None and record["semantic_warnings"] == [tag]
    path = tmp_path / "video.jsonl"
    write_scene_records(path, [record])
    stored = read_jsonl(path)[0]
    assert list(stored) == ["content_id", "scene_idx", "tokens", "warning", "scene_graph"]
    assert stored["warning"] == [tag] and stored["scene_graph"] == record["graph"]
    assert stored["scene_graph"] == text
    records = minimal_graph_records(read_scene_records(path), path)
    assert records[0]["semantic_warnings"] == [tag]
    assert json.loads(graph_summary_prompt("{scenes}", records))[0]["observation"] == record["graph"]
