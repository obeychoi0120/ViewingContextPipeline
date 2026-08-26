from __future__ import annotations

import pytest

from extraction.semantic_graph.json_repair import parse_or_repair_graph


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1,}', {"a": 1}),
        ("{'a': None, 'b': True}", {"a": None, "b": True}),
        ('{"entities":[{"id":"e1"}],"events":[', {"entities": [{"id": "e1"}], "events": []}),
        ('{"label":"unfinished', {"label": "unfinished"}),
    ],
)
def test_deterministic_repair_cases(text: str, expected: dict) -> None:
    result = parse_or_repair_graph(text)
    assert result.status == "repaired"
    assert result.graph == expected


def test_fenced_and_noisy_json_is_parsed_without_repair() -> None:
    result = parse_or_repair_graph('prefix ```json\n{"value": 1}\n``` suffix')
    assert result.status == "parsed"
    assert result.graph == {"value": 1}


def test_later_valid_object_is_selected_after_malformed_object() -> None:
    result = parse_or_repair_graph('bad {"a": } then {"value": 2}')
    assert result.status == "parsed"
    assert result.graph == {"value": 2}


@pytest.mark.parametrize(("text", "error"), [("", "empty"), ("plain prose", "repair")])
def test_unrepairable_output_is_reported(text: str, error: str) -> None:
    result = parse_or_repair_graph(text)
    assert result.status == "failed"
    assert result.graph is None
    assert error.lower() in str(result.error).lower()
