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
    assert result.graph == expected
    assert result.parse_mode == "repaired"


@pytest.mark.parametrize(("text", "error"), [("", "empty"), ("plain prose", "repair")])
def test_unrepairable_output_is_reported(text: str, error: str) -> None:
    result = parse_or_repair_graph(text)
    assert result.graph is None
    assert result.parse_mode is None
    assert error.lower() in str(result.error).lower()
