from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from extraction.semantic_graph import (
    SemanticGraphError,
    graph_semantic_warnings,
    graph_summary_prompt,
    validate_summary,
)
from extraction.summary_validation import (
    SUMMARY_SECTIONS,
    serialize_summary_sections,
)


ROOT = Path(__file__).resolve().parents[3]


def graph_with_dangling_reference() -> dict:
    return {
        "setting_context": "indoor",
        "entities": [
            {
                "local_id": "e1",
                "name": "person",
                "salience": "primary",
                "function": None,
                "count": "one",
            }
        ],
        "events": [
            {
                "local_id": "ev1",
                "actor_id": "e1",
                "action": "hold",
                "target_id": "e4",
                "instrument_id": None,
                "location_id": None,
            }
        ],
        "semantic_topics": [],
        "affect": {
            "valence": "neutral",
            "arousal": "medium",
        },
    }


def test_semantic_schema_and_reference_errors_are_not_rejected() -> None:
    dangling = graph_with_dangling_reference()
    assert graph_semantic_warnings(dangling) == ["unresolved_reference:events[0].target_id='e4'"]


def test_raw_graph_is_not_compacted_when_semantic_warnings_are_emitted() -> None:
    graph = graph_with_dangling_reference()
    graph["entities"].extend(
        {
            "local_id": f"e{index}",
            "name": f"visible entity {index}",
            "salience": "background",
            "function": None,
            "count": "one",
        }
        for index in range(2, 8)
    )
    graph["events"][0]["role"] = "legacy"
    original = deepcopy(graph)

    first = graph_semantic_warnings(graph)
    second = graph_semantic_warnings(graph)
    reordered = graph_semantic_warnings(
        {key: graph[key] for key in reversed(graph)}
    )

    assert graph == original
    assert first == second == reordered
    assert "entity_count_out_of_range:7" in first
    assert "unexpected_fields:events[0]:role" in first
    assert "removed_field:graph.events[0].role" in first
    assert "unresolved_reference:events[0].target_id='e4'" not in first

    arbitrary = {"triples": [], "extra": "preserved"}
    assert graph_semantic_warnings(arbitrary) == [
        "missing_fields:top_level:setting_context,entities,events,semantic_topics,affect",
        "unexpected_fields:top_level:extra,triples",
        "entity_count_out_of_range:0",
    ]


def test_graph_scene_prompt_is_file_backed_and_contains_the_contract() -> None:
    prompt = (ROOT / "config/prompts/graph_scene_v2.md").read_text(encoding="utf-8")

    assert "one to three chronological keyframes" in prompt
    assert "Output one to six semantically meaningful visible entities" in prompt
    assert '"salience"' in prompt
    assert '"function"' in prompt
    assert '"count"' in prompt
    assert "Output at most four directly visible events" in prompt
    assert "Output at most three topics" in prompt
    assert "static_relations" in prompt
    assert "subject_ids" in prompt
    assert "Extract only information directly grounded in visible pixels." in prompt
    for value in (
        "indoor",
        "outdoor_urban",
        "outdoor_nature",
        "transport",
        "unknown",
    ):
        assert value in prompt
    assert "ground truth" not in prompt.lower()
    assert "student" not in prompt.lower()
    assert "retry" not in prompt.lower()
    assert not (ROOT / "config/prompts/graph_scene_v1.md").exists()
    assert not (ROOT / "docs/next_graph_ontology/prompt.py").exists()
    assert not (ROOT / "docs/next_graph_ontology/taxonomy.py").exists()


def test_graph_summary_preserves_raw_graph_and_sorts_scenes() -> None:
    records = []
    for scene_idx, start in ((1, 30), (0, 0)):
        records.append(
            {
                "scene_idx": scene_idx,
                "keyframes": [start + 5, start + 15, start + 25],
                "graph": graph_with_dangling_reference(),
            }
        )

    prompt = graph_summary_prompt("Graphs:\n{scenes}", records)

    assert prompt.index("Scene 0") < prompt.index("Scene 1")
    assert '"target_id": "e4"' in prompt


def test_summary_requires_seven_labeled_lines_in_canonical_order() -> None:
    sections = {name: f"visible {name}" for name in SUMMARY_SECTIONS}
    labeled_text = "\n".join(f"{name}: {value}" for name, value in sections.items())
    parsed = validate_summary(labeled_text)

    assert parsed == sections
    assert serialize_summary_sections(parsed).splitlines() == [
        "Setting and environments: visible setting_and_environments.",
        "Main characters and objects: visible main_characters_and_objects.",
        "Chronological events: visible chronological_events.",
        "Relations: visible relations.",
        "Visual atmosphere: visible visual_atmosphere.",
        "Visible affect: visible visible_affect.",
        "Semantic topics: visible semantic_topics.",
    ]
    with pytest.raises(SemanticGraphError, match="exactly seven"):
        validate_summary("free-form summary")
    wrong_order = list(sections.items())
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(SemanticGraphError, match="canonical order"):
        validate_summary("\n".join(f"{name}: {value}" for name, value in wrong_order))
    with pytest.raises(SemanticGraphError, match="exactly seven"):
        validate_summary(labeled_text + "\nlegacy_field: not allowed")
