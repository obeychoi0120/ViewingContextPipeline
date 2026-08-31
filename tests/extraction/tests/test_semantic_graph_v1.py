from __future__ import annotations

import pytest

from extraction.semantic_graph import (
    SCENE_EXTRACTION_PROMPT,
    SemanticGraphError,
    graph_semantic_warnings,
    graph_summary_prompt,
    validate_summary,
)
from extraction.semantic_graph.taxonomy import ENTITY_GUIDANCE_MAX, SETTING_CONTEXTS
from extraction.summary_validation import (
    SUMMARY_SECTIONS,
    serialize_summary_sections,
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


def test_semantic_schema_and_reference_errors_are_not_rejected() -> None:
    dangling = graph_with_dangling_reference()
    assert graph_semantic_warnings(dangling) == [
        "unresolved_reference:static_relations[0].object_id='e4'"
    ]

    arbitrary = {"triples": [], "extra": "preserved"}
    assert graph_semantic_warnings(arbitrary) == [
        "missing_top_level_fields:setting_context,entities,events,static_relations,semantic_topics,affect",
        "unexpected_top_level_fields:extra,triples",
    ]


def test_taxonomy_and_prompt_are_multi_image_and_consistent() -> None:
    assert ENTITY_GUIDANCE_MAX == 6
    assert "chronological keyframes" in SCENE_EXTRACTION_PROMPT
    assert "If entities is empty" in SCENE_EXTRACTION_PROMPT
    for value in SETTING_CONTEXTS:
        assert value in SCENE_EXTRACTION_PROMPT


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
    assert '"object_id": "e4"' in prompt


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
