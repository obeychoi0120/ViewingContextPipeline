from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_provisional_ontology_v1_contains_only_used_taxonomy_fields() -> None:
    value = json.loads(
        (ROOT / "contracts/extraction/relational_graph_ontology_v1.json").read_text(encoding="utf-8")
    )
    assert value["ontology_id"] == "relational-graph-ontology/v1"
    assert value["status"] == "provisional"
    assert value["context_fields"] == ["setting", "scene_function", "mood_bin", "media_form"]
    assert set(value["entity_fields"]) == {"local_id", "name", "category", "role"}
    assert {"scene_type", "people_density", "graphic_density"}.issubset(value["excluded_fields"])
