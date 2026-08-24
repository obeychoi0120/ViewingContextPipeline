from __future__ import annotations

import json
from pathlib import Path

import pytest

from viewing_context_pipeline.extraction.common.manifest import (
    ManifestContractError,
    parse_manifest_text,
)
from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.ontology import (
    SCENE_CONTEXT_ONTOLOGY_ID,
    SceneContextOntologyError,
    load_scene_context_ontology,
)


def test_microlens_manifest_accepts_content_ids_only() -> None:
    assert parse_manifest_text("content_id\nmicrolens_100k_00001\n") == [
        {"content_id": "microlens_100k_00001"}
    ]
    with pytest.raises(ManifestContractError, match="columns"):
        parse_manifest_text("content_id,url\na,https://example.com\n")
    with pytest.raises(ManifestContractError, match="duplicates content_id"):
        parse_manifest_text("content_id\na\na\n")


def test_scene_context_ontology_is_current_v1() -> None:
    contract = load_scene_context_ontology()
    assert contract["ontology_id"] == SCENE_CONTEXT_ONTOLOGY_ID
    assert SCENE_CONTEXT_ONTOLOGY_ID == "scene_context_ontology/v1"


def test_scene_context_ontology_rejects_duplicates(tmp_path: Path) -> None:
    contract = load_scene_context_ontology()
    contract["vocabularies"]["scene_types"].append("unknown")
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(SceneContextOntologyError, match="duplicate"):
        load_scene_context_ontology(path)
