from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.common.manifest import (
    ManifestContractError,
    parse_manifest_text,
    read_manifest_rows,
)
from src.scene_context_extraction.graph_core.ontology import (
    ENTITY_CATEGORIES,
    RELATION_DEFINITIONS,
    RELATION_TYPES,
    SCENE_CONTEXT_ONTOLOGY_ID,
    SCENE_TYPES,
    SceneContextOntologyError,
    load_scene_context_ontology,
)
from src.scene_context_extraction.graph_core.video_context import (
    CONTEXT_FIELDS,
    VIDEO_CONTEXT_FIELDS,
)
from src.video_profile_generation.models import VideoProfileDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_import_does_not_require_google_cloud() -> None:
    script = """
import builtins

original_import = builtins.__import__

def import_without_google_cloud(name, *args, **kwargs):
    if name.startswith("google.cloud"):
        raise ImportError("google.cloud is unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_google_cloud
from src.common.manifest import read_manifest_rows
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_repository_manifest_is_canonical() -> None:
    canonical = read_manifest_rows(
        PROJECT_ROOT / "contracts" / "manifest.csv"
    )

    assert len(canonical) == 512


def test_manifest_contract_rejects_extra_columns_and_duplicates() -> None:
    with pytest.raises(ManifestContractError, match="columns"):
        parse_manifest_text("content_id,url,path\na,https://example.com,a\n")
    with pytest.raises(ManifestContractError, match="duplicates content_id"):
        parse_manifest_text(
            "content_id,url\n"
            "a,https://example.com/a\n"
            "a,https://example.com/b\n"
        )
    with pytest.raises(ManifestContractError, match="duplicates url"):
        parse_manifest_text(
            "content_id,url\n"
            "a,https://example.com/a\n"
            "b,https://example.com/a\n"
        )


def test_scene_context_ontology_contract_exports_expected_v1() -> None:
    contract = load_scene_context_ontology()

    assert contract["ontology_id"] == SCENE_CONTEXT_ONTOLOGY_ID
    assert SCENE_CONTEXT_ONTOLOGY_ID == "scene_context_ontology/v1"
    assert SCENE_TYPES == {
        "people_social",
        "person_portrait",
        "object_product",
        "food_drink",
        "nature_landscape",
        "animal_pet",
        "graphic_information",
        "sport_fitness",
        "vehicle_transport",
        "unknown",
    }
    assert ENTITY_CATEGORIES == {
        "person",
        "animal",
        "food",
        "vehicle",
        "device",
        "object",
        "building",
        "nature",
        "text",
        "unknown",
    }
    assert RELATION_TYPES == {
        "DOING",
        "WEARING",
        "IS_A",
        "AT",
        "INTERACTS_WITH",
    }
    assert (
        RELATION_DEFINITIONS["INTERACTS_WITH"]["value_kind"]
        == "entity_local_id"
    )


def test_scene_context_ontology_rejects_duplicate_and_invalid_relation(
    tmp_path: Path,
) -> None:
    contract = load_scene_context_ontology()
    contract["vocabularies"]["scene_types"].append("unknown")
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(SceneContextOntologyError, match="duplicate"):
        load_scene_context_ontology(duplicate_path)

    contract = load_scene_context_ontology()
    contract["relations"]["DOING"]["value_kind"] = "unbounded"
    relation_path = tmp_path / "relation.json"
    relation_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(SceneContextOntologyError, match="value_kind"):
        load_scene_context_ontology(relation_path)


def test_minimal_producer_profile_fixtures_remain_valid() -> None:
    ond = json.loads(
        (
            PROJECT_ROOT
            / "tests"
            / "fixtures"
            / "video_context_ond_minimal.json"
        ).read_text(encoding="utf-8")
    )
    gt_text = (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "video_profile_minimal.json"
    ).read_text(encoding="utf-8")

    assert set(ond) == VIDEO_CONTEXT_FIELDS
    assert set(ond["context"]) == CONTEXT_FIELDS
    assert VideoProfileDocument.model_validate_json(gt_text).ontology_version == (
        "viewing-ontology-contract/v3"
    )
