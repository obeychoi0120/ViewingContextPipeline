"""Load the versioned Scene Context ontology contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


SCENE_CONTEXT_ONTOLOGY_ID = "scene_context_ontology/v1"
ONTOLOGY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "scene_context_ontology_v1.json"
)


class SceneContextOntologyError(ValueError):
    """Raised when the Scene Context ontology contract is invalid."""


def load_scene_context_ontology(
    path: str | Path = ONTOLOGY_CONTRACT_PATH,
) -> dict[str, Any]:
    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneContextOntologyError(
            f"failed to read Scene Context ontology {contract_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SceneContextOntologyError("Scene Context ontology must be a JSON object")

    expected_top_level = {
        "ontology_id",
        "vocabularies",
        "visual_style_cues",
        "entities",
        "relations",
        "motif_filters",
    }
    if set(payload) != expected_top_level:
        raise SceneContextOntologyError(
            "Scene Context ontology top-level fields must be exactly "
            + ", ".join(sorted(expected_top_level))
        )
    if payload["ontology_id"] != SCENE_CONTEXT_ONTOLOGY_ID:
        raise SceneContextOntologyError(
            f"ontology_id must be {SCENE_CONTEXT_ONTOLOGY_ID!r}"
        )

    vocabularies = _require_mapping(payload, "vocabularies")
    expected_vocabularies = {
        "scene_types",
        "people_density",
        "face_prominence",
        "mood_bins",
        "affect_cues",
        "scene_functions",
        "styles",
        "settings",
    }
    _require_exact_keys(vocabularies, expected_vocabularies, "vocabularies")
    for name in expected_vocabularies:
        _require_string_list(vocabularies, name, f"vocabularies.{name}")

    visual_style_cues = _require_mapping(payload, "visual_style_cues")
    expected_cues = {
        "media_form",
        "fantasy_element",
        "shot_scale",
        "graphic_density",
        "composition_density",
    }
    _require_exact_keys(visual_style_cues, expected_cues, "visual_style_cues")
    for name in expected_cues:
        _require_string_list(
            visual_style_cues,
            name,
            f"visual_style_cues.{name}",
        )

    entities = _require_mapping(payload, "entities")
    _require_exact_keys(entities, {"categories", "roles"}, "entities")
    _require_string_list(entities, "categories", "entities.categories")
    _require_string_list(entities, "roles", "entities.roles")

    relations = _require_mapping(payload, "relations")
    if not relations:
        raise SceneContextOntologyError("relations must not be empty")
    for relation_name, relation in relations.items():
        if not isinstance(relation_name, str) or not relation_name:
            raise SceneContextOntologyError(
                "relations keys must be non-empty strings"
            )
        if not isinstance(relation, dict):
            raise SceneContextOntologyError(
                f"relations.{relation_name} must be an object"
            )
        _require_exact_keys(
            relation,
            {"value_kind", "description"},
            f"relations.{relation_name}",
        )
        if relation["value_kind"] not in {"free_text", "entity_local_id"}:
            raise SceneContextOntologyError(
                f"relations.{relation_name}.value_kind is invalid"
            )
        if not isinstance(relation["description"], str) or not relation[
            "description"
        ].strip():
            raise SceneContextOntologyError(
                f"relations.{relation_name}.description must be a non-empty string"
            )

    motif_filters = _require_mapping(payload, "motif_filters")
    expected_filters = {
        "excluded_entities",
        "excluded_values",
        "excluded_roles",
    }
    _require_exact_keys(motif_filters, expected_filters, "motif_filters")
    for name in expected_filters:
        _require_string_list(motif_filters, name, f"motif_filters.{name}")
    return payload


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise SceneContextOntologyError(f"{key} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    location: str,
) -> None:
    if set(value) != expected:
        raise SceneContextOntologyError(
            f"{location} fields must be exactly {', '.join(sorted(expected))}"
        )


def _require_string_list(
    parent: Mapping[str, Any],
    key: str,
    location: str,
) -> list[str]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise SceneContextOntologyError(
            f"{location} must be a non-empty string list"
        )
    if len(value) != len(set(value)):
        raise SceneContextOntologyError(f"{location} contains duplicate values")
    return value


_ONTOLOGY = load_scene_context_ontology()
_VOCABULARIES = _ONTOLOGY["vocabularies"]
_ENTITIES = _ONTOLOGY["entities"]
_MOTIF_FILTERS = _ONTOLOGY["motif_filters"]

SCENE_TYPES = frozenset(_VOCABULARIES["scene_types"])
PEOPLE_DENSITY = frozenset(_VOCABULARIES["people_density"])
FACE_PROMINENCE = frozenset(_VOCABULARIES["face_prominence"])
MOOD_BINS = frozenset(_VOCABULARIES["mood_bins"])
AFFECT_CUES = frozenset(_VOCABULARIES["affect_cues"])
SCENE_FUNCTIONS = frozenset(_VOCABULARIES["scene_functions"])
STYLES = frozenset(_VOCABULARIES["styles"])
SETTINGS = frozenset(_VOCABULARIES["settings"])

VISUAL_STYLE_CUES = MappingProxyType(
    {
        name: frozenset(values)
        for name, values in _ONTOLOGY["visual_style_cues"].items()
    }
)
MEDIA_FORM = VISUAL_STYLE_CUES["media_form"]
FANTASY_ELEMENT = VISUAL_STYLE_CUES["fantasy_element"]
SHOT_SCALE = VISUAL_STYLE_CUES["shot_scale"]
GRAPHIC_DENSITY = VISUAL_STYLE_CUES["graphic_density"]
COMPOSITION_DENSITY = VISUAL_STYLE_CUES["composition_density"]

ENTITY_CATEGORIES = frozenset(_ENTITIES["categories"])
ENTITY_ROLES = frozenset(_ENTITIES["roles"])
RELATION_DEFINITIONS = MappingProxyType(
    {
        name: MappingProxyType(dict(definition))
        for name, definition in _ONTOLOGY["relations"].items()
    }
)
RELATION_TYPES = frozenset(RELATION_DEFINITIONS)

MOTIF_EXCLUDE_ENTITIES = frozenset(_MOTIF_FILTERS["excluded_entities"])
MOTIF_EXCLUDE_VALUES = frozenset(_MOTIF_FILTERS["excluded_values"])
MOTIF_EXCLUDE_ROLES = frozenset(_MOTIF_FILTERS["excluded_roles"])
