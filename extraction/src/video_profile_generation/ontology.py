"""Load and validate the external canonical viewing ontology contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import VideoProfileDetails


ONTOLOGY_VERSION = "viewing-ontology-contract/v3"
_FACETS = ("topic", "content_type", "intent", "media_form", "presentation")
_CONCEPT_FACETS = ("topic", "content_type", "intent")
_PRESENTATION_DIMENSIONS = ("pace", "information_density", "complexity")
_CONTRACT_KEYS = {
    "ontology_version",
    "facet_specs",
    "concept_id_policy",
    "unmatched_concept_policy",
    "unknown_value_policy",
    "closed_axes",
    "closed_axis_definitions",
    "canonical_concepts",
}


class OntologyContractError(ValueError):
    """Raised when the configured viewing ontology contract is invalid."""


@dataclass(frozen=True)
class ViewingOntologyContract:
    payload: dict[str, Any]
    version: str
    concept_ids: dict[str, frozenset[str]]
    media_forms: frozenset[str]
    presentation_values: dict[str, frozenset[str]]
    facet_specs: dict[str, dict[str, Any]]


def _require_non_empty_text(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    if any(
        not isinstance(record.get(field), str) or not record[field].strip()
        for field in fields
    ):
        raise OntologyContractError(
            f"ontology record requires non-empty fields: {fields}"
        )


def _validate_facet_specs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("facet_specs")
    if not isinstance(raw, list):
        raise OntologyContractError("facet_specs must be a list")
    facets: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise OntologyContractError("facet_specs entries must be objects")
        _require_non_empty_text(
            item,
            ("facet", "ko_label", "description", "value_kind", "selection_rule"),
        )
        facet = item["facet"]
        facets.append(facet)
        base_keys = {
            "facet",
            "ko_label",
            "description",
            "value_kind",
            "selection_rule",
        }
        if facet in _CONCEPT_FACETS:
            if (
                set(item) != base_keys | {"min_items", "max_items"}
                or item["value_kind"] != "canonical_concept"
                or not isinstance(item["min_items"], int)
                or not isinstance(item["max_items"], int)
                or item["min_items"] < 0
                or item["max_items"] < item["min_items"]
            ):
                raise OntologyContractError(f"invalid facet_specs.{facet}")
        elif facet == "media_form":
            if (
                set(item) != base_keys | {"min_items", "max_items", "allow_unknown"}
                or item["value_kind"] != "closed_enum"
                or item["allow_unknown"] is not True
                or not isinstance(item["min_items"], int)
                or not isinstance(item["max_items"], int)
                or item["min_items"] < 1
                or item["max_items"] < item["min_items"]
            ):
                raise OntologyContractError("invalid facet_specs.media_form")
        elif facet == "presentation":
            if (
                set(item) != base_keys | {"dimensions", "allow_unknown"}
                or item["value_kind"] != "ordinal"
                or item["allow_unknown"] is not True
                or item["dimensions"] != list(_PRESENTATION_DIMENSIONS)
            ):
                raise OntologyContractError("invalid facet_specs.presentation")
        result[facet] = dict(item)
    if tuple(facets) != _FACETS:
        raise OntologyContractError("facet_specs does not match the canonical facets")
    return result


def _validate_closed_definitions(
    raw: object,
    expected_ids: tuple[str, ...],
    *,
    ordinal: bool,
) -> None:
    if not isinstance(raw, list) or len(raw) != len(expected_ids):
        raise OntologyContractError("closed axis definitions do not match its values")
    actual_ids = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise OntologyContractError("closed axis definitions must be objects")
        _require_non_empty_text(
            item,
            (
                "value_id",
                "ko_label",
                "description",
                "allowed_boundary",
                "forbidden_boundary",
            ),
        )
        actual_ids.append(item["value_id"])
        expected_rank = index if ordinal else None
        if item.get("ordinal_rank") != expected_rank:
            raise OntologyContractError("closed axis ordinal ranks are invalid")
    if tuple(actual_ids) != expected_ids:
        raise OntologyContractError("closed axis definition IDs are invalid")


def _parse_concept_ids(payload: dict[str, Any]) -> dict[str, frozenset[str]]:
    concepts_raw = payload.get("canonical_concepts")
    if not isinstance(concepts_raw, dict):
        raise OntologyContractError("canonical_concepts must be an object")
    concept_ids: dict[str, frozenset[str]] = {}
    for facet in _CONCEPT_FACETS:
        values = concepts_raw.get(facet)
        if not isinstance(values, list) or not values:
            raise OntologyContractError(
                f"canonical_concepts.{facet} must be a non-empty list"
            )
        ids = []
        parents: list[str | None] = []
        for item in values:
            if not isinstance(item, dict) or item.get("facet") != facet:
                raise OntologyContractError(f"invalid {facet} concept record")
            _require_non_empty_text(
                item,
                (
                    "concept_id",
                    "ko_label",
                    "description",
                    "allowed_boundary",
                    "forbidden_boundary",
                ),
            )
            aliases = item.get("aliases")
            if (
                not isinstance(aliases, list)
                or any(not isinstance(alias, str) or not alias for alias in aliases)
                or len(aliases) != len(set(aliases))
            ):
                raise OntologyContractError(f"invalid {facet} concept aliases")
            if not isinstance(item.get("assignable_as_preference"), bool):
                raise OntologyContractError(
                    f"invalid {facet} preference assignment flag"
                )
            parent_id = item.get("parent_id")
            if parent_id is not None and (
                not isinstance(parent_id, str) or not parent_id
            ):
                raise OntologyContractError(f"invalid {facet} parent concept")
            ids.append(item["concept_id"])
            parents.append(parent_id)
        if len(ids) != len(set(ids)):
            raise OntologyContractError(f"duplicate {facet} concept ID")
        unknown_parents = sorted(
            parent_id
            for parent_id in parents
            if parent_id is not None and parent_id not in ids
        )
        if unknown_parents:
            raise OntologyContractError(
                f"{facet} contains unknown parent IDs: {unknown_parents}"
            )
        concept_ids[facet] = frozenset(ids)
    return concept_ids


def load_viewing_ontology_contract(
    path: str | Path,
) -> ViewingOntologyContract:
    contract_path = Path(path)
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OntologyContractError(
            f"failed to read viewing ontology contract {contract_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise OntologyContractError("viewing ontology contract must be a JSON object")
    if set(raw) != _CONTRACT_KEYS:
        missing = sorted(_CONTRACT_KEYS - set(raw))
        extra = sorted(set(raw) - _CONTRACT_KEYS)
        raise OntologyContractError(
            f"invalid viewing ontology keys: missing={missing}, extra={extra}"
        )
    if raw.get("ontology_version") != ONTOLOGY_VERSION:
        raise OntologyContractError("unsupported viewing ontology version")
    facet_specs = _validate_facet_specs(raw)
    for field in (
        "concept_id_policy",
        "unmatched_concept_policy",
        "unknown_value_policy",
    ):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise OntologyContractError(f"{field} must be non-empty")
    concept_ids = _parse_concept_ids(raw)

    closed_axes = raw.get("closed_axes")
    media_forms = (
        closed_axes.get("media_form") if isinstance(closed_axes, dict) else None
    )
    if (
        not isinstance(media_forms, list)
        or not media_forms
        or any(not isinstance(value, str) or not value for value in media_forms)
        or len(media_forms) != len(set(media_forms))
    ):
        raise OntologyContractError("closed_axes.media_form must be a non-empty list")
    presentation = (
        closed_axes.get("presentation") if isinstance(closed_axes, dict) else None
    )
    if not isinstance(presentation, dict):
        raise OntologyContractError("closed_axes.presentation must be an object")
    presentation_values: dict[str, frozenset[str]] = {}
    for dimension in _PRESENTATION_DIMENSIONS:
        values = presentation.get(dimension)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise OntologyContractError(
                f"closed_axes.presentation.{dimension} must be a non-empty list"
            )
        presentation_values[dimension] = frozenset(values)

    definitions = raw.get("closed_axis_definitions")
    if not isinstance(definitions, dict):
        raise OntologyContractError("closed_axis_definitions must be an object")
    _validate_closed_definitions(
        definitions.get("media_form"),
        tuple(media_forms),
        ordinal=False,
    )
    presentation_definitions = definitions.get("presentation")
    if not isinstance(presentation_definitions, dict):
        raise OntologyContractError(
            "closed_axis_definitions.presentation must be an object"
        )
    for dimension in _PRESENTATION_DIMENSIONS:
        _validate_closed_definitions(
            presentation_definitions.get(dimension),
            tuple(presentation[dimension]),
            ordinal=True,
        )
    return ViewingOntologyContract(
        payload=raw,
        version=ONTOLOGY_VERSION,
        concept_ids=concept_ids,
        media_forms=frozenset(media_forms),
        presentation_values=presentation_values,
        facet_specs=facet_specs,
    )


def validate_profile_details(
    details: VideoProfileDetails,
    contract: ViewingOntologyContract,
) -> None:
    profile = details.profile
    fields = {
        "topic": profile.topic,
        "content_type": profile.content_type,
        "intent": profile.intent,
    }
    for facet, values in fields.items():
        field_contract = contract.facet_specs[facet]
        if (
            not field_contract["min_items"]
            <= len(values)
            <= field_contract["max_items"]
        ):
            raise OntologyContractError(f"{facet} cardinality violates facet_specs")
        invalid = sorted(
            {
                item.concept_id
                for item in values
                if item.concept_id not in contract.concept_ids[facet]
            }
        )
        if invalid:
            raise OntologyContractError(
                f"{facet} contains unknown ontology IDs: {invalid}"
            )
    invalid_media = sorted(
        value
        for value in profile.media_form
        if value != "unknown" and value not in contract.media_forms
    )
    if invalid_media:
        raise OntologyContractError(
            f"media_form contains unknown values: {invalid_media}"
        )
    media_contract = contract.facet_specs["media_form"]
    if (
        not media_contract["min_items"]
        <= len(profile.media_form)
        <= media_contract["max_items"]
    ):
        raise OntologyContractError("media_form cardinality violates facet_specs")
    presentation = profile.presentation
    for dimension in contract.facet_specs["presentation"]["dimensions"]:
        value = getattr(presentation, dimension)
        if value != "unknown" and value not in contract.presentation_values[dimension]:
            raise OntologyContractError(
                f"presentation.{dimension} contains unknown value: {value}"
            )
