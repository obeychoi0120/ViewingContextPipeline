from __future__ import annotations

import json
from typing import Any

from extraction.semantic_graph.taxonomy import (
    AFFECT_AROUSAL,
    AFFECT_VALENCE,
    EVENT_REFERENCE_SLOTS,
    SETTING_CONTEXTS,
)
from extraction.summary_validation import SummaryContractError, parse_summary_sections


GRAPH_FIELDS = (
    "setting_context",
    "entities",
    "events",
    "static_relations",
    "semantic_topics",
    "affect",
)
SUMMARY_SCHEMA_VERSION = "graph-video-summary/v3"


class SemanticGraphError(RuntimeError):
    pass


def graph_semantic_warnings(graph: dict[str, Any]) -> list[str]:
    """Return deterministic structural/reference warnings without rejecting the graph."""
    warnings: list[str] = []
    missing = [field for field in GRAPH_FIELDS if field not in graph]
    unexpected = sorted(str(field) for field in graph if field not in GRAPH_FIELDS)
    if missing:
        warnings.append(f"missing_top_level_fields:{','.join(missing)}")
    if unexpected:
        warnings.append(f"unexpected_top_level_fields:{','.join(unexpected)}")

    setting = graph.get("setting_context")
    if "setting_context" in graph and setting not in SETTING_CONTEXTS:
        warnings.append(f"invalid_setting_context:{setting!r}")

    collections: dict[str, list[Any]] = {}
    for field in ("entities", "events", "static_relations", "semantic_topics"):
        value = graph.get(field)
        if field in graph and not isinstance(value, list):
            warnings.append(f"invalid_field_type:{field}:expected=list")
        collections[field] = value if isinstance(value, list) else []

    entity_ids = _local_ids(collections["entities"], "entities", warnings)
    event_ids = _local_ids(collections["events"], "events", warnings)

    for index, event in enumerate(collections["events"]):
        if not isinstance(event, dict):
            continue
        for slot in EVENT_REFERENCE_SLOTS:
            _warn_unresolved_reference(
                event.get(slot), entity_ids, f"events[{index}].{slot}", warnings
            )
    for index, relation in enumerate(collections["static_relations"]):
        if not isinstance(relation, dict):
            warnings.append(f"invalid_record_type:static_relations[{index}]:expected=object")
            continue
        for slot in ("subject_id", "object_id"):
            _warn_unresolved_reference(
                relation.get(slot),
                entity_ids,
                f"static_relations[{index}].{slot}",
                warnings,
            )
    for index, topic in enumerate(collections["semantic_topics"]):
        if not isinstance(topic, dict):
            warnings.append(f"invalid_record_type:semantic_topics[{index}]:expected=object")
            continue
        _warn_reference_list(
            topic.get("evidence_entity_ids"),
            entity_ids,
            f"semantic_topics[{index}].evidence_entity_ids",
            warnings,
        )
        _warn_reference_list(
            topic.get("evidence_event_ids"),
            event_ids,
            f"semantic_topics[{index}].evidence_event_ids",
            warnings,
        )

    affect = graph.get("affect")
    if "affect" in graph and not isinstance(affect, dict):
        warnings.append("invalid_field_type:affect:expected=object")
    elif isinstance(affect, dict):
        _warn_reference_list(
            affect.get("subject_ids"), entity_ids, "affect.subject_ids", warnings
        )
        if affect.get("valence") not in AFFECT_VALENCE:
            warnings.append(f"invalid_affect_valence:{affect.get('valence')!r}")
        if affect.get("arousal") not in AFFECT_AROUSAL:
            warnings.append(f"invalid_affect_arousal:{affect.get('arousal')!r}")
    return warnings


def _local_ids(
    records: list[Any],
    field: str,
    warnings: list[str],
) -> set[str]:
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            warnings.append(f"invalid_record_type:{field}[{index}]:expected=object")
            continue
        identifier = record.get("local_id")
        if not isinstance(identifier, str) or not identifier.strip():
            warnings.append(f"invalid_local_id:{field}[{index}]")
            continue
        if identifier in identifiers:
            warnings.append(f"duplicate_local_id:{field}:{identifier}")
        identifiers.add(identifier)
    return identifiers


def _warn_unresolved_reference(
    value: Any,
    identifiers: set[str],
    path: str,
    warnings: list[str],
) -> None:
    if value is not None and (
        not isinstance(value, str) or value not in identifiers
    ):
        warnings.append(f"unresolved_reference:{path}={value!r}")


def _warn_reference_list(
    value: Any,
    identifiers: set[str],
    path: str,
    warnings: list[str],
) -> None:
    if not isinstance(value, list):
        warnings.append(f"invalid_field_type:{path}:expected=list")
        return
    for reference in value:
        _warn_unresolved_reference(reference, identifiers, path, warnings)


def graph_summary_prompt(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise SemanticGraphError("graph summary requires scene records")
    ordered = sorted(records, key=lambda row: int(row["scene_idx"]))
    scenes = [
        "\n".join(
            [
                f"Scene {int(record['scene_idx'])} "
                f"(keyframes: {', '.join(f'{value}s' for value in record['keyframes'])}):",
                json.dumps(record["graph"], ensure_ascii=False, sort_keys=True),
            ]
        )
        for record in ordered
    ]
    return template.format(scenes="\n\n".join(scenes))


def validate_summary(text: str) -> dict[str, str]:
    try:
        return parse_summary_sections(text)
    except SummaryContractError as exc:
        raise SemanticGraphError(str(exc)) from exc
