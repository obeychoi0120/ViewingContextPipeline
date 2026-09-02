from __future__ import annotations

import json
import re
from typing import Any

from extraction.summary_validation import SummaryContractError, parse_summary_sections


GRAPH_SCHEMA_VERSION = "minimal-semantic-v2"
GRAPH_FIELDS = (
    "setting_context",
    "entities",
    "events",
    "semantic_topics",
    "affect",
)
ENTITY_FIELDS = ("local_id", "name", "salience", "function", "count")
EVENT_FIELDS = (
    "local_id",
    "actor_id",
    "action",
    "target_id",
    "instrument_id",
    "location_id",
)
TOPIC_FIELDS = ("label", "evidence_entity_ids", "evidence_event_ids")
AFFECT_FIELDS = ("valence", "arousal")
SETTING_CONTEXTS = (
    "indoor",
    "outdoor_nature",
    "outdoor_urban",
    "transport",
    "unknown",
)
ENTITY_SALIENCE = ("background", "context", "primary", "secondary")
ENTITY_COUNTS = ("one", "few", "many")
AFFECT_VALENCE = ("negative", "neutral", "positive", "unknown")
AFFECT_AROUSAL = ("high", "low", "medium", "unknown")
EVENT_REFERENCE_SLOTS = (
    "actor_id",
    "target_id",
    "instrument_id",
    "location_id",
)
REMOVED_FIELDS = frozenset(
    {
        "affect_cues",
        "category",
        "composition_density",
        "confidence",
        "entity_category",
        "entity_type",
        "ethnicity",
        "fantasy_element",
        "gender",
        "graphic_density",
        "media_form",
        "mood",
        "mood_bin",
        "people_density",
        "relation",
        "role",
        "scene_function",
        "scene_type",
        "score",
        "shot_scale",
        "static_relations",
        "subject_ids",
        "type",
        "visual_style_cues",
    }
)
DISALLOWED_GENERIC_ENTITY_NAMES = frozenset({"object", "thing", "item", "something", "stuff"})
PRIVACY_ENTITY_NAMES = frozenset(
    {
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "kid",
        "guy",
        "lady",
        "gentleman",
        "teenager",
        "toddler",
        "elderly man",
        "elderly woman",
        "old man",
        "old woman",
        "young man",
        "young woman",
        "men",
        "women",
        "boys",
        "girls",
        "children",
        "kids",
        "couple",
        "family",
    }
)
FUNCTION_BLOCKLIST_TOKENS = frozenset(
    {
        "mother",
        "father",
        "wife",
        "husband",
        "girlfriend",
        "boyfriend",
        "friend",
        "family",
        "couple",
        "brother",
        "sister",
        "son",
        "daughter",
        "male",
        "female",
        "man",
        "woman",
        "boy",
        "girl",
        "lady",
        "gentleman",
        "young",
        "old",
        "elderly",
        "teenage",
        "famous",
        "celebrity",
    }
)
SUMMARY_SCHEMA_VERSION = "graph-video-summary/v3"


class SemanticGraphError(RuntimeError):
    pass


def graph_semantic_warnings(graph: dict[str, Any]) -> list[str]:
    """Report minimal-semantic-v2 deviations without modifying the raw graph."""
    warnings: list[str] = []
    _warn_exact_fields(graph, GRAPH_FIELDS, "top_level", warnings)
    _warn_removed_fields(graph, "graph", warnings)

    setting = graph.get("setting_context")
    if "setting_context" in graph and setting not in SETTING_CONTEXTS:
        warnings.append(f"invalid_setting_context:{setting!r}")

    collections: dict[str, list[Any]] = {}
    for field in ("entities", "events", "semantic_topics"):
        value = graph.get(field)
        if field in graph and not isinstance(value, list):
            warnings.append(f"invalid_field_type:{field}:expected=list")
        collections[field] = value if isinstance(value, list) else []

    entities = collections["entities"]
    if not 1 <= len(entities) <= 6:
        warnings.append(f"entity_count_out_of_range:{len(entities)}")
    if len(collections["events"]) > 4:
        warnings.append(f"event_count_exceeds_cap:{len(collections['events'])}")
    if len(collections["semantic_topics"]) > 3:
        warnings.append(f"semantic_topic_count_exceeds_cap:{len(collections['semantic_topics'])}")

    entity_ids = _record_local_ids(entities, "entities", "e", ENTITY_FIELDS, warnings)
    event_ids = _record_local_ids(collections["events"], "events", "ev", EVENT_FIELDS, warnings)

    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        path = f"entities[{index}]"
        name = entity.get("name")
        if not isinstance(name, str) or not name.strip():
            warnings.append(f"invalid_entity_name:{path}")
        else:
            canonical = name.strip()
            if canonical != canonical.lower():
                warnings.append(f"noncanonical_entity_name:{path}={name!r}")
            if canonical.lower() in DISALLOWED_GENERIC_ENTITY_NAMES:
                warnings.append(f"disallowed_generic_entity_name:{path}={name!r}")
            if canonical.lower() in PRIVACY_ENTITY_NAMES:
                warnings.append(f"privacy_entity_name:{path}={name!r}")
        salience = entity.get("salience")
        if salience not in ENTITY_SALIENCE:
            warnings.append(f"invalid_entity_salience:{path}={salience!r}")
        count = entity.get("count")
        if count not in ENTITY_COUNTS:
            warnings.append(f"invalid_entity_count:{path}={count!r}")
        function = entity.get("function")
        if function is not None:
            if not isinstance(function, str) or not function.strip():
                warnings.append(f"invalid_entity_function:{path}")
            else:
                words = re.findall(r"[A-Za-z0-9']+", function.lower())
                if function.strip() != function.strip().lower():
                    warnings.append(f"noncanonical_entity_function:{path}={function!r}")
                if len(words) > 2:
                    warnings.append(f"entity_function_too_long:{path}={function!r}")
                blocked = sorted(set(words) & FUNCTION_BLOCKLIST_TOKENS)
                if blocked:
                    warnings.append(f"blocked_entity_function_tokens:{path}={','.join(blocked)}")

    for index, event in enumerate(collections["events"]):
        if not isinstance(event, dict):
            continue
        path = f"events[{index}]"
        action = event.get("action")
        if not isinstance(action, str) or not action.strip():
            warnings.append(f"invalid_event_action:{path}")
        elif action.strip() != action.strip().lower():
            warnings.append(f"noncanonical_event_action:{path}={action!r}")
        for slot in EVENT_REFERENCE_SLOTS:
            _warn_unresolved_reference(event.get(slot), entity_ids, f"{path}.{slot}", warnings)

    for index, topic in enumerate(collections["semantic_topics"]):
        if not isinstance(topic, dict):
            warnings.append(
                f"invalid_record_type:semantic_topics[{index}]:expected=object"
            )
            continue
        path = f"semantic_topics[{index}]"
        _warn_exact_fields(topic, TOPIC_FIELDS, path, warnings)
        label = topic.get("label")
        if not isinstance(label, str) or not label.strip():
            warnings.append(f"invalid_semantic_topic_label:{path}")
        elif len(re.findall(r"[A-Za-z0-9']+", label)) > 4:
            warnings.append(f"semantic_topic_label_too_long:{path}={label!r}")
        entity_evidence = topic.get("evidence_entity_ids")
        event_evidence = topic.get("evidence_event_ids")
        _warn_reference_list(
            entity_evidence,
            entity_ids,
            f"{path}.evidence_entity_ids",
            warnings,
        )
        _warn_reference_list(
            event_evidence,
            event_ids,
            f"{path}.evidence_event_ids",
            warnings,
        )
        if entity_evidence == [] and event_evidence == []:
            warnings.append(f"semantic_topic_without_evidence:{path}")

    affect = graph.get("affect")
    if "affect" in graph and not isinstance(affect, dict):
        warnings.append("invalid_field_type:affect:expected=object")
    elif isinstance(affect, dict):
        _warn_exact_fields(affect, AFFECT_FIELDS, "affect", warnings)
        if affect.get("valence") not in AFFECT_VALENCE:
            warnings.append(f"invalid_affect_valence:{affect.get('valence')!r}")
        if affect.get("arousal") not in AFFECT_AROUSAL:
            warnings.append(f"invalid_affect_arousal:{affect.get('arousal')!r}")
    return warnings


def _warn_exact_fields(
    value: dict[str, Any],
    expected: tuple[str, ...],
    path: str,
    warnings: list[str],
) -> None:
    missing = [field for field in expected if field not in value]
    unexpected = sorted(str(field) for field in value if field not in expected)
    if missing:
        warnings.append(f"missing_fields:{path}:{','.join(missing)}")
    if unexpected:
        warnings.append(f"unexpected_fields:{path}:{','.join(unexpected)}")


def _warn_removed_fields(value: Any, path: str, warnings: list[str]) -> None:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = value[key]
            child_path = f"{path}.{key}"
            if key in REMOVED_FIELDS:
                warnings.append(f"removed_field:{child_path}")
            _warn_removed_fields(child, child_path, warnings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _warn_removed_fields(child, f"{path}[{index}]", warnings)


def _record_local_ids(
    records: list[Any],
    field: str,
    prefix: str,
    expected_fields: tuple[str, ...],
    warnings: list[str],
) -> set[str]:
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        path = f"{field}[{index}]"
        if not isinstance(record, dict):
            warnings.append(f"invalid_record_type:{path}:expected=object")
            continue
        _warn_exact_fields(record, expected_fields, path, warnings)
        identifier = record.get("local_id")
        if not isinstance(identifier, str) or not identifier.strip():
            warnings.append(f"invalid_local_id:{path}")
            continue
        expected_id = f"{prefix}{index + 1}"
        if identifier != expected_id:
            warnings.append(
                f"nonsequential_local_id:{path}:expected={expected_id},actual={identifier}"
            )
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
    if value is not None and (not isinstance(value, str) or value not in identifiers):
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
