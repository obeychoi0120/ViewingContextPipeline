from __future__ import annotations

from pathlib import Path
from typing import Any

from extraction.backends import VLMBackend
from extraction.evidence import build_scene_evidence, load_images
from extraction.json_repair import extract_json
from extraction.semantic_graph.taxonomy import (
    AFFECT_AROUSAL,
    AFFECT_VALENCE,
    DISALLOWED_GENERIC_ENTITY_NAMES,
    ENTITY_ROLES,
    EVENT_REFERENCE_SLOTS,
    MAX_ENTITIES,
    MAX_EVENTS,
    MAX_SEMANTIC_TOPICS,
    MAX_STATIC_RELATIONS,
    MAX_TOPIC_WORDS,
    SCENE_SCHEMA_VERSION,
    SETTING_CONTEXTS,
    STATIC_RELATION_TYPES,
)


GRAPH_SUMMARY_SCHEMA = "graph-video-summary/v1"
TOP_LEVEL_FIELDS = {
    "setting_context",
    "entities",
    "events",
    "static_relations",
    "semantic_topics",
    "affect",
}


class SemanticGraphError(RuntimeError):
    pass


def parse_graph_output(text: str) -> dict[str, Any]:
    payload = extract_json(text)
    if payload is None:
        raise SemanticGraphError("VLM output does not contain a JSON object")
    return _validate_payload(payload)


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
        raise SemanticGraphError(
            f"graph must contain exactly {sorted(TOP_LEVEL_FIELDS)}"
        )

    setting = _enum(payload["setting_context"], SETTING_CONTEXTS, "setting_context")
    entities = _entities(payload["entities"])
    entity_ids = {row["local_id"] for row in entities}
    events = _events(payload["events"], entity_ids)
    event_ids = {row["local_id"] for row in events}
    static_relations = _static_relations(
        payload["static_relations"], entity_ids
    )
    topics = _topics(payload["semantic_topics"], entity_ids, event_ids)
    affect = _affect(payload["affect"], entity_ids)

    if not entities:
        if events or static_relations or topics:
            raise SemanticGraphError(
                "an empty entity list requires empty events, relations, and topics"
            )
        if affect != {
            "subject_ids": [],
            "valence": "unknown",
            "arousal": "unknown",
        }:
            raise SemanticGraphError(
                "an empty entity list requires unknown/unknown affect"
            )

    return {
        "setting_context": setting,
        "entities": entities,
        "events": events,
        "static_relations": static_relations,
        "semantic_topics": topics,
        "affect": affect,
    }


def extract_scene_graphs(
    *,
    content_id: str,
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
    timestamp_json_path: str | Path,
    backend: VLMBackend,
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in build_scene_evidence(scenes, frames_dir, timestamp_json_path):
        image_paths = scene["image_paths"]
        if not scene["keyframes"] or len(image_paths) != len(scene["keyframes"]):
            raise SemanticGraphError(
                f"scene {scene['scene_idx']} has {len(image_paths)} of "
                f"{len(scene['keyframes'])} keyframes"
            )
        graph = parse_graph_output(
            backend.generate(load_images(image_paths), prompt, max_new_tokens)
        )
        records.append(
            {
                "schema_version": SCENE_SCHEMA_VERSION,
                "content_id": content_id,
                "scene_idx": scene["scene_idx"],
                "scene_start_seconds": scene["scene_start_seconds"],
                "scene_end_seconds": scene["scene_end_seconds"],
                "keyframes": scene["keyframes"],
                "image_paths": image_paths,
                "graph": graph,
            }
        )
    if not records:
        raise SemanticGraphError("video has no scenes")
    return records


def graph_summary_prompt(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise SemanticGraphError("graph summary requires scene records")
    try:
        scene_indices = [int(record["scene_idx"]) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise SemanticGraphError("graph summary contains an invalid scene index") from exc
    if len(scene_indices) != len(set(scene_indices)):
        raise SemanticGraphError("graph summary contains duplicate scene indices")
    scenes = [_format_scene(record) for record in sorted(
        records, key=lambda row: (int(row["scene_idx"]), list(row["keyframes"]))
    )]
    return template.format(scenes="\n\n".join(scenes))


def validate_summary(text: str) -> str:
    summary = str(text or "").strip()
    words = len(summary.split())
    if not 1 <= words <= 150:
        raise SemanticGraphError(
            f"video summary must contain 1-150 words; got {words}"
        )
    return summary


def _entities(value: Any) -> list[dict[str, str]]:
    rows = _array(value, MAX_ENTITIES, "entities")
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        _fields(row, {"local_id", "name", "role"}, f"entity {index}")
        local_id = _exact_id(row["local_id"], f"e{index}", f"entity {index}")
        name = _text(row["name"], f"entity {index} name").lower()
        if name in DISALLOWED_GENERIC_ENTITY_NAMES:
            raise SemanticGraphError(f"entity {index} uses generic name {name!r}")
        normalized.append(
            {
                "local_id": local_id,
                "name": name,
                "role": _enum(row["role"], ENTITY_ROLES, f"entity {index} role"),
            }
        )
    return normalized


def _events(value: Any, entity_ids: set[str]) -> list[dict[str, Any]]:
    rows = _array(value, MAX_EVENTS, "events")
    fields = {"local_id", "action", *EVENT_REFERENCE_SLOTS}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        _fields(row, fields, f"event {index}")
        refs = {
            slot: _reference(row[slot], entity_ids, f"event {index} {slot}")
            for slot in EVENT_REFERENCE_SLOTS
        }
        if not any(refs.values()):
            raise SemanticGraphError(
                f"event {index} must reference at least one entity"
            )
        normalized.append(
            {
                "local_id": _exact_id(
                    row["local_id"], f"ev{index}", f"event {index}"
                ),
                "actor_id": refs["actor_id"],
                "action": _text(row["action"], f"event {index} action").lower(),
                "target_id": refs["target_id"],
                "instrument_id": refs["instrument_id"],
                "location_id": refs["location_id"],
            }
        )
    return normalized


def _static_relations(value: Any, entity_ids: set[str]) -> list[dict[str, str]]:
    rows = _array(value, MAX_STATIC_RELATIONS, "static_relations")
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        _fields(
            row,
            {"subject_id", "relation", "object_id"},
            f"static relation {index}",
        )
        normalized.append(
            {
                "subject_id": _required_reference(
                    row["subject_id"], entity_ids, f"static relation {index} subject_id"
                ),
                "relation": _enum(
                    row["relation"], STATIC_RELATION_TYPES, f"static relation {index} relation"
                ),
                "object_id": _required_reference(
                    row["object_id"], entity_ids, f"static relation {index} object_id"
                ),
            }
        )
    return normalized


def _topics(
    value: Any, entity_ids: set[str], event_ids: set[str]
) -> list[dict[str, Any]]:
    rows = _array(value, MAX_SEMANTIC_TOPICS, "semantic_topics")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        _fields(
            row,
            {"label", "evidence_entity_ids", "evidence_event_ids"},
            f"topic {index}",
        )
        label = _text(row["label"], f"topic {index} label").lower()
        if len(label.split()) > MAX_TOPIC_WORDS:
            raise SemanticGraphError(
                f"topic {index} label exceeds {MAX_TOPIC_WORDS} words"
            )
        entity_refs = _reference_array(
            row["evidence_entity_ids"], entity_ids, f"topic {index} entity evidence"
        )
        event_refs = _reference_array(
            row["evidence_event_ids"], event_ids, f"topic {index} event evidence"
        )
        if not entity_refs and not event_refs:
            raise SemanticGraphError(f"topic {index} must cite evidence")
        normalized.append(
            {
                "label": label,
                "evidence_entity_ids": entity_refs,
                "evidence_event_ids": event_refs,
            }
        )
    return normalized


def _affect(value: Any, entity_ids: set[str]) -> dict[str, Any]:
    _fields(value, {"subject_ids", "valence", "arousal"}, "affect")
    subject_ids = _reference_array(value["subject_ids"], entity_ids, "affect subjects")
    valence = _enum(value["valence"], AFFECT_VALENCE, "affect valence")
    arousal = _enum(value["arousal"], AFFECT_AROUSAL, "affect arousal")
    if not subject_ids and (valence != "unknown" or arousal != "unknown"):
        raise SemanticGraphError(
            "affect without subjects must use unknown valence and arousal"
        )
    return {"subject_ids": subject_ids, "valence": valence, "arousal": arousal}


def _format_scene(record: dict[str, Any]) -> str:
    required = {
        "schema_version",
        "scene_idx",
        "scene_start_seconds",
        "scene_end_seconds",
        "keyframes",
        "image_paths",
        "graph",
    }
    if not required.issubset(record) or record.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise SemanticGraphError("graph summary received an invalid scene record")
    keyframes = record["keyframes"]
    image_paths = record["image_paths"]
    if (
        not isinstance(keyframes, list)
        or not keyframes
        or keyframes != sorted(keyframes)
        or not isinstance(image_paths, list)
        or len(image_paths) != len(keyframes)
    ):
        raise SemanticGraphError("graph summary received incomplete scene evidence")
    graph = record.get("graph")
    if not isinstance(graph, dict):
        raise SemanticGraphError("graph summary received an invalid graph")
    graph = _validate_payload(graph)
    names = {row["local_id"]: row["name"] for row in graph["entities"]}
    lines = [
        f"Scene {int(record['scene_idx'])} "
        f"({record['scene_start_seconds']}-{record['scene_end_seconds']}s):",
        f"Setting: {graph['setting_context']}",
    ]
    if graph["entities"]:
        lines.append(
            "Entities: "
            + "; ".join(
                f"{row['local_id']}={row['name']} ({row['role']})"
                for row in graph["entities"]
            )
        )
    for event in graph["events"]:
        refs = [
            f"{slot.removesuffix('_id')}={names[value]}"
            for slot in EVENT_REFERENCE_SLOTS
            if (value := event[slot]) is not None
        ]
        lines.append(
            f"Event {event['local_id']}: {event['action']} ({', '.join(refs)})"
        )
    for relation in graph["static_relations"]:
        lines.append(
            f"Wearing: {names[relation['subject_id']]} wears "
            f"{names[relation['object_id']]}"
        )
    for topic in graph["semantic_topics"]:
        lines.append(f"Topic: {topic['label']}")
    affect = graph["affect"]
    if affect["subject_ids"]:
        subjects = ", ".join(names[item] for item in affect["subject_ids"])
        lines.append(
            f"Visible affect: {subjects}; {affect['valence']}, {affect['arousal']}"
        )
    return "\n".join(lines)


def _array(value: Any, maximum: int, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SemanticGraphError(f"{label} must be an array")
    if len(value) > maximum:
        raise SemanticGraphError(f"{label} exceeds maximum {maximum}")
    return value


def _fields(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise SemanticGraphError(f"{label} must contain exactly {sorted(fields)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticGraphError(f"{label} must be a non-empty string")
    return " ".join(value.strip().split())


def _enum(value: Any, allowed: tuple[str, ...], label: str) -> str:
    normalized = _text(value, label).lower()
    by_lower = {item.lower(): item for item in allowed}
    if normalized not in by_lower:
        raise SemanticGraphError(f"{label} must be one of {list(allowed)}")
    return by_lower[normalized]


def _exact_id(value: Any, expected: str, label: str) -> str:
    normalized = _text(value, f"{label} local_id")
    if normalized != expected:
        raise SemanticGraphError(f"{label} local_id must be {expected!r}")
    return normalized


def _reference(value: Any, allowed: set[str], label: str) -> str | None:
    if value is None:
        return None
    return _required_reference(value, allowed, label)


def _required_reference(value: Any, allowed: set[str], label: str) -> str:
    normalized = _text(value, label)
    if normalized not in allowed:
        raise SemanticGraphError(f"{label} has dangling reference {normalized!r}")
    return normalized


def _reference_array(value: Any, allowed: set[str], label: str) -> list[str]:
    if not isinstance(value, list):
        raise SemanticGraphError(f"{label} must be an array")
    normalized = [_required_reference(item, allowed, label) for item in value]
    if len(normalized) != len(set(normalized)):
        raise SemanticGraphError(f"{label} contains duplicate references")
    return normalized
