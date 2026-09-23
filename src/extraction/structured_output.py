"""TOBE output contract, independent of the selected prompt filename."""

from __future__ import annotations

import re


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


STRING = {"type": "string", "minLength": 1}
RELATION_GRAPH_JSON_SCHEMA = _object(
    {
        "entities": _array(_object({"id": STRING, "name": STRING, "attributes": _array(STRING)})),
        "relations": _array(
            _object({"subject_id": STRING, "predicate": STRING, "object_id": STRING})
        ),
    }
)
_LEGACY_GRAPH_JSON_SCHEMA = _object(
    {**RELATION_GRAPH_JSON_SCHEMA["properties"], "context": _array(STRING)}
)
MEDIA = ("live_action", "animation", "gameplay", "screen_recording", "mixed", "unknown")
FORMATS = ("demonstration", "explanation", "review", "narrative", "highlights",
           "performance", "interview", "other", "unknown")
REFERENCE = {"type": ["string", "null"], "minLength": 1}
GRAPH_JSON_SCHEMA = _object({
    "medium": {**STRING, "enum": list(MEDIA)},
    "format": {**STRING, "enum": list(FORMATS)},
    "topics": {**_array(STRING), "maxItems": 3},
    "entities": {**_array(_object({
        "id": STRING, "name": STRING,
        "attributes": {**_array(STRING), "maxItems": 2},
    })), "maxItems": 6},
    "actions": {**_array(_object({
        "actor": REFERENCE, "action": STRING, "target": REFERENCE, "tool": REFERENCE,
    })), "maxItems": 4},
})


class OutputValidationError(ValueError):
    pass


def validate_graph_structure(value, schema=None, path="graph"):
    """Check new actions strictly without changing the permissive legacy contract."""
    if schema is None:
        if isinstance(value, dict) and set(value) & {"actions", "medium", "format", "topics"}:
            schema = GRAPH_JSON_SCHEMA
        else:
            schema = (_LEGACY_GRAPH_JSON_SCHEMA if isinstance(value, dict) and "context" in value
                      else RELATION_GRAPH_JSON_SCHEMA)
    _validate_shape(value, schema, path)
    if schema is GRAPH_JSON_SCHEMA:
        _validate_actions(value, path)


def _validate_shape(value, schema, path):
    kind = schema["type"]
    if isinstance(kind, list):
        if value is None and "null" in kind:
            return
        kind = "string"
    expected = {"object": dict, "array": list, "string": str}[kind]
    if not isinstance(value, expected):
        raise OutputValidationError(f"{path}: expected {kind}")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise OutputValidationError(f"{path}: missing or extra fields")
        for key, child in schema["properties"].items():
            _validate_shape(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if len(value) > schema.get("maxItems", len(value)):
            raise OutputValidationError(f"{path}: too many items")
        for index, item in enumerate(value):
            _validate_shape(item, schema["items"], f"{path}[{index}]")
    elif not value.strip():
        raise OutputValidationError(f"{path}: empty string")
    if "enum" in schema and value not in schema["enum"]:
        raise OutputValidationError(f"{path}: invalid value {value!r}")


def _validate_actions(graph, path):
    ids = set()
    for entity in graph["entities"]:
        identifier = entity["id"]
        if (not re.fullmatch(r"[\w.-]+", identifier) or identifier.casefold() == "none"
                or identifier in ids):
            raise OutputValidationError(f"{path}: invalid or duplicate entity ID {identifier!r}")
        ids.add(identifier)
        if any(len(attribute.split()) > 6 for attribute in entity["attributes"]):
            raise OutputValidationError(f"{path}: attribute exceeds six words")
    if any(len(topic.split()) > 4 or topic.casefold() == "none"
           for topic in graph["topics"]):
        raise OutputValidationError(f"{path}: invalid topic; use up to four words or an empty list")
    for action in graph["actions"]:
        if action["actor"] is None and action["target"] is None:
            raise OutputValidationError(f"{path}: action requires an actor or target")
        if any(mark in action["action"] for mark in (" - ", "->", "→", "⇒", ";", "\n")):
            raise OutputValidationError(f"{path}: ambiguous action phrase")
        for field in ("actor", "target", "tool"):
            if action[field] is not None and action[field] not in ids:
                raise OutputValidationError(f"{path}: undeclared {field} {action[field]!r}")
