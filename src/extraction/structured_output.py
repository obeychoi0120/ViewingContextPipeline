"""TOBE output contract, independent of the selected prompt filename."""

from __future__ import annotations

import re

from extraction.graph_warnings import (
    MISSING_REQUIRED, INVALID_ACTION_SYNTAX, INVALID_REFERENCE, DUPLICATE_ENTITY_ID, PARSE_ERROR,
)


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
    "medium": STRING,
    "format": STRING,
    "topics": _array(STRING),
    "entities": _array(_object({
        "id": STRING, "name": STRING,
        "attributes": _array(STRING),
    })),
    "actions": _array(_object({
        "actor": REFERENCE, "action": STRING, "target": REFERENCE, "tool": REFERENCE,
    })),
})


class OutputValidationError(ValueError):
    def __init__(self, message, tags=(PARSE_ERROR,)):
        super().__init__(message)
        self.tags = list(dict.fromkeys(tags))


def validate_graph_structure(value, schema=None, path="graph"):
    """Enforce structural integrity; selection budgets and enums are prompt guidance."""
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
        if set(schema["properties"]) - set(value):
            tag = INVALID_ACTION_SYNTAX if ".actions[" in path else MISSING_REQUIRED
            raise OutputValidationError(f"{path}: missing required fields", (tag,))
        if set(value) - set(schema["properties"]):
            raise OutputValidationError(f"{path}: extra fields")
        for key, child in schema["properties"].items():
            _validate_shape(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if len(value) > schema.get("maxItems", len(value)):
            raise OutputValidationError(f"{path}: too many items")
        for index, item in enumerate(value):
            _validate_shape(item, schema["items"], f"{path}[{index}]")
    elif not value.strip():
        raise OutputValidationError(f"{path}: empty string",
                                    (INVALID_ACTION_SYNTAX if ".actions[" in path else PARSE_ERROR,))
    if "enum" in schema and value not in schema["enum"]:
        raise OutputValidationError(f"{path}: invalid value {value!r}")


def _validate_actions(graph, path):
    ids, tags, messages = set(), [], []

    def report(tag, message):
        tags.append(tag)
        messages.append(message)

    for entity in graph["entities"]:
        identifier = entity["id"]
        if not re.fullmatch(r"[\w.-]+", identifier) or identifier.casefold() == "none":
            report(PARSE_ERROR, f"invalid entity ID {identifier!r}")
        if identifier in ids:
            report(DUPLICATE_ENTITY_ID, f"duplicate entity ID {identifier!r}")
        ids.add(identifier)
    for action in graph["actions"]:
        if action["actor"] is None and action["target"] is None:
            report(INVALID_ACTION_SYNTAX, "action requires an actor or target")
        if any(mark in action["action"] for mark in (" - ", "->", "→", "⇒", ";", "\n")):
            report(INVALID_ACTION_SYNTAX, "ambiguous action phrase")
        for field in ("actor", "target", "tool"):
            if action[field] is not None and action[field] not in ids:
                report(INVALID_REFERENCE, f"undeclared {field} {action[field]!r}")
    if tags:
        raise OutputValidationError(f"{path}: " + "; ".join(messages), tags)
