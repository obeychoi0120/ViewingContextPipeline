"""Portable output contracts; vLLM is imported only by the GPU backend."""
from __future__ import annotations

import json

from extraction.semantic_graph.schema import (
    AFFECT_AROUSAL, AFFECT_VALENCE, ENTITY_COUNTS, ENTITY_SALIENCE, SETTING_CONTEXTS,
)
from extraction.summary_validation import SUMMARY_SECTIONS


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


STRING = {"type": "string"}
NULLABLE_STRING = {"type": ["string", "null"]}
GRAPH_JSON_SCHEMA = _object({
    "setting_context": {"type": "string", "enum": list(SETTING_CONTEXTS)},
    "entities": _array(_object({
        "local_id": STRING, "name": STRING,
        "salience": {"type": "string", "enum": list(ENTITY_SALIENCE)},
        "function": NULLABLE_STRING,
        "count": {"type": "string", "enum": list(ENTITY_COUNTS)},
    })),
    "events": _array(_object({
        "local_id": STRING, "actor_id": NULLABLE_STRING, "action": STRING,
        "target_id": NULLABLE_STRING, "instrument_id": NULLABLE_STRING,
        "location_id": NULLABLE_STRING,
    })),
    "semantic_topics": _array(_object({
        "label": STRING, "evidence_entity_ids": _array(STRING),
        "evidence_event_ids": _array(STRING),
    })),
    "affect": _object({
        "valence": {"type": "string", "enum": list(AFFECT_VALENCE)},
        "arousal": {"type": "string", "enum": list(AFFECT_AROUSAL)},
    }),
})

# Empty values are permitted; the semantic validator rejects seven empty fields.
SUMMARY_GRAMMAR = 'root ::= ' + ' "\\n" '.join(
    f'{json.dumps(name + ":")} value' for name in SUMMARY_SECTIONS
) + '\nvalue ::= [^\\r\\n]*\n'


class OutputValidationError(ValueError):
    pass


def validate_graph_structure(value, schema=GRAPH_JSON_SCHEMA, path="graph"):
    """Validate the small JSON Schema subset above without coercion or semantic repair."""
    kind = schema["type"]
    if isinstance(kind, list):
        if value is None and "null" in kind:
            return
        kind = "string"
    expected = {"object": dict, "array": list, "string": str}[kind]
    if not isinstance(value, expected):
        raise OutputValidationError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise OutputValidationError(f"{path}: value outside allowed enum")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise OutputValidationError(f"{path}: missing or extra fields")
        for key, child in schema["properties"].items():
            validate_graph_structure(value[key], child, f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            validate_graph_structure(item, schema["items"], f"{path}[{index}]")
