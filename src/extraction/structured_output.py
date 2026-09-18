"""TOBE output contract, independent of the selected prompt filename."""

from __future__ import annotations


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
GRAPH_JSON_SCHEMA = _object(
    {
        "entities": _array(_object({"id": STRING, "name": STRING, "attributes": _array(STRING)})),
        "relations": _array(
            _object({"subject_id": STRING, "predicate": STRING, "object_id": STRING})
        ),
    }
)
_LEGACY_GRAPH_JSON_SCHEMA = _object(
    {**GRAPH_JSON_SCHEMA["properties"], "context": _array(STRING)}
)


class OutputValidationError(ValueError):
    pass


def validate_graph_structure(value, schema=GRAPH_JSON_SCHEMA, path="graph"):
    """Validate JSON shape only; preserve duplicate IDs and unresolved references."""
    if schema is GRAPH_JSON_SCHEMA and isinstance(value, dict) and "context" in value:
        schema = _LEGACY_GRAPH_JSON_SCHEMA
    kind = schema["type"]
    expected = {"object": dict, "array": list, "string": str}[kind]
    if not isinstance(value, expected):
        raise OutputValidationError(f"{path}: expected {kind}")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise OutputValidationError(f"{path}: missing or extra fields")
        for key, child in schema["properties"].items():
            validate_graph_structure(value[key], child, f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            validate_graph_structure(item, schema["items"], f"{path}[{index}]")
    elif not value.strip():
        raise OutputValidationError(f"{path}: empty string")
