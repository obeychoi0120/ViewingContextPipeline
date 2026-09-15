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
        "context": _array(STRING),
    }
)


class OutputValidationError(ValueError):
    pass


def validate_graph_structure(value, schema=GRAPH_JSON_SCHEMA, path="graph"):
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
    if path == "graph":
        ids = [entity["id"] for entity in value["entities"]]
        if len(ids) != len(set(ids)):
            raise OutputValidationError("graph: duplicate entity IDs")
        for relation in value["relations"]:
            if relation["subject_id"] not in ids or relation["object_id"] not in ids:
                raise OutputValidationError("graph: unresolved relation reference")
