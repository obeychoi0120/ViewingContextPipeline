"""Current content context and receiver-free Actions graphs."""

from __future__ import annotations
from extraction.graph_warnings import TRASH_INFO
from extraction.structured_output import OutputValidationError, validate_graph_structure

GRAPH_SCHEMA_VERSION = "scene-graph/v8"
_RESERVED = {"none", "unknown"}


class SemanticGraphError(ValueError):
    pass


def _informative(value):
    return isinstance(value, str) and value.strip().casefold() not in _RESERVED | {""}


def is_trash_info(graph):
    """No usable information: an all-unknown context, or no named entity and no action."""
    if not any(_informative(value) for value in (graph["medium"], graph["format"], *graph["topics"])):
        return True
    return not any(_informative(entity["name"]) for entity in graph["entities"]) and not any(
        _informative(action["action"]) for action in graph["actions"]
    )


def graph_semantic_warnings(graph):
    try:
        validate_graph_structure(graph)
    except OutputValidationError as exc:
        return exc.tags
    return [TRASH_INFO] if is_trash_info(graph) else []
