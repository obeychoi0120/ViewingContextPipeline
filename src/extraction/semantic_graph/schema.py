"""Current content context and receiver-free Actions graphs."""

from __future__ import annotations
from extraction.structured_output import OutputValidationError, validate_graph_structure

GRAPH_SCHEMA_VERSION = "scene-graph/v8"


class SemanticGraphError(ValueError):
    pass


def graph_semantic_warnings(graph):
    try:
        validate_graph_structure(graph)
    except OutputValidationError as exc:
        return exc.tags
    return []
