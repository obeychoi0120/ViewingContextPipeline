"""Open vocabulary scene entities, directed relations, and contextual observations."""

from __future__ import annotations
import json
from extraction.structured_output import OutputValidationError, validate_graph_structure
from extraction.summary_validation import (
    SUMMARY_SCHEMA_VERSION as SUMMARY_SCHEMA_VERSION,
    validate_summary as validate_summary,
)

GRAPH_SCHEMA_VERSION = "scene-graph/v3"


class SemanticGraphError(ValueError):
    pass


def graph_semantic_warnings(graph):
    try:
        validate_graph_structure(graph)
    except OutputValidationError as exc:
        return [str(exc)]
    return []


def graph_summary_prompt(template, records):
    if not records:
        raise SemanticGraphError("graph summary requires scene records")
    observations = []
    for record in records:
        graph = record.get("graph", record.get("raw_response"))
        if graph is None:
            raise SemanticGraphError("missing graph observation")
        observations.append({"scene": record["scene_idx"], "observation": graph})
    return template.replace("{scenes}", json.dumps(observations, ensure_ascii=False))
