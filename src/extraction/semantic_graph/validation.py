from __future__ import annotations

from typing import Any

from extraction.semantic_graph.taxonomy import ENTITY_GUIDANCE_MAX


GRAPH_SOFT_VALIDATION_VERSION = "semantic-graph-soft-validation/v1"


def graph_soft_warnings(graph: dict[str, Any]) -> list[str]:
    """Report advisory contract deviations without changing model output."""
    entities = graph.get("entities")
    if not isinstance(entities, list) or len(entities) <= ENTITY_GUIDANCE_MAX:
        return []
    return [
        "entity_guidance_exceeded: "
        f"observed={len(entities)} guidance_max={ENTITY_GUIDANCE_MAX}"
    ]
