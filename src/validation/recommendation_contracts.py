from __future__ import annotations
from arm_registry import registry, select_arms

ARCHITECTURE_VERSION = "sasrec-content-v3"
DEFAULT_PROTOCOL = {
    "protocol": {
        "arms": ["desc_gemini", "desc_qwen", "graph_gemini", "graph_qwen", "metadata"],
    }
}
RECOMMENDATION_ARMS = {name: name for name in registry(DEFAULT_PROTOCOL)}
TARGET_SOURCES = dict(RECOMMENDATION_ARMS)


def resolve_target_arms(target=None, *, config=None):
    return {name: name for name in select_arms(config or DEFAULT_PROTOCOL, target)}


def target_scope(arms, *, config=None):
    return {
        "target_sources": list(arms),
        "selected_arms": list(arms),
        "excluded_arms": [
            name for name in registry(config or DEFAULT_PROTOCOL) if name not in arms
        ],
    }
