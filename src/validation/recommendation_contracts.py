from __future__ import annotations
from arm_registry import registry, select_arms, active_arms

ARCHITECTURE_VERSION = "sasrec-content-v3"
# Both versions use the same persisted event/rank and rolling training contracts.
DIAGNOSIS_ARCHITECTURE_VERSIONS = frozenset({"sasrec-content-v2", "sasrec-content-v3"})
DEFAULT_PROTOCOL = {
    "protocol": {
        "arms": ["meta", "graph_qwen", "graph_gemini", "desc_qwen", "desc_gemini",
                 "graph_meta_qwen", "graph_meta_gemini", "desc_meta_qwen", "desc_meta_gemini"],
    }
}
RECOMMENDATION_ARMS = {name: name for name in registry(DEFAULT_PROTOCOL)}
TARGET_SOURCES = dict(RECOMMENDATION_ARMS)


def resolve_target_arms(target=None, *, config=None):
    configured = config or DEFAULT_PROTOCOL
    if target is not None and set(target) - set(active_arms(configured)):
        raise ValueError("--target must be a subset of protocol.arms")
    return {name: name for name in select_arms(configured, target)}


def target_scope(arms, *, config=None):
    return {
        "target_sources": list(arms),
        "selected_arms": list(arms),
        "excluded_arms": [
            name for name in registry(config or DEFAULT_PROTOCOL) if name not in arms
        ],
    }

# Bump for changes to selection/refit, loss, masking, ranking, or metric semantics.
TRAINING_IMPLEMENTATION_VERSION = "shared-scenes-training-evaluation/v3"
