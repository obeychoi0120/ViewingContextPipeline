from __future__ import annotations
from arm_registry import EXPERIMENT_CONFIG_VERSION, registry, select_arms

ARCHITECTURE_VERSION = "sasrec-content-v8"
# Current text and direct-graph architectures.
DIAGNOSIS_ARCHITECTURE_VERSIONS = frozenset({
    "sasrec-content-v8", "sasrec-role-graph/v9", "sasrec-role-graph/v10"
})
ITEM_MODEL = {
    "baseline_dim": 512,
    "title_dim": 128,
    "video_dim": 384,
    "normalization": "none",
    "item_mlp": True,
    "item_mlp_activation": "relu",
    "item_mlp_hidden_dim": 512,
    "item_mlp_linear_layers": 1,
    "item_mlp_formula": "x_plus_relu_linear_x",
    "input_dim": 1024,
}
DEFAULT_PROTOCOL = {"experiment_config_version": EXPERIMENT_CONFIG_VERSION}
RECOMMENDATION_ARMS = {name: name for name in registry(DEFAULT_PROTOCOL)}
TARGET_SOURCES = dict(RECOMMENDATION_ARMS)


def resolve_target_arms(target=None, *, config=None):
    configured = config or DEFAULT_PROTOCOL
    if target is None:
        raise ValueError("--target is required; select arms explicitly")
    return {name: name for name in select_arms(configured, target)}


def target_scope(arms, *, config=None):
    return {
        "target_sources": list(arms),
        "selected_arms": list(arms),
        "excluded_arms": [
            name for name in registry(config or DEFAULT_PROTOCOL) if name not in arms
        ],
    }


# Bump for changes to partition/checkpoint selection, loss, masking, ranking, or metric semantics.
TRAINING_IMPLEMENTATION_VERSION = "shared-scenes-leave-one-out-training/v1"
