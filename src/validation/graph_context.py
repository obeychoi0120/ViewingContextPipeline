"""Explicit graph execution scope and mode-specific artifact layout."""

from dataclasses import dataclass

from pipeline_runtime import RunContext

GRAPH_ARMS = frozenset({"meta", "graph_qwen", "graph_qwen_meta", "graph_gemini_meta"})
GRAPH_ARCHITECTURE = "sasrec-role-graph/v7"
GRAPH_MODEL = {
    "hidden_dim": 128,
    "scene_dim": 512,
    "title_dim": 128,
    "fusion": "role_gather_scene_residual_mlp_then_mean",
    "scene_readout": "title_entity_action_context_concat",
    "projections": "separate_title_entity_action_context",
    "roles": ["actor", "target", "tool", "location"],
    "role_fusion": "action_plus_relu_linear_640_to_128",
    "scene_mlp": {"hidden_dim": 512, "activation": "relu", "normalization": "none"},
    "aggregation": "mean",
    "input_dim": 1024,
    "state_lookup": "embedding",
}


@dataclass(frozen=True)
class GraphContext(RunContext):
    scene_aggregation: str | None = None

    @property
    def representations_dir(self):
        return self.run_root / "validation" / "representations" / "graph"

    @property
    def recommendations_dir(self):
        return self.run_root / "validation" / "recommendations" / "graph" / self.scene_aggregation

    @property
    def diagnosis_path(self):
        return (
            self.run_root
            / "validation"
            / "diagnosis"
            / f"graph_{self.scene_aggregation}_diagnosis.json"
        )


def is_graph(context):
    return isinstance(context, GraphContext)


def graph_context(context, aggregation=None):
    if aggregation not in (None, "mean"):
        raise ValueError("scene aggregation supports only mean")
    return GraphContext(context.root, context.run_id, context.config, context.run_root, aggregation)


def validate_mode(mode, aggregation, step, targets):
    if mode not in ("text", "graph"):
        raise ValueError("--representation-mode text|graph is required")
    if mode == "graph" and set(targets or ()) - GRAPH_ARMS:
        raise ValueError("graph mode supports: " + ", ".join(sorted(GRAPH_ARMS)))
    needs_pool = mode == "graph" and step != "embed-representations"
    if needs_pool and aggregation != "mean":
        raise ValueError(
            "--scene-aggregation mean is required; attention is unsupported by model v7"
        )
    if not needs_pool and aggregation is not None:
        raise ValueError("--scene-aggregation is only used for graph recommendation/diagnosis")
