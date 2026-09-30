"""Explicit graph execution scope and mode-specific artifact layout."""

from dataclasses import dataclass

from pipeline_runtime import RunContext

GRAPH_ARMS = frozenset({"meta", "graph_qwen", "graph_qwen_meta", "graph_gemini_meta"})
GRAPH_ARCHITECTURE = "sasrec-role-graph/v3"
GRAPH_MODEL = {
    "layers": 1,
    "hidden_dim": 128,
    "scene_dim": 384,
    "title_dim": 128,
    "attention_dim": 128,
    "fusion": "title_then_video_concat_residual_mlp",
    "scene_readout": "entity_action_context_concat_layernorm",
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
    if aggregation not in (None, "mean", "attention"):
        raise ValueError("scene aggregation must be mean or attention")
    return GraphContext(context.root, context.run_id, context.config, context.run_root, aggregation)


def validate_mode(mode, aggregation, step, targets):
    if mode not in ("text", "graph"):
        raise ValueError("--representation-mode text|graph is required")
    if mode == "graph" and set(targets or ()) - GRAPH_ARMS:
        raise ValueError("graph mode supports: " + ", ".join(sorted(GRAPH_ARMS)))
    needs_pool = mode == "graph" and step != "embed-representations"
    if needs_pool and aggregation not in ("mean", "attention"):
        raise ValueError(
            "--scene-aggregation mean|attention is required for graph recommendation/diagnosis"
        )
    if not needs_pool and aggregation is not None:
        raise ValueError("--scene-aggregation is only used for graph recommendation/diagnosis")
