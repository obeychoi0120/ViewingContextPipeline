"""Recommendation experiments and independent, title-free extraction sources."""

from dataclasses import dataclass

CONCAT_POLICY = "title-summary-double-newline/v1"
EXPERIMENT_CONFIG_VERSION = "v6"
RECOMMENDATION_ARM_CONTRACT = "gnn-attn-aggregation/v1"


@dataclass(frozen=True)
class Arm:
    name: str
    representation: str
    model: str | None = None
    uses_title: bool = False
    scene_arm: str | None = None
    aggregation: str = "mean"
    role_fusion: bool = False
    # Optional gnn_attn stages (validation.graph_context.EXTENSION_MODELS); with "segments",
    # the scene_segments sidecar variant of the source bundle ("true" or "shuffled").
    extensions: tuple[str, ...] = ()
    segments: str = "true"


ARMS = (
    Arm("base", "metadata", uses_title=True),
    Arm("d0", "description", "qwen3vl", True, "desc_qwen3vl"),
    Arm("d1", "description", "qwen3vl", True, "desc_qwen3vl", "gnn_attn"),
    Arm("g0", "graph", "qwen3vl", True, "graph_qwen3vl", role_fusion=True),
    Arm("g1", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", True),
    # G1 plus gnn_attn extensions: video node, activity segments, both.
    Arm("g2", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", True, ("video_node",)),
    Arm("g3", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", True, ("segments",)),
    Arm("g4", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", True, ("segments", "video_node")),
    # Ablations: G1 without Fusion, corrupted role links, self-only exchange; G3 with shuffled segments.
    Arm("g1a", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", False),
    Arm("g1b", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", True),
    Arm("g1c", "graph", "qwen3vl", True, "graph_qwen3vl", "self_only", True),
    Arm("g3a", "graph", "qwen3vl", True, "graph_qwen3vl", "gnn_attn", True, ("segments",), "shuffled"),
)
SOURCES = tuple(
    Arm(name, kind, model, False, name)
    for name, kind, model in (
        ("graph_qwen3vl", "graph", "qwen3vl"),
        ("desc_qwen3vl", "description", "qwen3vl"),
        ("graph_qwen3.5", "graph", "qwen3.5"),
        ("desc_qwen3.5", "description", "qwen3.5"),
        ("graph_gemini", "graph", "gemini"),
        ("desc_gemini", "description", "gemini"),
    )
)


def arm_contract(config):
    return RECOMMENDATION_ARM_CONTRACT


def registry(config):
    return {arm.name: arm for arm in ARMS}


def feature_source(arm):
    return arm.scene_arm or "metadata"


def target_set(config, target):
    return "_".join(select_arms(config, target))


def active_arms(config):
    if "arms" in config.get("protocol", {}):
        raise ValueError("protocol.arms is not supported; select --target explicitly")
    return list(registry(config))


def select_arms(config, target=None):
    registered = registry(config)
    names = active_arms(config) if target is None else target
    if not names or len(names) != len(set(names)) or set(names) - registered.keys():
        raise ValueError(f"--target must select from {', '.join(registered)}")
    return {name: arm for name, arm in registered.items() if name in names}


def generation_registry(config):
    sources = {arm.name: arm for arm in SOURCES}
    if "minicpm" in config.get("models", {}):
        sources["graph_minicpm"] = Arm("graph_minicpm", "graph", "minicpm", False, "graph_minicpm")
    return sources


def generated_arm(config, representation, model):
    if model not in {"gemini", "qwen3vl", "qwen3.5", "minicpm"}:
        raise ValueError("model/source must be qwen3vl, qwen3.5, gemini or minicpm")
    source = next(
        (arm for arm in generation_registry(config).values() if arm.representation == representation and arm.model == model),
        None,
    )
    if source is None:
        raise ValueError(f"unsupported generation source: {representation}/{model}")
    return source


def resolve_generation_arm(config, name, representation, model, *, summary=False):
    arm = generation_registry(config).get(name)
    if arm is None:
        source = registry(config).get(name)
        hint = f"; use source --arm {source.scene_arm}" if source and source.scene_arm else ""
        raise ValueError(f"invalid generation source: {name}{hint}")
    if arm.representation != representation:
        raise ValueError(f"invalid {representation} generation arm: {name}")
    if summary and arm.model == "minicpm":
        raise ValueError("MiniCPM supports only Graph scene extraction, not summarization")
    if not summary and arm.model != model:
        raise ValueError(
            "Scene arm must identify the requested extraction model and representation"
        )
    return arm
