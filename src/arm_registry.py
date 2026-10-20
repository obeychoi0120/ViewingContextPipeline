"""Current evaluation arms and their shared, title-free generation sources."""

from dataclasses import dataclass

CONCAT_ARM_CONTRACT = "title-summary-nine-arms/v3"
CONCAT_POLICY = "title-summary-double-newline/v1"
EXPERIMENT_CONFIG_VERSION = "v6"


@dataclass(frozen=True)
class Arm:
    name: str
    representation: str
    model: str | None = None
    uses_title: bool = False
    scene_arm: str | None = None


ARMS = (
    Arm("meta", "metadata", uses_title=True),
    Arm("graph_qwen3vl", "graph", "qwen3vl", False, "graph_qwen3vl"),
    Arm("graph_qwen3vl_meta", "graph", "qwen3vl", True, "graph_qwen3vl"),
    Arm("graph_qwen3.5", "graph", "qwen3.5", False, "graph_qwen3.5"),
    Arm("graph_qwen3.5_meta", "graph", "qwen3.5", True, "graph_qwen3.5"),
    Arm("desc_qwen3.5_meta", "description", "qwen3.5", True, "desc_qwen3.5"),
    Arm("graph_gemini_meta", "graph", "gemini", True, "graph_gemini"),
    Arm("desc_qwen3vl_meta", "description", "qwen3vl", True, "desc_qwen3vl"),
    Arm("desc_gemini_meta", "description", "gemini", True, "desc_gemini"),
)
# Direct Graph extensions do not change the existing Text concatenation contract.
GRAPH_ONLY_ARMS = (
    Arm("graph_minicpm", "graph", "minicpm", False, "graph_minicpm"),
    Arm("graph_minicpm_meta", "graph", "minicpm", True, "graph_minicpm"),
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
    return CONCAT_ARM_CONTRACT


def registry(config):
    arms = {arm.name: arm for arm in ARMS}
    # Without a model configuration, expose the complete supported Arm catalog.
    if "models" not in config or "minicpm" in config["models"]:
        arms.update((arm.name, arm) for arm in GRAPH_ONLY_ARMS)
    return arms


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
