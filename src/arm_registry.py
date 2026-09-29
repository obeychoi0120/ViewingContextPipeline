"""Current evaluation arms and their shared, title-free generation sources."""

from dataclasses import dataclass

CONCAT_ARM_CONTRACT = "title-summary-six-arms/v2"
CONCAT_POLICY = "title-summary-double-newline/v1"
EXPERIMENT_CONFIG_VERSION = "v4"


@dataclass(frozen=True)
class Arm:
    name: str
    representation: str
    model: str | None = None
    uses_title: bool = False
    scene_arm: str | None = None


ARMS = (
    Arm("meta", "metadata", uses_title=True),
    Arm("graph_qwen", "graph", "qwen", False, "graph_qwen"),
    Arm("graph_qwen_meta", "graph", "qwen", True, "graph_qwen"),
    Arm("graph_gemini_meta", "graph", "gemini", True, "graph_gemini"),
    Arm("desc_qwen_meta", "description", "qwen", True, "desc_qwen"),
    Arm("desc_gemini_meta", "description", "gemini", True, "desc_gemini"),
)
SOURCES = tuple(
    Arm(name, kind, model, False, name)
    for name, kind, model in (
        ("graph_qwen", "graph", "qwen"),
        ("desc_qwen", "description", "qwen"),
        ("graph_gemini", "graph", "gemini"),
        ("desc_gemini", "description", "gemini"),
    )
)


def arm_contract(config):
    return CONCAT_ARM_CONTRACT


def registry(config):
    return {arm.name: arm for arm in ARMS}


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
    return {arm.name: arm for arm in SOURCES}


def generated_arm(config, representation, model):
    if model not in {"gemini", "qwen"}:
        raise ValueError("model/source must be qwen or gemini")
    source = next(
        (arm for arm in SOURCES if arm.representation == representation and arm.model == model),
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
    if not summary and arm.model != model:
        raise ValueError(
            "Scene arm must identify the requested extraction model and representation"
        )
    return arm
