"""Recommendation experiments and independent, title-free extraction sources."""

from dataclasses import dataclass

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
    # Internal sidecar extensions; public module names live in rec_experiments.
    extensions: tuple[str, ...] = ()


SOURCES = tuple(
    Arm(name, kind, model, False, name)
    for name, kind, model in (
        ("graph_qwen3vl", "graph", "qwen3vl"),
        ("desc_qwen3vl", "description", "qwen3vl"),
        ("graph_gemini", "graph", "gemini"),
        ("desc_gemini", "description", "gemini"),
    )
)


def arm_contract(config):
    return RECOMMENDATION_ARM_CONTRACT


def registry(config):
    from rec_experiments import CONFIG_KEY

    if CONFIG_KEY not in config:
        raise ValueError("recommendation experiments are not loaded; load rec_experiments_config.yaml first")
    sources = generation_registry(config)
    result = {}
    for row in config[CONFIG_KEY]["experiments"]:
        source = sources.get(row["source"])
        result[row["label"]] = Arm(
            row["label"], source.representation if source else "metadata",
            source.model if source else None, True, source.name if source else None,
            row["readout"], "role_fusion" in row["modules"],
            ("segments",) if "activity_hypergraph" in row["modules"] else (),
        )
    return result


def feature_source(arm):
    return arm.scene_arm or "metadata"


def target_set(config, target):
    return "_".join(select_arms(config, target))


def active_arms(config):
    if "arms" in config.get("protocol", {}):
        raise ValueError("protocol.arms is not supported; edit rec_experiments_config.yaml")
    return list(registry(config))


def select_arms(config, target=None):
    registered = registry(config)
    names = active_arms(config) if target is None else target
    if not names or len(names) != len(set(names)) or set(names) - registered.keys():
        raise ValueError(f"experiments must select from {', '.join(registered)}")
    return {name: arm for name, arm in registered.items() if name in names}


def generation_registry(config):
    return {arm.name: arm for arm in SOURCES}


def generated_arm(config, representation, model):
    if model not in {"gemini", "qwen3vl"}:
        raise ValueError("model/source must be qwen3vl or gemini")
    source = next(
        (arm for arm in generation_registry(config).values() if arm.representation == representation and arm.model == model),
        None,
    )
    if source is None:
        raise ValueError(f"unsupported generation source: {representation}/{model}")
    return source


def resolve_generation_arm(config, name, representation, model):
    arm = generation_registry(config).get(name)
    if arm is None:
        source = registry(config).get(name) if "_rec_experiments" in config else None
        hint = f"; use source --arm {source.scene_arm}" if source and source.scene_arm else ""
        raise ValueError(f"invalid generation source: {name}{hint}")
    if arm.representation != representation:
        raise ValueError(f"invalid {representation} generation arm: {name}")
    if arm.model != model:
        raise ValueError(
            "Scene arm must identify the requested extraction model and representation"
        )
    return arm
