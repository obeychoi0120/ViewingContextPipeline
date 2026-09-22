"""Evaluation arms and their explicit shared Scene inputs."""
from __future__ import annotations
from dataclasses import dataclass

ARM_CONTRACT = "shared-scenes-nine-arms/v1"


@dataclass(frozen=True)
class Arm:
    name: str
    representation: str
    model: str | None = None
    uses_title: bool = False
    scene_arm: str | None = None

    @property
    def fallback(self):
        return self.name.removesuffix("gemini") + "qwen" if self.model == "gemini" else None


def legacy_layout(config):
    return (config.get("schema_version") == "viewing-context-config/v5"
            or ("schema_version" not in config and "metadata" in config.get("protocol", {}).get("arms", [])))


def registry(config):
    if legacy_layout(config):
        arms = [Arm(f"{prefix}_{model}", kind, model, True, f"{prefix}_{model}")
                for prefix, kind in (("desc", "description"), ("graph", "graph"))
                for model in ("gemini", "qwen")]
        arms.append(Arm("metadata", "metadata", uses_title=True))
    else:
        arms = [Arm("meta", "metadata", uses_title=True)]
        for title in (False, True):
            for prefix, kind in (("graph", "graph"), ("desc", "description")):
                for model in ("qwen", "gemini"):
                    scene = f"{prefix}_{model}"
                    name = f"{prefix}_meta_{model}" if title else scene
                    arms.append(Arm(name, kind, model, title, scene))
    return {a.name: a for a in arms}


def active_arms(config):
    registered = registry(config)
    names = config["protocol"]["arms"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) for n in names)
            or len(names) != len(set(names)) or set(names) - registered.keys()):
        raise ValueError("protocol.arms must be a nonempty, unique list of registered lowercase arms")
    return names


def select_arms(config, target=None):
    registered = registry(config)
    names = active_arms(config) if target is None else target
    if not names or len(names) != len(set(names)) or set(names) - registered.keys():
        raise ValueError(f"--target must select from {', '.join(registered)}")
    return {name: arm for name, arm in registered.items() if name in names}


def generated_arm(config, representation, model):
    if model not in {"gemini", "qwen"}:
        raise ValueError("model/source must be qwen or gemini")
    return next(a for a in registry(config).values()
                if a.representation == representation and a.model == model
                and a.name == a.scene_arm)


def resolve_generation_arm(config, name, representation, model, *, summary=False):
    arm = registry(config).get(name)
    if arm is None or arm.representation != representation or arm.model is None:
        raise ValueError(f"invalid {representation} generation arm: {name}")
    if not summary and (arm.name != arm.scene_arm or arm.model != model):
        raise ValueError("Scene arm must identify the requested extraction model and representation")
    return arm
