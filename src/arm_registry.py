"""Fixed evaluation arms; selected prompts identify each run's experiment."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    representation: str
    model: str | None = None

    @property
    def fallback(self) -> str | None:
        return self.name.removesuffix("gemini") + "qwen" if self.model == "gemini" else None


def registry(config: dict) -> dict[str, Arm]:
    arms = [
        Arm(f"{prefix}_{model}", kind, model)
        for prefix, kind in (("desc", "description"), ("graph", "graph"))
        for model in ("gemini", "qwen")
    ]
    arms.append(Arm("metadata", "metadata"))
    return {arm.name: arm for arm in arms}


def active_arms(config: dict) -> list[str]:
    registered = registry(config)
    names = config["protocol"]["arms"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) for n in names)
        or len(names) != len(set(names))
        or set(names) - registered.keys()
    ):
        raise ValueError(
            "protocol.arms must be a nonempty, unique list of registered lowercase arms"
        )
    return names


def select_arms(config: dict, target: list[str] | None = None) -> dict[str, Arm]:
    registered = registry(config)
    names = active_arms(config) if target is None else target
    if not names or set(names) - registered.keys():
        raise ValueError(f"--target must select from {', '.join(registered)}")
    return {name: arm for name, arm in registered.items() if name in names}


def generated_arm(config: dict, representation: str, model: str) -> Arm:
    if model not in {"gemini", "qwen"}:
        raise ValueError("model/source must be qwen or gemini")
    return next(
        a
        for a in registry(config).values()
        if a.representation == representation and a.model == model
    )
