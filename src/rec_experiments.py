"""Validated recommendation definitions, loaded once per validation invocation."""

from pathlib import Path
import re

import yaml

CONFIG_PATH = Path("rec_experiments_config.yaml")
CONFIG_KEY = "_rec_experiments"
MODULE_VERSIONS = {
    "role_fusion": "role-fusion/v1",
    "activity_hypergraph": "segments-v1",
}
# Names written in rec_experiments_config.yaml. Results keep the internal names
# above in their identity, so renaming config entries never invalidates a Run.
MODULE_ALIASES = {
    "REAF": "role_fusion",
    "ASHA": "activity_hypergraph",
}


def parse_experiments(value):
    from arm_registry import generation_registry

    if not isinstance(value, dict) or set(value) - {"version", "defaults", "experiments"}:
        raise ValueError("recommendation config requires version, optional defaults, and experiments")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("recommendation config version must be 1")
    defaults = value.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - {"readout"}:
        raise ValueError("defaults allows only readout; source must be explicit in every experiment")
    if defaults.get("readout", "mean") != "mean":
        raise ValueError("readout must be mean")
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    sources = generation_registry({})
    names, result, metadata = set(), [], 0
    for row in experiments:
        if not isinstance(row, dict) or set(row) - {"label", "source", "modules", "readout"}:
            raise ValueError("experiment allows only label, source, modules, readout")
        name = row.get("label")
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9-]*", name) is None:
            raise ValueError("label must start with a lowercase letter and contain only a-z, 0-9, hyphens")
        if name in names:
            raise ValueError(f"duplicate experiment label: {name}")
        names.add(name)
        source = row.get("source")
        if not isinstance(source, str) or source not in {"metadata", *sources}:
            raise ValueError(f"{name}: source must explicitly select metadata or a registered extraction source")
        modules = row.get("modules", [])
        if not isinstance(modules, list) or any(not isinstance(m, str) for m in modules):
            raise ValueError(f"{name}: modules must be a list of module names")
        modules = [MODULE_ALIASES.get(m, m) for m in modules]
        if len(set(modules)) != len(modules) or set(modules) - MODULE_VERSIONS.keys():
            raise ValueError(f"{name}: modules must be unique supported module names "
                             f"({', '.join(MODULE_ALIASES)})")
        if modules and (source == "metadata" or sources[source].representation != "graph"):
            raise ValueError(f"{name}: modules are graph-only")
        readout = row.get("readout", defaults.get("readout", "mean"))
        if readout != "mean":
            raise ValueError(f"{name}: readout must be mean")
        metadata += source == "metadata"
        if metadata > 1:
            raise ValueError("at most one metadata experiment is allowed")
        result.append({"label": name, "source": source,
                       "modules": [m for m in MODULE_VERSIONS if m in modules], "readout": readout})
    return {"version": 1, "experiments": result}


def load_experiments(context):
    """Freeze a resolved snapshot in the context; spawned workers inherit it."""
    if CONFIG_KEY not in context.config:
        path = context.root / CONFIG_PATH
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            context.config[CONFIG_KEY] = parse_experiments(value)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"invalid recommendation config {path}: {exc}") from exc
    return context.config[CONFIG_KEY]


def experiment_definition(arm):
    from arm_registry import feature_source

    modules = (["role_fusion"] if arm.role_fusion else [])
    if "segments" in arm.extensions:
        modules.append("activity_hypergraph")
    return {"label": arm.name, "source": feature_source(arm), "modules": modules,
            "readout": arm.aggregation,
            "module_versions": {name: MODULE_VERSIONS[name] for name in modules}}
