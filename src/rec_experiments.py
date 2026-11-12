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
    names, result = set(), []
    for row in experiments:
        if not isinstance(row, dict) or set(row) - {"label", "source", "modules", "readout",
                                                     "segments", "segments_seed"}:
            raise ValueError("experiment allows only label, source, modules, readout, segments, segments_seed")
        name = row.get("label")
        # Labels name result directories: only keep characters that are safe in a path.
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+.-]*", name) is None:
            raise ValueError("label must start with a letter or digit and contain only "
                             "letters, digits, _, +, ., -")
        if name.lower() in names:
            raise ValueError(f"duplicate experiment label (case-insensitive): {name}")
        names.add(name.lower())
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
        resolved = {"label": name, "source": source,
                    "modules": [m for m in MODULE_VERSIONS if m in modules], "readout": readout}
        # Omitted segments keep the default ASHA incidence and add nothing to the resolved row,
        # so existing experiments keep their identity.
        if "segments" in row or "segments_seed" in row:
            resolved["segments"] = parse_segments(name, row, modules)
        result.append(resolved)
    return {"version": 1, "experiments": result}


def parse_segments(name, row, modules):
    from validation.segment_variants import VARIANTS

    if "activity_hypergraph" not in modules:
        raise ValueError(f"{name}: segments requires the ASHA module")
    variant = row.get("segments")
    if variant not in VARIANTS:
        raise ValueError(f"{name}: segments must be one of {', '.join(VARIANTS)}")
    seed = row.get("segments_seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError(f"{name}: segments_seed must be a non-negative integer")
    return {"variant": variant, "seed": seed}


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
    definition = {"label": arm.name, "source": feature_source(arm), "modules": modules,
                  "readout": arm.aggregation,
                  "module_versions": {name: MODULE_VERSIONS[name] for name in modules}}
    if arm.segments is not None:
        definition["segments"] = {"variant": arm.segments[0], "seed": arm.segments[1]}
    return definition
