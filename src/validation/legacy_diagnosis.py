"""Read the pre-experiment-config v4 results without rewriting recommendation evidence."""

from dataclasses import asdict, replace

from arm_registry import Arm, registry
from artifact_io import read_json
from rec_experiments import CONFIG_KEY, experiment_definition
from validation.leave_one_out_recommendation import combination_dir


# Definitions from the last fixed registry, not aliases for earlier GNN architectures.
LEGACY_ARMS = {
    arm.name: arm for arm in (
        Arm("base", "metadata", uses_title=True),
        Arm("d", "description", "qwen3vl", True, "desc_qwen3vl", "mean"),
        Arm("g0", "graph", "qwen3vl", True, "graph_qwen3vl", "mean"),
        Arm("g1", "graph", "qwen3vl", True, "graph_qwen3vl", "mean",
            extensions=("segments",)),
        Arm("g2", "graph", "qwen3vl", True, "graph_qwen3vl", "mean", role_fusion=True,
            extensions=("segments",)),
    )
}


def stored_arm(arm):
    value = asdict(arm)
    value["extensions"] = list(arm.extensions)
    return value


def diagnosis_context(context, targets):
    """Use historical labels only when the selected definitions match the fixed registry."""
    configured = registry(context.config)
    resolved = []
    changed = False
    for name in targets:
        arm = configured[name]
        legacy = next((old for old in LEGACY_ARMS.values()
                       if replace(arm, name=old.name) == old), None)
        current_paths = [combination_dir(context, seed, name)
                         for seed in context.config["validation"]["model"]["seeds"]]
        if legacy is None or legacy.name == name:
            resolved.append(name)
            continue
        old_paths = [path.parent / legacy.name for path in current_paths]
        if any(path.exists() for path in old_paths):
            if any(path.exists() for path in current_paths):
                raise ValueError(f"ambiguous diagnosis results: both {name} and {legacy.name} exist")
            if not all(path.is_dir() for path in old_paths):
                raise ValueError(f"missing legacy recommendation seed directory: {legacy.name}")
            resolved.append(legacy.name)
            changed = True
        else:
            resolved.append(name)
    if not changed:
        return context, list(targets)
    if len(set(resolved)) != len(resolved):
        raise ValueError("multiple selected experiments resolve to the same legacy result")
    rows = []
    for name, label in zip(targets, resolved, strict=True):
        definition = experiment_definition(replace(configured[name], name=label))
        rows.append({key: value for key, value in definition.items() if key != "module_versions"})
    config = {**context.config, CONFIG_KEY: {"version": 1, "experiments": rows}}
    print(f"[Diagnosis] 기존 recommendation 이름으로 분석합니다: {', '.join(resolved)}", flush=True)
    return replace(context, config=config), resolved


def diagnosis_identity(directory, identity):
    """Omit only the newly added field, after checking the exact historical Arm definition."""
    training = read_json(directory / "training.json")
    if "experiment_definition" in training:
        return identity
    complete = read_json(directory / "complete.json")
    arm = LEGACY_ARMS.get(identity["arm"])
    if (arm is None or identity["architecture_version"] != "sasrec-scene-content/v4"
            or "experiment_definition" in complete.get("identity", {})
            or identity["experiment_arm"] != stored_arm(arm)
            or training.get("experiment_arm") != stored_arm(arm)):
        raise ValueError(f"unrecognized legacy experiment definition: {directory}")
    return {key: value for key, value in identity.items() if key != "experiment_definition"}
