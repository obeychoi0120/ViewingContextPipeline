"""Compact generation history survives deletion of completed recovery journals."""
from collections import Counter
from arm_registry import select_arms
from pipeline_runtime import read_jsonl


def recovery_report(context, *, branches=None):
    selected = select_arms(context.config, list(branches) if branches is not None else None)
    report = {}
    for name, arm in selected.items():
        if arm.model is None:
            continue
        modes = Counter()
        attempt_count = unknown = 0
        directory = context.extraction_dir(arm.representation, arm.model, "scenes")
        for path in directory.glob("*.jsonl"):
            for row in read_jsonl(path):
                history = row.get("generation", {})
                unknown += not bool(history)
                attempt_count += history.get("attempt_count", 0)
                modes[row.get("status", history.get("repair_mode", "unknown"))] += 1
        report[name] = {"scene_modes": dict(modes), "scene_attempt_count": attempt_count,
                        "unknown_history_count": unknown}
    return report
