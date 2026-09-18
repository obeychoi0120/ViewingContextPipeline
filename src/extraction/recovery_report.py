"""Read existing scene metadata without treating legacy aggregate failure files as scene payloads."""
from collections import Counter
from arm_registry import select_arms
from extraction.scene_storage import read_scene_records


def recovery_report(context, *, branches=None, content_ids=None):
    selected = select_arms(context.config, list(branches) if branches is not None else None)
    contents = set(content_ids) if content_ids is not None else None
    report = {}
    for name, arm in selected.items():
        if arm.model is None:
            continue
        modes = Counter()
        attempt_count = unknown = 0
        directory = context.extraction_dir(arm.representation, arm.model, "scenes")
        for path in directory.glob("*.jsonl"):
            if path.name in {"failure.jsonl", "failures.jsonl"}:
                continue
            if contents is not None and path.stem not in contents:
                continue
            for row in read_scene_records(path):
                history = row.get("generation", {})
                unknown += not bool(history)
                attempt_count += history.get("attempt_count", int(bool(history.get("input_key"))))
                modes[row.get("status", row.get("parse_mode", history.get("repair_mode", "unknown")))] += 1
        report[name] = {"scene_modes": dict(modes), "scene_attempt_count": attempt_count,
                        "unknown_history_count": unknown}
    return report
