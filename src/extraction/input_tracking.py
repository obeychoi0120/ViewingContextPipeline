"""Content hashes plus durable dirty markers protect pre-fingerprint artifacts too."""
from __future__ import annotations

import json

from artifact_io import atomic_write_json
from extraction.recovery import fingerprint


def marker(path):
    return path.parent / ".changed" / f"{path.stem}.json"


def mark_changed(path):
    changed = marker(path)
    if not changed.exists():
        atomic_write_json(changed, {"changed": True}, durable=True)


def clear_changed(path):
    marker(path).unlink(missing_ok=True)


def input_state_path(path):
    return path.parent / ".inputs" / f"{path.stem}.json"


def inputs_match(path, records):
    state_path = input_state_path(path)
    if state_path.with_suffix(".dirty").exists():
        return False
    if not state_path.exists():
        return True  # Legacy output: existing validation remains authoritative.
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state["input_hash"] == fingerprint(records)


def record_inputs(path, records):
    raw_count = sum(row.get("status") == "raw_fallback" for row in records)
    atomic_write_json(input_state_path(path), {
        "input_hash": fingerprint(records),
        "scene_count": len(records), "normal_scene_count": len(records) - raw_count,
        "raw_scene_count": raw_count,
    }, durable=True)
    input_state_path(path).with_suffix(".dirty").unlink(missing_ok=True)


def invalidate_inputs(path):
    dirty = input_state_path(path).with_suffix(".dirty")
    # Missing summaries already require generation; one marker covers all scene changes.
    if path.exists() and not dirty.exists():
        atomic_write_json(dirty, {"changed": True}, durable=True)
