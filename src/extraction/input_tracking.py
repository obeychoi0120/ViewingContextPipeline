"""Input state lives in the output; only dirty markers are temporary sidecars."""

from __future__ import annotations
from artifact_io import atomic_write_json
from extraction.recovery import fingerprint
from pipeline_runtime import read_json


def input_state_path(path):
    return path.parent / ".dirty" / f"{path.stem}.json"


def inputs_match(path, records):
    return (
        not input_state_path(path).exists()
        and path.is_file()
        and read_json(path).get("provenance", {}).get("scene_input_hash") == fingerprint(records)
    )


def clear_dirty(path):
    marker = input_state_path(path)
    marker.unlink(missing_ok=True)
    if marker.parent.exists() and not any(marker.parent.iterdir()):
        marker.parent.rmdir()


def invalidate_inputs(path):
    if path.exists():
        atomic_write_json(input_state_path(path), {"changed": True}, durable=True)
