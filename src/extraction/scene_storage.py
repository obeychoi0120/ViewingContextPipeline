"""Scene payloads carry their own identity; no metadata sidecars are written."""

from pathlib import Path

from artifact_io import atomic_write_jsonl
from extraction.recovery import fingerprint
from pipeline_runtime import read_json, read_jsonl

METADATA_SCHEMA = "scene-metadata/v1"  # Read-only compatibility with older artifacts.


def metadata_path(path):
    path = Path(path)
    return path.parent / ".metadata" / f"{path.stem}.json"


def is_compact_scene(row):
    return isinstance(row, dict) and set(row) in (
        {"content_id", "description"}, {"content_id", "scene_graph"},
        {"content_id", "scene_idx", "description"}, {"content_id", "scene_idx", "scene_graph"},
    )


def _payload(path, record):
    fields = set(record) & {"description", "graph", "raw_response", "scene_graph"}
    if len(fields) != 1 or record.get("content_id", path.stem) != path.stem:
        raise ValueError(f"invalid scene record: {path}")
    index = record.get("scene_idx")
    if type(index) is not int or index < 0:
        raise ValueError(f"invalid scene index: {path}")
    field = fields.pop()
    public_field = "description" if field == "description" else "scene_graph"
    return {"content_id": path.stem, "scene_idx": index, public_field: record[field]}


def read_scene_records(path):
    """Read current payloads or recover explicit indices from legacy metadata."""
    path = Path(path)
    rows = read_jsonl(path)
    if not rows:
        return []
    if any(is_compact_scene(row) and "scene_idx" not in row for row in rows):
        sidecar = metadata_path(path)
        if not sidecar.is_file():
            raise ValueError(f"legacy scene file has no scene_idx or metadata: {path}; "
                             "restore its original .metadata before migration")
        metadata = read_json(sidecar)
        if (metadata.get("schema_version") != METADATA_SCHEMA
                or metadata.get("content_id") != path.stem
                or metadata.get("payload_hash") != fingerprint(rows)
                or not isinstance(metadata.get("rows"), list)
                or len(metadata["rows"]) != len(rows)):
            raise ValueError(f"scene metadata does not match legacy payload: {path}")
        rows = [{**row, "scene_idx": saved["record"]["scene_idx"]}
                for row, saved in zip(rows, metadata["rows"])]
    payload = [_payload(path, row) for row in rows]
    if len({row["scene_idx"] for row in payload}) != len(payload):
        raise ValueError(f"duplicate scene index: {path}")
    failure_path = path.parent / "failures" / f"{path.stem}.jsonl"
    failures = {row["scene_idx"]: row for row in read_jsonl(failure_path)} if failure_path.exists() else {}
    records = []
    for row in sorted(payload, key=lambda row: row["scene_idx"]):
        common = {"scene_idx": row["scene_idx"], "keyframes": []}
        if "description" in row:
            record = {"schema_version": "scene-description/v2", "content_id": path.stem,
                      **common, "description": row["description"]}
            failed = failures.get(row["scene_idx"])
            if failed and failed.get("raw_output") == row["description"]:
                record["status"] = "raw_fallback"
        elif isinstance(row["scene_graph"], str):
            record = {"schema_version": "graph-scene-raw/v1", "status": "raw_fallback",
                      **common, "raw_response": row["scene_graph"]}
        else:
            record = {**common, "graph": row["scene_graph"], "parse_mode": "unknown", "semantic_warnings": []}
        records.append(record)
    return records


def _remove_metadata(path):
    sidecar = metadata_path(path)
    sidecar.unlink(missing_ok=True)
    try:
        sidecar.parent.rmdir()
    except (FileNotFoundError, OSError):
        pass  # Other legacy contents may still need their metadata.


def write_scene_records(path, records):
    """Atomically publish self-contained payloads before deleting legacy metadata."""
    path = Path(path)
    payload = sorted((_payload(path, row) for row in records), key=lambda row: row["scene_idx"])
    if len({row["scene_idx"] for row in payload}) != len(payload):
        raise ValueError(f"duplicate scene index: {path}")
    if payload:
        atomic_write_jsonl(path, payload, durable=True)
    else:
        path.unlink(missing_ok=True)
    _remove_metadata(path)


def migrate_scene_file(path, records):
    path = Path(path)
    if metadata_path(path).exists() or any(not is_compact_scene(row) or "scene_idx" not in row
                                         for row in read_jsonl(path)):
        write_scene_records(path, records)
        return True
    return False


def migrate_scene_schema(context, *, force=False):
    """Remove legacy metadata after preserving scene indices, without model calls."""
    converted = unchanged = 0
    for representation in ("description", "graph"):
        for model in ("qwen", "gemini"):
            directory = context.extraction_dir(representation, model, "scenes")
            for path in sorted(directory.glob("*.jsonl")):
                if path.name in {"failure.jsonl", "failures.jsonl"}:
                    continue
                if migrate_scene_file(path, read_scene_records(path)):
                    converted += 1
                else:
                    unchanged += 1
    print(f"[SCHEMA] converted={converted} unchanged={unchanged}", flush=True)
    return {"converted": converted, "unchanged": unchanged}
