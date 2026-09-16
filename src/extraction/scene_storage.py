"""Small public scene records paired with lossless scene metadata."""

from pathlib import Path

from artifact_io import atomic_write_json, atomic_write_jsonl
from extraction.recovery import fingerprint
from pipeline_runtime import read_json, read_jsonl


METADATA_SCHEMA = "scene-metadata/v1"


def metadata_path(path):
    path = Path(path)
    return path.parent / ".metadata" / f"{path.stem}.json"


def is_compact_scene(row):
    return isinstance(row, dict) and set(row) in (
        {"content_id", "description"}, {"content_id", "scene_graph"},
    )


def read_scene_records(path):
    """Reconstruct the original records, preserving existing input hashes exactly."""
    path = Path(path)
    rows = read_jsonl(path)
    if not rows or not any(is_compact_scene(row) for row in rows):
        return rows  # Existing full records remain readable without regeneration.
    if not all(is_compact_scene(row) and row["content_id"] == path.stem for row in rows):
        raise ValueError(f"invalid or mixed scene schema: {path}")
    sidecar = metadata_path(path)
    if not sidecar.is_file():
        raise ValueError(f"missing scene metadata: {sidecar}")
    metadata = read_json(sidecar)
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != METADATA_SCHEMA
            or metadata.get("content_id") != path.stem
            or metadata.get("payload_hash") != fingerprint(rows)
            or not isinstance(metadata.get("rows"), list)
            or len(metadata["rows"]) != len(rows)):
        raise ValueError(f"scene metadata does not match payload: {path}")
    records = []
    for row, saved in zip(rows, metadata["rows"]):
        if not isinstance(saved, dict):
            raise ValueError(f"invalid scene metadata row: {sidecar}")
        field = saved.get("body_field")
        record = saved.get("record")
        public_field = "description" if field == "description" else "scene_graph"
        if (field not in {"description", "graph", "raw_response"}
                or not isinstance(record, dict) or field in record
                or public_field not in row
                or record.get("content_id", path.stem) != path.stem):
            raise ValueError(f"invalid scene metadata row: {sidecar}")
        records.append({**record, field: row[public_field]})
    return records


def write_scene_records(path, records):
    """Publish metadata and payload atomically per file; interrupted pairs are regenerated."""
    path = Path(path)
    sidecar = metadata_path(path)
    if not records:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        return
    payload, metadata = [], []
    for record in records:
        fields = set(record) & {"description", "graph", "raw_response"}
        if len(fields) != 1 or record.get("content_id", path.stem) != path.stem:
            raise ValueError(f"invalid scene record: {path}")
        field = fields.pop()
        public_field = "description" if field == "description" else "scene_graph"
        payload.append({"content_id": path.stem, public_field: record[field]})
        metadata.append({"body_field": field,
                         "record": {key: value for key, value in record.items() if key != field}})
    atomic_write_json(sidecar, {"schema_version": METADATA_SCHEMA, "content_id": path.stem,
                               "payload_hash": fingerprint(payload), "rows": metadata}, durable=True)
    atomic_write_jsonl(path, payload, durable=True)


def migrate_scene_schema(context, *, force=False):
    """Convert existing scene files without model calls or changing logical records."""
    from extraction.failures import FailureLog
    converted = unchanged = 0
    for representation in ("description", "graph"):
        for model in ("qwen", "gemini"):
            directory = context.extraction_dir(representation, model, "scenes")
            FailureLog(directory, scenes=True)
            for path in sorted(directory.glob("*.jsonl")):
                if path.name in {"failure.jsonl", "failures.jsonl"}:
                    continue
                records = read_scene_records(path)
                if all(is_compact_scene(row) for row in read_jsonl(path)):
                    unchanged += 1
                    continue
                write_scene_records(path, records)
                converted += 1
                if converted % 1000 == 0:
                    print(f"[SCHEMA] converted={converted} unchanged={unchanged}", flush=True)
    print(f"[SCHEMA] converted={converted} unchanged={unchanged}", flush=True)
    return {"converted": converted, "unchanged": unchanged}
