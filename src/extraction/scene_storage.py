"""Scene payloads carry their own identity; no metadata sidecars are written."""

from pathlib import Path

from artifact_io import atomic_write_jsonl
from extraction.token_usage import validate_tokens
from extraction.graph_warnings import warning_tags
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from artifact_io import read_jsonl

GRAPH_FIELDS = ("content_id", "scene_idx", "tokens", "warning", "scene_graph")
DESCRIPTION_FIELDS = ("content_id", "scene_idx", "tokens", "description")


def _graph_warning(record, value):
    if "warning" in record:
        return warning_tags(record["warning"])
    supplied = record.get("semantic_warnings")
    if supplied:
        try:
            return warning_tags(supplied)
        except ValueError:
            pass  # Historical descriptions are not stable tags; recheck the payload.
    if isinstance(value, str):
        parsed = parse_or_repair_graph(value)
        return (
            list(parsed.warning) if parsed.graph is None else graph_semantic_warnings(parsed.graph)
        )
    return []


def _payload(path, record):
    fields = set(record) & {"description", "graph", "raw_response", "scene_graph"}
    if len(fields) != 1 or record.get("content_id", path.stem) != path.stem:
        raise ValueError(f"invalid scene record: {path}")
    index = record.get("scene_idx")
    if type(index) is not int or index < 0:
        raise ValueError(f"invalid scene index: {path}")
    field = fields.pop()
    public_field = "description" if field == "description" else "scene_graph"
    return {
        "content_id": path.stem,
        **(
            {"warning": _graph_warning(record, record[field])}
            if public_field == "scene_graph"
            else {}
        ),
        "scene_idx": index,
        "tokens": validate_tokens(record.get("tokens")),
        public_field: record[field],
        **(
            {"graph_format": "text"}
            if isinstance(record.get("graph"), str) or record.get("graph_format") == "text"
            else {}
        ),
        **({"provenance": record["provenance"]} if "provenance" in record else {}),
    }


def read_scene_records(path):
    """Read self-contained payloads with explicit scene indices."""
    path = Path(path)
    rows = read_jsonl(path)
    if not rows:
        return []
    legacy_raw_indices = {
        row["scene_idx"]
        for row in rows
        if "raw_response" in row
        or row.get("status") == "raw_fallback"
        or (
            isinstance(row.get("scene_graph"), str)
            and "provenance" in row
            and row.get("graph_format") != "text"
        )
    }
    payload = [_payload(path, row) for row in rows]
    if len({row["scene_idx"] for row in payload}) != len(payload):
        raise ValueError(f"duplicate scene index: {path}")
    failure_path = path.parent / "failures" / f"{path.stem}.jsonl"
    failures = (
        {row["scene_idx"]: row for row in read_jsonl(failure_path)} if failure_path.exists() else {}
    )
    records = []
    for row in sorted(payload, key=lambda row: row["scene_idx"]):
        common = {"scene_idx": row["scene_idx"], "keyframes": [], "tokens": row["tokens"]}
        if "description" in row:
            record = {
                "schema_version": "scene-description/v2",
                "content_id": path.stem,
                **common,
                "description": row["description"],
            }
            failed = failures.get(row["scene_idx"])
            if failed and failed.get("raw_output") == row["description"]:
                record["status"] = "raw_fallback"
        elif row.get("graph_format") == "text":
            if not isinstance(row["scene_graph"], str):
                raise ValueError(f"invalid text graph: {path}")
            record = {
                **common,
                "graph": row["scene_graph"],
                "parse_mode": "text",
                "semantic_warnings": [],
            }
        elif isinstance(row["scene_graph"], str) and (
            row["scene_idx"] in legacy_raw_indices or row["scene_idx"] in failures
        ):
            record = {
                "schema_version": "graph-scene-raw/v1",
                "status": "raw_fallback",
                **common,
                "raw_response": row["scene_graph"],
            }
        elif isinstance(row["scene_graph"], str):
            record = {
                **common,
                "graph": row["scene_graph"],
                "parse_mode": "text",
                "semantic_warnings": [],
            }
        else:
            record = {
                **common,
                "graph": row["scene_graph"],
                "parse_mode": "unknown",
                "semantic_warnings": [],
            }
        if "scene_graph" in row:
            record["semantic_warnings"] = row["warning"]
        if "provenance" in row:
            record["provenance"] = row["provenance"]
        records.append(record)
    return records


def write_scene_records(path, records):
    """Atomically publish self-contained scene payloads."""
    path = Path(path)
    records = list(records)
    payload = sorted((_payload(path, row) for row in records), key=lambda row: row["scene_idx"])
    # Graph values carry their format in their type: object or preserved raw text.
    # Persist only the public fields, in their specified display order.
    payload = [
        {key: row[key] for key in (GRAPH_FIELDS if "scene_graph" in row else DESCRIPTION_FIELDS)}
        for row in payload
    ]
    if len({row["scene_idx"] for row in payload}) != len(payload):
        raise ValueError(f"duplicate scene index: {path}")
    legacy_failures = [row for row in records if "raw_response" in row]
    if legacy_failures:
        # Old raw-fallback status belongs in the existing failure log, not in
        # the compact payload. Publish it first so migration cannot lose retry state.
        from extraction.failures import FailureLog

        failures = FailureLog(path.parent, scenes=True)
        for row in legacy_failures:
            if not failures.contains(path.stem, row["scene_idx"]):
                failures.record(
                    path.stem,
                    row["scene_idx"],
                    "graph validation failed; raw response retained",
                    row["raw_response"],
                )
    if payload:
        atomic_write_jsonl(path, payload, durable=True, sort_keys=False)
    else:
        path.unlink(missing_ok=True)
