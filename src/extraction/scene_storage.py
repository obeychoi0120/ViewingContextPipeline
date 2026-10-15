"""Current compact scene payloads and atomic writes from execution records."""

from pathlib import Path

from artifact_io import atomic_write_jsonl, read_jsonl
from extraction.errors import UnsupportedArtifactError
from extraction.token_usage import validate_tokens
from extraction.graph_warnings import warning_tags
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.structured_output import (
    GRAPH_JSON_SCHEMA, OutputValidationError, normalize_graph_references,
)

GRAPH_FIELDS = ("content_id", "scene_idx", "tokens", "warning", "scene_graph")
DESCRIPTION_FIELDS = ("content_id", "scene_idx", "tokens", "description")


def validate_scene_payload(path, row):
    """Reject obsolete storage/Graph contracts without transforming source data."""
    path = Path(path)
    fields = GRAPH_FIELDS if "scene_graph" in row else DESCRIPTION_FIELDS
    if set(row) != set(fields):
        raise UnsupportedArtifactError(f"unsupported compact scene fields: {path}")
    if row["content_id"] != path.stem or type(row["scene_idx"]) is not int or row["scene_idx"] < 0:
        raise ValueError(f"invalid scene identity: {path}")
    validate_tokens(row["tokens"])
    if "scene_graph" in row:
        warning_tags(row["warning"])
        graph = row["scene_graph"]
        if isinstance(graph, dict):
            if set(graph) != set(GRAPH_JSON_SCHEMA["properties"]):
                raise UnsupportedArtifactError(f"unsupported Graph fields; expected Context/Entities/Actions: {path}")
            if isinstance(graph["actions"], list) and any(
                isinstance(action, dict) and set(action) != set(GRAPH_JSON_SCHEMA["properties"]["actions"]["items"]["properties"])
                for action in graph["actions"]
            ):
                raise UnsupportedArtifactError(f"unsupported Action fields; receiver-free roles and place required: {path}")
        elif not isinstance(graph, str) or not graph.strip():
            raise ValueError(f"invalid graph observation: {path}")
    elif not isinstance(row["description"], str) or not row["description"].strip():
        raise ValueError(f"invalid description observation: {path}")


def _graph_warning(record, value):
    if "warning" in record:
        return warning_tags(record["warning"])
    if "semantic_warnings" in record:
        return warning_tags(record["semantic_warnings"])
    if isinstance(value, str):
        parsed = parse_or_repair_graph(value)
        return list(parsed.warning)
    return graph_semantic_warnings(value)


def recover_scene_graph(graph, warnings):
    """Reparse stored text and normalize references without semantic warning checks."""
    warnings = warning_tags(warnings)
    if isinstance(graph, str):
        parsed = parse_or_repair_graph(graph)
        return (
            graph if parsed.graph is None else parsed.graph,
            list(parsed.warning),
        )
    try:
        normalized, _ = normalize_graph_references(graph)
    except OutputValidationError as exc:
        return graph, exc.tags
    return normalized, warnings


def _payload(path, record):
    fields = set(record) & {"description", "graph", "scene_graph"}
    if len(fields) != 1 or "raw_response" in record:
        raise UnsupportedArtifactError(f"unsupported scene execution record: {path}")
    field = fields.pop()
    payload = {
        "content_id": record.get("content_id", path.stem),
        "scene_idx": record.get("scene_idx"),
        "tokens": validate_tokens(record.get("tokens")),
    }
    if field == "description":
        payload["description"] = record[field]
    else:
        payload.update(warning=_graph_warning(record, record[field]), scene_graph=record[field])
    validate_scene_payload(path, payload)
    return {key: payload[key] for key in (GRAPH_FIELDS if "scene_graph" in payload else DESCRIPTION_FIELDS)}


def read_scene_records(path):
    """Adapt current persisted fields to the current in-memory execution contract."""
    path = Path(path)
    rows = read_jsonl(path)
    for row in rows:
        validate_scene_payload(path, row)
    if len({row["scene_idx"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate scene index: {path}")
    records = []
    for row in sorted(rows, key=lambda row: row["scene_idx"]):
        common = {"scene_idx": row["scene_idx"], "keyframes": [], "tokens": row["tokens"]}
        if "description" in row:
            record = {
                "schema_version": "scene-description/v2", "content_id": path.stem,
                **common, "description": row["description"],
            }
        else:
            graph, warnings = recover_scene_graph(row["scene_graph"], row["warning"])
            record = {
                **common, "graph": graph,
                "parse_mode": "text" if isinstance(graph, str) else (
                    "repaired" if graph != row["scene_graph"] else "unknown"
                ),
                "semantic_warnings": warnings,
            }
        records.append(record)
    return records


def write_scene_records(path, records):
    """Validate all rows before publishing; never migrate existing artifacts."""
    path = Path(path)
    payload = sorted((_payload(path, row) for row in records), key=lambda row: row["scene_idx"])
    if len({row["scene_idx"] for row in payload}) != len(payload):
        raise ValueError(f"duplicate scene index: {path}")
    if payload:
        atomic_write_jsonl(path, payload, durable=True, sort_keys=False)
    else:
        path.unlink(missing_ok=True)


def ensure_generation_identity(directory, provenance, *, force=False):
    """Record initial generation settings without blocking resume after policy changes."""
    from artifact_io import atomic_write_json
    from model_provenance import canonical

    directory = Path(directory)
    identity_path = directory / "generation_identity.json"
    identity = canonical(provenance)
    directory.mkdir(parents=True, exist_ok=True)
    if force:
        # A new identity must never label old outputs after an interrupted regeneration.
        for path in [*directory.glob("*.jsonl"), *(directory / "failures").glob("*.jsonl")]:
            path.unlink()
    if force or not identity_path.exists():
        atomic_write_json(identity_path, {"identity": identity}, durable=True)
