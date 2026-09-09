"""Explicit raw artifact variants; ordinary malformed artifacts remain errors."""
RAW_GRAPH_SCHEMA = "graph-scene-raw/v1"
RAW_SUMMARY_SCHEMA = "video-summary-raw/v1"


def raw_graph_record(row, text):
    return {"schema_version": RAW_GRAPH_SCHEMA, "status": "raw_fallback",
            "scene_idx": row["scene_idx"], "keyframes": row["keyframes"],
            "raw_response": text}


def is_raw_graph(row):
    return isinstance(row, dict) and row.get("schema_version") == RAW_GRAPH_SCHEMA


def valid_raw_graph(row):
    return (is_raw_graph(row) and set(row) == {
        "schema_version", "status", "scene_idx", "keyframes", "raw_response",
    } and row.get("status") == "raw_fallback"
        and type(row.get("scene_idx")) is int and row["scene_idx"] >= 0
        and isinstance(row.get("keyframes"), list) and bool(row["keyframes"])
        and isinstance(row.get("raw_response"), str) and bool(row["raw_response"].strip()))


def raw_summary_document(content_id, arm, scene_count, text):
    return {"schema_version": RAW_SUMMARY_SCHEMA, "content_id": content_id, "arm": arm,
            "status": "raw_fallback", "scene_count": scene_count, "text": text}
