import json

import numpy as np

from pipeline_runtime import read_json, write_json
from validation.diagnosis_representations import representation_report
from validation.representation_provenance import state_path


def test_report_bounds_details_and_preserves_full_evidence(v5_context):
    context = v5_context
    rows = [{
        "content_id": f"video-{i}", "actual_arm": "desc_qwen", "source_arm": "desc_qwen",
        "status": "complete", "summary_schema": "video-summary/v4", "summary_policy": "prompt",
        "source_provenance": {"model": "qwen", "prompt_hash": "prompt"},
        "generation": None, "violations": ["warning"], "correction_count": 1,
        "word_count": 100 + i, "document_hash": str(i), "source_path": f"/summaries/{i}.json",
    } for i in range(25)]
    state = {
        "input_hash": "inputs", "embedding_hash": "embedding", "recommendation_hash": "rec",
        "truncation": {"text_count": 25, "truncated_count": 2}, "sources": rows,
    }
    path = state_path(context, "desc_gemini")
    write_json(path, state)
    original = path.read_bytes()
    report, fallbacks = representation_report(context, ["desc_gemini"])
    arm = report["desc_gemini"]
    summary = arm["sources_summary"]
    assert "sources" not in arm
    assert arm["input_hash"] == "inputs" and arm["embedding_hash"] == "embedding"
    assert arm["recommendation_hash"] == "rec" and arm["truncation"] == state["truncation"]
    assert summary["count"] == summary["issue_count"] == 25
    assert len(summary["issue_examples"]) == 10
    assert summary["issue_examples_omitted_count"] == 15
    assert summary["generation_record_count"] == 0
    assert summary["word_count"] == {"count": 25, "min": 100, "max": 124, "mean": 112}
    assert summary["distributions"]["source_provenance"] == {
        "distinct_count": 1,
        "values": [{"value": rows[0]["source_provenance"], "count": 25}],
        "omitted_count": 0,
    }
    fallback = fallbacks["desc_gemini"]
    assert fallback["count"] == 25 and fallback["omitted_count"] == 15
    assert len(fallback["examples"]) == 10
    for example in fallback["examples"]:
        source = state["sources"][example["source_index"]]
        assert source["content_id"] == example["content_id"]
        assert source["actual_arm"] == example["actual_arm"]
    assert context.run_root / arm["details_path"] == path
    assert fallback["details_path"] == arm["details_path"]
    assert path.read_bytes() == original


def test_mixed_provenance_is_bounded_and_metadata_needs_no_summary_fields(v5_context):
    context = v5_context
    write_json(state_path(context, "graph_qwen"), {"sources": [
        {"content_id": str(i), "source_provenance": {"prompt_hash": str(i)}}
        for i in range(30)
    ]})
    write_json(state_path(context, "metadata"), {"sources": [
        {"content_id": "one", "source_path": "/metadata.jsonl"}
    ]})
    report, fallbacks = representation_report(context, ["graph_qwen", "metadata", "desc_gemini"])
    distribution = report["graph_qwen"]["sources_summary"]["distributions"]["source_provenance"]
    assert distribution["distinct_count"] == 30
    assert len(distribution["values"]) == 10 and distribution["omitted_count"] == 20
    assert report["metadata"]["sources_summary"]["count"] == 1
    assert report["metadata"]["sources_summary"]["issue_count"] == 0
    assert report["desc_gemini"]["sources_summary"]["count"] == 0
    assert report["desc_gemini"]["sources_summary"]["word_count"]["mean"] is None
    assert fallbacks["desc_gemini"]["count"] == 0
    assert set(fallbacks) == {"desc_gemini"}


def test_diagnose_writes_compact_v3_with_statistics(ready_context, monkeypatch):
    from validation import rolling_diagnosis

    context = ready_context
    # Enough repeated rows to detect accidentally copying the entire state again.
    write_json(state_path(context, "metadata"), {"sources": [
        {"content_id": str(i), "source_path": "/metadata.jsonl"} for i in range(20000)
    ]})
    monkeypatch.setattr("validation.metadata.verify_missing_metadata", lambda *args: {})
    monkeypatch.setattr(rolling_diagnosis, "collect_metrics", lambda *args, **kwargs: (
        np.ones((3, 7, 1)), np.ones((3, 7)), {"means": {"metadata": {"HR@10": 0.1}}}
    ))
    assert rolling_diagnosis.diagnose(context, target=["metadata"])["status"] == "pass"
    document = read_json(context.diagnosis_path)
    assert document["schema_version"] == "rolling-diagnosis/v3"
    assert document["statistics"]["status"] == "computed"
    assert document["representations"]["metadata"]["sources_summary"]["count"] == 20000
    assert "sources" not in document["representations"]["metadata"]
    assert len(json.dumps(document)) < 20000
    assert len(read_json(state_path(context, "metadata"))["sources"]) == 20000
