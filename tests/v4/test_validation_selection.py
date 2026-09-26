"""Current arms share the full catalog even when generation is incomplete."""

import numpy as np
import pytest

from arm_registry import registry
from pipeline_runtime import read_jsonl, write_jsonl
from validation.representation_inputs import documents_for_arm
from validation.rolling_data import EventTable
from validation.selection import cohort_directory, load_validation_cohort, prepare_validation_cohort
from validation.steps import embed_representations


@pytest.fixture
def summaries(ready_context, fake_models, generate_all):
    generate_all(ready_context)
    return ready_context


def test_full_catalog_applies_to_all_targets_and_preserves_shared_data(summaries):
    context = summaries
    original = {p: p.read_bytes() for p in context.cohort_dir.iterdir() if p.is_file()}
    for path in context.summary_arm_dir("graph_qwen").glob("*.json"):
        path.unlink()
    before = context.require_ready_cohort()
    embed_representations(context, target=list(registry(context.config)))
    cohort = load_validation_cohort(context)
    assert len(cohort["catalog"]) == 4 and not cohort["excluded"]
    assert cohort["events"] == read_jsonl(context.cohort_dir / "events.jsonl")
    assert cohort["plan"]["splits"] == before["plan"]["splits"]
    table = EventTable(cohort["events"])
    assert table.items == ["1", "2", "3", "4"]
    assert len(table.rows) == 72
    with np.load(context.representations_dir / "graph_qwen_embeddings.npz") as data:
        assert data["values"].shape == (4, 1024) and not data["values"].any()
    with np.load(context.representations_dir / "meta_embeddings.npz") as data:
        assert not data["values"][1].any()
    assert original == {p: p.read_bytes() for p in original}


def test_missing_summary_does_not_fall_back_to_another_arm(summaries):
    context = summaries
    cohort = context.require_ready_cohort()
    cid = cohort["catalog"][0]["content_id"]
    (context.summary_arm_dir("graph_qwen") / f"{cid}.json").unlink()
    assert (context.summary_arm_dir("graph_gemini") / f"{cid}.json").is_file()
    docs = documents_for_arm(context, cohort, registry(context.config)["graph_qwen"])
    assert docs[0]["text"] == "" and docs[0]["status"] == "missing"
    assert all(row["text"] for row in docs[1:])


def test_tampered_or_interrupted_selection_is_not_readable(summaries, monkeypatch):
    context = summaries
    prepare_validation_cohort(context)
    events_path = cohort_directory(context) / "events.jsonl"
    events = read_jsonl(events_path)
    events[0]["source_event_id"] = 99999
    write_jsonl(events_path, events)
    with pytest.raises(RuntimeError, match="invalid"):
        load_validation_cohort(context, verify_current=False)
    prepare_validation_cohort(context)
    events = read_jsonl(context.cohort_dir / "events.jsonl")
    events[0]["timestamp"] += 1
    write_jsonl(context.cohort_dir / "events.jsonl", events)
    from validation import selection

    original = selection.atomic_write_json

    def interrupt(path, value, **kwargs):
        if path.name == "manifest.json":
            raise OSError("interrupted publish")
        return original(path, value, **kwargs)

    monkeypatch.setattr(selection, "atomic_write_json", interrupt)
    with pytest.raises(OSError, match="interrupted publish"):
        prepare_validation_cohort(context)
    with pytest.raises(RuntimeError, match="invalid"):
        load_validation_cohort(context, verify_current=False)
