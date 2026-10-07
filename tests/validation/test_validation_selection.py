"""Current arms share the full catalog even when generation is incomplete."""

import numpy as np
import pytest

from arm_registry import registry
from artifact_io import read_json, read_jsonl, write_jsonl
from validation.leave_one_out_data import EventTable
from validation.selection import cohort_directory, load_validation_cohort, prepare_validation_cohort
from validation.steps import embed_representations


@pytest.fixture
def summaries(ready_context, fake_models, generate_all):
    generate_all(ready_context)
    return ready_context


def test_full_catalog_applies_to_all_targets_and_preserves_shared_data(summaries):
    context = summaries
    legacy = context.run_root / "validation" / "recommendations" / "text" / "2022-09-10" / "seed_42" / "meta"
    legacy.mkdir(parents=True)
    for name in ("sasrec.pt", "training.json", "complete.json"):
        (legacy / name).write_bytes(b"preserve original rolling artifact")
    legacy_shared = context.cohort_dir.parent / "cohort_plan.json"
    legacy_shared.write_bytes(b"preserve original rolling cohort")
    protected = {p: p.read_bytes() for p in (*legacy.iterdir(), legacy_shared)}
    original = {p: p.read_bytes() for p in context.cohort_dir.iterdir() if p.is_file()}
    for path in context.summary_arm_dir("graph_qwen").glob("*.json"):
        path.unlink()
    embed_representations(context, target=list(registry(context.config)))
    cohort = load_validation_cohort(context)
    assert len(cohort["catalog"]) == 4 and not cohort["excluded"]
    assert [r["source_event_id"] for r in cohort["events"]] == [u * 24 + i for u in range(3) for i in range(11, 24)]
    assert cohort["plan"]["eligible_test_count"] == 3
    table = EventTable(cohort["events"])
    assert table.items == ["1", "2", "3", "4"]
    assert len(table.rows) == 39
    with np.load(context.representations_dir / "graph_qwen_embeddings.npz") as data:
        assert data["video_values"].shape == (4, 1024) and not data["video_values"].any()
    with np.load(context.representations_dir / "meta_embeddings.npz") as data:
        assert not data["title_values"][1].any()
    assert original == {p: p.read_bytes() for p in original}
    assert protected == {p: p.read_bytes() for p in protected}
    report = read_json(context.run_root / "reports" / "leave_one_out_cohort.json")
    assert report["source_counts"]["interaction_count"] == 72
    assert report["selected_counts"]["interaction_count"] == 39


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
    events[-1]["timestamp"] += 1
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
