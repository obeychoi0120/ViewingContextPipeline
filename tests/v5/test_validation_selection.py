"""One full cohort for all validation arms, irrespective of generation outcomes."""

from copy import deepcopy
from queue import Queue

import numpy as np
import pytest

from arm_registry import select_arms
from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl
from extraction.summary_validation import SUMMARY_SCHEMA_VERSION
from validation.representation_checks import verify_representations
from validation.rolling_data import EventTable
from validation.selection import (
    build_selection,
    cohort_directory,
    load_validation_cohort,
    prepare_validation_cohort,
    training_signature,
)
from validation.steps import embed_representations, validation_config


@pytest.fixture
def summaries(ready_context, fake_models):
    ctx = ready_context
    for model in ("qwen", "gemini"):
        for name, arm in select_arms(ctx.config).items():
            if name == "metadata":
                continue
            for row in ctx.require_ready_cohort()["catalog"]:
                path = (
                    ctx.summary_dir(arm.representation, arm.model, model)
                    / f"{row['content_id']}.json"
                )
                write_json(
                    path,
                    {
                        "schema_version": SUMMARY_SCHEMA_VERSION,
                        "content_id": row["content_id"],
                        "arm": name,
                        "scene_count": 1,
                        "status": "complete",
                        "text": "A person waves.",
                        "word_count": 3,
                        "violations": [],
                        "correction_count": 0,
                        "provenance": {
                            "summary_model": model,
                            "arm": name,
                            "representation": arm.representation,
                        },
                    },
                )
    return ctx


def summary_path(ctx, arm_name, index, model="gemini"):
    arm = select_arms(ctx.config)[arm_name]
    cid = ctx.require_ready_cohort()["catalog"][index]["content_id"]
    return ctx.summary_dir(arm.representation, arm.model, model) / f"{cid}.json"


def test_full_catalog_applies_to_all_targets_and_preserves_shared_data(summaries):
    ctx = summaries
    original = {p: p.read_bytes() for p in ctx.cohort_dir.iterdir() if p.is_file()}
    summary_path(ctx, "desc_gemini", 0).unlink()
    path = summary_path(ctx, "graph_qwen", 2)
    write_json(path, {**read_json(path), "status": "raw_fallback", "violations": ["max_tokens"]})
    embed_representations(ctx, summary_source="gemini", target=["metadata"])
    cohort = load_validation_cohort(ctx)
    assert [r["item_id"] for r in cohort["catalog"]] == ["1", "2", "3", "4"]
    assert len(cohort["excluded"]) == 0
    assert cohort["manifest"]["statistics"]["removed_event_count"] == 0
    assert cohort["events"] == read_jsonl(ctx.cohort_dir / "events.jsonl")
    table = EventTable(cohort["events"])
    assert table.items == ["1", "2", "3", "4"]
    assert all(r["event_id"] == i for i, r in enumerate(table.rows))
    for i in range(len(table.rows)):
        assert all(1 <= item <= 4 for item in table.history(i))
    with np.load(ctx.representations_dir / "metadata_embeddings.npz") as arrays:
        assert arrays["values"].shape == (4, 1024)
        assert not arrays["values"][1].any()
    embed_representations(ctx, summary_source="gemini", target=["desc_qwen"])
    assert load_validation_cohort(ctx)["manifest"] == cohort["manifest"]
    verify_representations(ctx, arms=["metadata", "desc_qwen"])
    assert original == {p: p.read_bytes() for p in original}


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("missing", "missing"),
        ("invalid", "invalid"),
        ("raw", "raw_fallback"),
        ("failure", "failed"),
        ("wrong_model", "invalid"),
    ],
)
def test_empty_representations_and_no_cross_arm_fallback(summaries, damage, reason):
    path = summary_path(summaries, "graph_gemini", 0)
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("{broken")
    elif damage == "raw":
        write_json(path, {**read_json(path), "status": "raw_fallback", "violations": []})
    elif damage == "wrong_model":
        doc = read_json(path)
        doc["provenance"]["summary_model"] = "qwen"
        write_json(path, doc)
    else:
        write_jsonl(
            path.parent / "failures.jsonl",
            [
                {
                    "content_id": path.stem,
                    "summary_model": "gemini",
                    "raw_output": "",
                    "error": "blocked",
                }
            ],
        )
    cohort = build_selection(summaries, "gemini")
    assert len(cohort["catalog"]) == 4 and not cohort["excluded"]
    from validation.representation_inputs import documents_for_arm
    arm = select_arms(summaries.config)["graph_gemini"]
    if damage in {"invalid", "wrong_model"}:
        with pytest.raises((ValueError, RuntimeError)):
            documents_for_arm(summaries, cohort, arm, summary_source="gemini")
    else:
        docs = documents_for_arm(summaries, cohort, arm, summary_source="gemini")
        assert docs[0]["text"] == ""
        assert docs[0]["status"] == ("missing" if damage == "missing" else "failed")
        assert all(d["text"] for d in docs[1:])
    assert summary_path(summaries, "graph_qwen", 0).is_file()


def test_recovery_deletion_source_changes_and_partial_cache_invalidation(summaries):
    ctx = summaries
    path = summary_path(ctx, "desc_gemini", 0)
    saved = read_json(path)
    path.unlink()
    embed_representations(ctx, summary_source="gemini")
    old = load_validation_cohort(ctx)["manifest"]["selection_hash"]
    write_json(path, saved)
    assert load_validation_cohort(ctx)["manifest"]["selection_hash"] == old
    with pytest.raises(RuntimeError, match="stale representation inputs"):
        verify_representations(ctx, arms=["desc_gemini"])
    verify_representations(ctx, arms=["metadata", "desc_qwen", "graph_qwen", "graph_gemini"])
    result = embed_representations(ctx, summary_source="gemini")
    assert result["generated_arms"] == ["desc_gemini"]
    embed_representations(ctx, summary_source="qwen", target=["metadata"])
    verify_representations(ctx, arms=["desc_qwen"])
    embed_representations(ctx, summary_source="qwen")
    summary_path(ctx, "graph_qwen", 0, "qwen").unlink()
    verify_representations(ctx, arms=["metadata"])


def test_missing_manifest_metadata_only_and_outside_target(ready_context, fake_models):
    ctx = ready_context
    with pytest.raises(RuntimeError, match="rerun embed"):
        load_validation_cohort(ctx)
    ctx.config["protocol"]["arms"] = ["metadata"]
    with pytest.raises(ValueError, match="protocol.arms"):
        embed_representations(ctx, summary_source="gemini", target=["desc_qwen"])
    assert not cohort_directory(ctx).exists()
    assert embed_representations(ctx, summary_source="gemini")["content_count"] == 4
    assert not load_validation_cohort(ctx)["excluded"]


def test_fixed_dates_and_history_survive_generation_failures(summaries, monkeypatch):
    ctx = summaries
    source = deepcopy(ctx.require_ready_cohort())
    events = [
        {"event_id": 0, "user_id": "1", "item_id": "1", "timestamp": 1},
        {"event_id": 1, "user_id": "1", "item_id": "2", "timestamp": 11},
        {"event_id": 2, "user_id": "2", "item_id": "2", "timestamp": 1},
        {"event_id": 3, "user_id": "2", "item_id": "4", "timestamp": 2},
        {"event_id": 4, "user_id": "2", "item_id": "2", "timestamp": 11},
        {"event_id": 5, "user_id": "2", "item_id": "1", "timestamp": 100000000},
    ]
    bounds = {"start_ms": None, "end_ms": 10}
    source["plan"]["splits"] = [
        {
            "evaluation_date": "fixed-date",
            "phases": {
                "selection": bounds,
                "validation": bounds,
                "refit": bounds,
                "test": {"start_ms": 10, "end_ms": 20},
            },
        }
    ]
    monkeypatch.setattr(type(ctx), "require_ready_cohort", lambda self: source)
    write_jsonl(ctx.cohort_dir / "events.jsonl", events)
    summary_path(ctx, "graph_gemini", 0).unlink()
    # Only items present in source events belong in this synthetic catalog.
    source["catalog"] = [r for r in source["catalog"] if r["item_id"] != "3"]
    source["metadata_titles"] = [r for r in source["metadata_titles"] if r["item_id"] != "3"]
    cohort = build_selection(ctx, "gemini")
    assert cohort["plan"]["splits"][0]["evaluation_date"] == "fixed-date"
    assert cohort["plan"]["eligible_test_count"] == 2
    assert cohort["manifest"]["statistics"]["lost_history_test_event_count"] == 0
    assert cohort["plan"]["splits"][0]["phases"]["test"]["end_ms"] == 20


def test_all_missing_summaries_preserve_the_full_cohort(summaries):
    ctx = summaries
    for index in (0, 1):
        summary_path(ctx, "graph_gemini", index).unlink()
    assert len(build_selection(ctx, "gemini")["catalog"]) == 4
    for index in (2, 3):
        summary_path(ctx, "graph_gemini", index).unlink()
    assert len(build_selection(ctx, "gemini")["catalog"]) == 4
    embed_representations(ctx, summary_source="gemini", target=["graph_gemini"])
    with np.load(ctx.representations_dir / "graph_gemini_embeddings.npz") as data:
        assert not data["values"].any()


def test_worker_uses_same_full_table_and_rejects_changed_selection(summaries, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "validation.model.torch",
        SimpleNamespace(
            set_num_threads=lambda n: None, device=lambda name: SimpleNamespace(type=name)
        ),
    )
    from validation.rolling_workers import _consume_combinations
    from validation.rolling_recommendation import prepare_split

    ctx = summaries
    summary_path(ctx, "desc_gemini", 0).unlink()
    cohort = prepare_validation_cohort(ctx, "gemini")
    config = validation_config(ctx)
    split = cohort["plan"]["splits"][0]
    identity = {"training_input_hash": training_signature(ctx, cohort, config)}
    expected = prepare_split(EventTable(cohort["events"]), split)
    observed = []

    def run(context, config, table, worker_split, worker_identity, branch, prepared, device):
        assert table.rows == cohort["events"]
        assert worker_split == split
        for phase in expected[0]:
            np.testing.assert_array_equal(prepared[0][phase], expected[0][phase])
        observed.append(worker_identity)

    monkeypatch.setattr("validation.rolling_recommendation.run_combination", run)
    jobs, results = Queue(), Queue()
    jobs.put((split, identity, "metadata"))
    jobs.put(None)
    _consume_combinations(ctx, "cpu", jobs, results)
    assert results.get_nowait() == ("complete", identity)
    assert observed == [identity]
    jobs.put((split, {"training_input_hash": "stale"}, "metadata"))
    jobs.put(None)
    _consume_combinations(ctx, "cpu", jobs, results)
    status, error = results.get_nowait()
    assert status == "error" and "selection changed" in error


def test_diagnosis_reports_full_selection_and_rejects_corrupt_scenes(
    ready_context, fake_models, monkeypatch
):
    from test_pipeline import generate_all
    from validation.rolling_diagnosis import diagnose

    ctx = ready_context
    generate_all(ctx)
    path = summary_path(ctx, "desc_gemini", 0, "qwen")
    path.unlink()
    # Missing representations do not hide malformed scene artifacts from diagnosis.
    (ctx.description_scene_dir("gemini") / f"{path.stem}.jsonl").write_text("broken")
    write_jsonl(
        ctx.description_scene_dir("gemini") / "failures" / f"{path.stem}.jsonl", [{"bad": True}]
    )
    embed_representations(ctx, summary_source="qwen")
    monkeypatch.setattr(
        "validation.rolling_diagnosis.collect_metrics",
        lambda *a, **k: (
            np.ones((3, 7, 5)),
            np.ones((3, 7)),
            {"means": {"metadata": {"HR@10": 0.1}}},
        ),
    )
    with pytest.raises(RuntimeError, match="diagnosis failed"):
        diagnose(ctx)
    doc = read_json(ctx.diagnosis_path)
    assert doc["selection"]["statistics"]["excluded_item_count"] == 0
    assert "included_item_ids" not in doc["selection"]
    assert doc["runtime_decision"]["errors"]


def test_tampered_or_interrupted_selection_is_not_readable(summaries, monkeypatch):
    ctx = summaries
    prepare_validation_cohort(ctx, "gemini")
    events_path = cohort_directory(ctx) / "events.jsonl"
    events = read_jsonl(events_path)
    events[0]["source_event_id"] = 99999
    write_jsonl(events_path, events)
    with pytest.raises(RuntimeError, match="invalid"):
        load_validation_cohort(ctx, verify_current=False)
    prepare_validation_cohort(ctx, "gemini")
    events = read_jsonl(ctx.cohort_dir / "events.jsonl")
    events[0]["timestamp"] += 1
    write_jsonl(ctx.cohort_dir / "events.jsonl", events)
    original = __import__("validation.selection", fromlist=["atomic_write_json"]).atomic_write_json

    def interrupt(path, value, **kwargs):
        if path.name == "manifest.json":
            raise OSError("interrupted publish")
        return original(path, value, **kwargs)

    monkeypatch.setattr("validation.selection.atomic_write_json", interrupt)
    with pytest.raises(OSError):
        prepare_validation_cohort(ctx, "gemini")
    with pytest.raises(RuntimeError, match="invalid"):
        load_validation_cohort(ctx, verify_current=False)
