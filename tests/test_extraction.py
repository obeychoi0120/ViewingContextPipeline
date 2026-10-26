from copy import deepcopy

import pytest

from artifact_io import read_jsonl, write_jsonl
from extraction.errors import ExtractionStepError
from extraction.graph_warnings import MISSING_REQUIRED, PARSE_FAILED
from extraction.qwen_runtime import QwenRuntime
from extraction.scene_executor import graph_scene_result, run_qwen_scenes
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import parse_or_repair_graph
from extraction.steps import prepare_scene_batch
from extraction.step_support import minimal_graph_records, scene_generation_rows
from extraction.summary_executor import prepare_summary_batch
from extraction.summary_storage import validate_summary_template
from helpers import (
    GRAPH_TEXT,
    NAMED_GRAPH_TEXT,
    FakeGenerator,
    extract_scenes,
    failure_row,
    graph,
    quiet_progress,
    ready_cohort,
    scene_row,
    summary_options,
)


def test_graph_v9_named_references():
    parsed = parse_or_repair_graph(NAMED_GRAPH_TEXT)
    assert parsed.error is None and not parsed.warning
    expected = graph()
    for entity, name in zip(
        expected["entities"], ["person 1", "tomato 2", "knife", "kitchen"], strict=True
    ):
        entity["name"] = name
    for index, entity in enumerate(expected["entities"], 1):
        entity["id"] = f"e{index}"
    expected["actions"][0].update(actor="e1", target="e2", tool="e3", place="e4")
    assert parsed.graph == expected


def test_graph_legacy_id_references():
    parsed = parse_or_repair_graph(GRAPH_TEXT)
    assert parsed.error is None
    assert parsed.parse_mode == "native"
    assert parsed.graph == graph()


def test_invalid_graph_preserves_raw_output():
    invalid = GRAPH_TEXT.split("[Actions]")[0]
    record, failure = graph_scene_result({"scene_idx": 7, "keyframes": [2.5]}, invalid)
    assert failure is None
    assert record["graph"] == invalid
    assert record["parse_mode"] == "text"
    assert record["semantic_warnings"] == [MISSING_REQUIRED]
    assert record["scene_idx"] == 7


def test_undeclared_roles_become_unknown():
    text = GRAPH_TEXT.replace("p1 - cutting - f1; t1; l1", "p1 - cutting - absent; none; unknown")
    parsed = parse_or_repair_graph(text)
    assert parsed.error is None
    assert parsed.parse_mode == "repaired"
    assert parsed.graph["entities"] == graph()["entities"]
    assert parsed.graph["actions"] == [
        dict(actor="p1", action="cutting", target="unknown", tool=None, place="unknown")
    ]
    # A name resembling a generated ID must not accidentally resolve to that ID.
    named = parse_or_repair_graph(NAMED_GRAPH_TEXT.replace("- tomato 2;", "- e1;"))
    assert named.graph["actions"][0]["target"] == "unknown"


def test_compact_scene_storage_roundtrip(tmp_path):
    cid = "video"
    path = tmp_path / f"{cid}.jsonl"
    records = [
        failure_row(cid, 2),
        scene_row(cid),
        {"content_id": cid, "scene_idx": 1, "tokens": 8, "description": "A person cuts."},
    ]
    write_scene_records(path, records)
    assert read_jsonl(path) == sorted(records, key=lambda r: r["scene_idx"])
    restored = read_scene_records(path, include_failures=True)
    assert [r["scene_idx"] for r in restored] == [0, 1, 2]
    assert restored[0]["graph"] == graph() and restored[0]["tokens"] == 19
    assert restored[1]["description"] == "A person cuts."
    assert restored[2] == failure_row(cid, 2)
    assert len(read_scene_records(path)) == 2
    # Replacing an attempt must leave exactly one row for that scene.
    write_scene_records(path, [records[1], {**records[2], "description": "Updated."}, records[0]])
    assert read_jsonl(path)[1]["description"] == "Updated."
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate scene"):
        write_scene_records(path, [records[1], records[1]])
    assert path.read_bytes() == before


def test_scene_resume_selects_pending_only(context):
    cid = "video"
    directory = context.scene_arm_dir("graph_qwen3vl")
    path = directory / f"{cid}.jsonl"
    write_scene_records(
        path, [scene_row(cid), failure_row(cid, 1), scene_row(cid, 2, warnings=[PARSE_FAILED])]
    )
    visual = {"content_id": cid}

    def prepare(visual):
        return [
            {
                "scene_idx": i,
                "keyframes": [i + 0.5],
                "scene_start_seconds": i,
                "scene_end_seconds": i + 1,
            }
            for i in range(4)
        ]

    before = path.read_bytes()
    existing, pending = prepare_scene_batch(
        [visual], directory, prepare, minimal_graph_records, force=False
    )
    assert len(existing[cid]) == 3
    assert [r["scene_idx"] for r in pending[0][1]] == [1, 2, 3]
    assert path.read_bytes() == before
    _, forced = prepare_scene_batch([visual], directory, prepare, minimal_graph_records, force=True)
    assert [r["scene_idx"] for r in forced[0][1]] == [0, 1, 2, 3]


@pytest.mark.torch
def test_truncated_scene_is_failure(context, monkeypatch):
    from preparation.input_data import prepare_input_data
    from extraction.step_support import visual_rows
    from helpers import install_doubles

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    visual = visual_rows(context)[0]
    rows = scene_generation_rows(visual, prompt="observe", max_new_tokens=384)[:1]
    factory = FakeGenerator(finish_reason="length")
    runtime = QwenRuntime()
    directory = context.scene_arm_dir("graph_qwen3vl")
    run_qwen_scenes(
        [(visual, rows)],
        scene_dir=directory,
        model_path=context.path("models", "qwen3vl"),
        generator_factory=factory,
        names={},
        progress=quiet_progress(),
        arm="graph",
        existing_records={},
        runtime=runtime,
    )
    saved = read_jsonl(directory / f"{visual['content_id']}.jsonl")
    assert len(saved) == 1
    assert "truncated" in saved[0]["error"]
    assert saved[0]["raw_output"] == GRAPH_TEXT
    assert saved[0]["tokens"] == 19
    assert "scene_graph" not in saved[0]


def test_summary_prompt_excludes_title(context, monkeypatch):
    extract_scenes(context, monkeypatch)
    options = summary_options(context)
    batch, _ = prepare_summary_batch(context, **options)
    assert batch.next_penalties
    for records, provenance, task in batch.pending.values():
        assert records and "TITLE_SECRET" not in task.prompt
        assert "cutting" in task.prompt
        assert provenance["uses_title"] is False
        assert "english_title" not in provenance
    for template in ("No scene placeholder", "{scenes} {english_title}"):
        with pytest.raises(ValueError):
            validate_summary_template(template, uses_title=False)


def test_summary_reuse_tracks_scene_content(context, monkeypatch):
    from extraction.steps import summarize
    from helpers import FakeGenerator

    extract_scenes(context, monkeypatch)
    monkeypatch.setattr("extraction.steps.qwen_generator", FakeGenerator("A person cuts a tomato."))
    summarize(context, model="qwen3vl", schema="prompts/summary_graph_v9.md", arm="graph_qwen3vl")
    options = summary_options(context)
    cid = options["catalog"][0]["content_id"]
    output = context.summary_arm_dir("graph_qwen3vl") / f"{cid}.json"
    before = output.read_bytes()
    stat = output.stat().st_mtime_ns
    batch, _ = prepare_summary_batch(context, **options)
    assert not batch.next_penalties and cid in batch.documents
    assert output.stat().st_mtime_ns == stat
    source = context.scene_arm_dir("graph_qwen3vl") / f"{cid}.jsonl"
    rows = read_jsonl(source)
    updated = deepcopy(rows)
    updated[0]["scene_graph"]["actions"][0]["action"] = "washing"
    write_jsonl(source, updated)
    with pytest.raises(ExtractionStepError, match="scene input changed"):
        prepare_summary_batch(context, **options)
    assert output.read_bytes() == before
