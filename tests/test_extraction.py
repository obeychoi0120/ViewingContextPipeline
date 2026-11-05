import pytest

from artifact_io import read_jsonl
from extraction.graph_warnings import MISSING_REQUIRED, PARSE_FAILED
from extraction.qwen_runtime import QwenRuntime
from extraction.scene_executor import graph_scene_result, run_qwen_scenes
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import parse_or_repair_graph
from extraction.steps import prepare_scene_batch
from extraction.step_support import minimal_graph_records, scene_generation_rows
from helpers import (
    GRAPH_TEXT,
    NAMED_GRAPH_TEXT,
    FakeGenerator,
    failure_row,
    graph,
    quiet_progress,
    ready_cohort,
    scene_row,
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


@pytest.mark.parametrize("failure_count, expected_exit", [(0, 0), (1, 1)])
def test_extraction_cli_reports_incomplete_outputs(
    context, monkeypatch, capsys, failure_count, expected_exit,
):
    from extraction import cli

    monkeypatch.setattr(cli.RunContext, "load", lambda *args, **kwargs: context)
    monkeypatch.setitem(
        cli.STEP_HANDLERS,
        "extract-graph-scenes",
        lambda *args, **kwargs: {"content_count": 10, "failure_count": failure_count},
    )
    status = cli.main([
        "extract-graph-scenes", "--run-id", context.run_id,
        "--dataset", context.dataset, "--schema", "prompts/scene_graph_v9.md",
        "--model", "qwen3vl", "--arm", "graph_qwen3vl",
    ])
    assert status == expected_exit
    assert ("[FAILED]" in capsys.readouterr().err) == bool(failure_count)


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


@pytest.mark.parametrize("representation", ["graph", "description"])
@pytest.mark.parametrize("model", ["qwen3vl", "qwen3.5", "gemini"])
def test_scene_sources_extract_reuse_and_retry(
    context, monkeypatch, representation, model,
):
    from extraction.backends.gemini_workers import GeminiGenerationOutcome
    from extraction.steps import extract_description_scenes, extract_graph_scenes
    from helpers import install_doubles
    from preparation.input_data import prepare_input_data

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    graph_source = representation == "graph"
    generator = FakeGenerator(GRAPH_TEXT if graph_source else "A person cuts a tomato.")

    class GeminiPool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, complete, **kwargs):
            for task in tasks:
                generator.tasks.append(task)
                complete(GeminiGenerationOutcome(task.task_id, generator.text))

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    monkeypatch.setattr("extraction.steps.GeminiWorkerPool", GeminiPool)
    extract = extract_graph_scenes if graph_source else extract_description_scenes
    options = dict(
        model=model,
        arm=f"{'graph' if graph_source else 'desc'}_{model}",
        schema="prompts/scene_graph_v9.md" if graph_source else "prompts/scene_description_v3.md",
    )
    assert extract(context, **options)["failure_count"] == 0
    assert generator.tasks
    for task in generator.tasks:
        assert task.max_new_tokens == context.config["extraction"][representation][model]["scene_max_new_tokens"]
        assert "TITLE_SECRET" not in task.prompt
        if model != "gemini":
            assert task.presence_penalty == 1.5
            assert task.repetition_penalty == (1.05 if graph_source else 1.0)
    generator.tasks.clear()
    assert extract(context, **options)["failure_count"] == 0
    assert not generator.tasks

    path = next(context.scene_arm_dir(options["arm"]).glob("*.jsonl"))
    rows = read_jsonl(path)
    rows[0] = failure_row(path.stem, rows[0]["scene_idx"])
    write_scene_records(path, rows)
    assert extract(context, **options)["failure_count"] == 0
    assert len(generator.tasks) == 1
    assert all("error" not in row for row in read_scene_records(path, include_failures=True))
