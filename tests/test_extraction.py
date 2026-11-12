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


def test_graph_parsing_and_invalid_output_preservation():
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

    invalid = GRAPH_TEXT.split("[Actions]")[0]
    record, failure = graph_scene_result({"scene_idx": 7, "keyframes": [2.5]}, invalid)
    assert failure is None
    assert record["graph"] == invalid
    assert record["parse_mode"] == "text"
    assert record["semantic_warnings"] == [MISSING_REQUIRED]
    assert record["scene_idx"] == 7


def test_named_graph_keeps_roles_missing_from_entities():
    text = (
        "[Context]\nmedium: live_action\nformat: narrative\ntopics: hospital visit\n"
        "[Entities]\nman; wearing coat\n"
        "[Actions]\nman - removing - sunglasses; None; hospital room\n"
        "man - holding - sunglasses; none; unknown"
    )
    parsed = parse_or_repair_graph(text)
    assert parsed.error is None and not parsed.warning and parsed.parse_mode == "repaired"
    assert parsed.graph["entities"] == [
        {"id": "e1", "name": "man", "attributes": ["wearing coat"]},
        {"id": "e2", "name": "sunglasses", "attributes": []},
        {"id": "e3", "name": "hospital room", "attributes": []},
    ]
    assert parsed.graph["actions"] == [
        {"actor": "e1", "action": "removing", "target": "e2", "tool": None, "place": "e3"},
        {"actor": "e1", "action": "holding", "target": "e2", "tool": None, "place": "unknown"},
    ]
    # ID-form output is unchanged: an undeclared ID has no name to keep.
    legacy = parse_or_repair_graph(
        "[Context]\nmedium: live_action\nformat: narrative\ntopics: hospital visit\n"
        "[Entities]\np1: man\n[Actions]\np1 - removing - t9; none; unknown"
    )
    assert legacy.graph["entities"] == [{"id": "p1", "name": "man", "attributes": []}]
    assert legacy.graph["actions"][0]["target"] == "unknown"


def test_named_graph_regroups_actions_joined_on_one_line():
    head = "[Context]\nmedium: animation\nformat: narrative\ntopics: dorm\n[Entities]\n"
    names = "woman1; white blouse\nman1\ndoor\nknife\nroom\n[Actions]\n"

    def actions(text):
        parsed = parse_or_repair_graph(head + names + text)
        assert parsed.error is None and parsed.parse_mode == "repaired", parsed.error
        return [tuple(a.values()) for a in parsed.graph["actions"]]

    # e1 woman1, e2 man1, e3 door, e4 knife, e5 room
    assert actions("woman1 - pushing - door; man1 - looking - woman1; man1 - cutting - door") == [
        ("e1", "pushing", "e3", None, "unknown"),
        ("e2", "looking", "e1", None, "unknown"),
        ("e2", "cutting", "e3", None, "unknown"),
    ]
    assert actions("man1 - cutting - door; knife; room; woman1 - leaving\nman1 - waiting - none; knife") == [
        ("e2", "cutting", "e3", "e4", "e5"),
        ("e1", "leaving", "unknown", None, "unknown"),
        ("e2", "waiting", None, "e4", "unknown"),
    ]
    for ambiguous in ("man1 - cutting - door; knife; room; door", "knife; man1 - cutting - door",
                      "man1 - cutting - door - room"):
        assert parse_or_repair_graph(head + names + ambiguous).warning == (PARSE_FAILED,)
    # ID-form output keeps its earlier behavior.
    legacy = head + "p1: woman\nf1: door\n[Actions]\np1 - pushing - f1; p1 - leaving - f1"
    assert parse_or_repair_graph(legacy).warning == (PARSE_FAILED,)


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
    directory = context.scene_arm_dir("graph_qwen3vl_2b")
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
    directory = context.scene_arm_dir("graph_qwen3vl_2b")
    run_qwen_scenes(
        [(visual, rows)],
        scene_dir=directory,
        model_path=context.path("models", "qwen3vl_2b"),
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
@pytest.mark.parametrize("model", ["qwen3vl_2b", "gemini"])
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
        schema="prompts/scene_graph_v9_2b.md" if graph_source else "prompts/scene_description_v3.md",
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


LOOPED_GRAPH_TEXT = (
    "[Context]\nmedium: animation\nformat: narrative\ntopics: forest scene\n"
    "[Entities]\npanda1; white robe; white robe\npanda2; orange robe\njar\n"
    "[Actions]\n"
    + "panda1 - greeting - panda2; none; unknown\npanda2 - lifting - jar; none; unknown\n" * 9
    + "panda1 - greet"
)


def test_truncated_named_graph_is_salvaged_without_inventing_content():
    from extraction.semantic_graph import recover_truncated_graph

    graph = recover_truncated_graph(LOOPED_GRAPH_TEXT).graph
    assert [(e["name"], e["attributes"]) for e in graph["entities"]] == [
        ("panda1", ["white robe"]), ("panda2", ["orange robe"]), ("jar", [])]
    assert [(a["actor"], a["action"], a["target"]) for a in graph["actions"]] == [
        ("e1", "greeting", "e2"), ("e2", "lifting", "e3")]

    head = "[Context]\nmedium: live_action\nformat: narrative\ntopics: army\n[Entities]\n"
    counting = head + "".join(f"man{i}; wearing armor\n" for i in range(1, 31)) + "man31; wea"
    graph = recover_truncated_graph(counting).graph
    assert [e["name"] for e in graph["entities"]] == [f"man{i}" for i in range(1, 9)]
    assert graph["actions"] == []

    joined = head + "woman\nbook\n[Actions]\n" + "woman - reading - book; woman - closing - book; " * 20
    actions = recover_truncated_graph(joined + "woman - rea").graph["actions"]
    assert [a["action"] for a in actions] == ["reading", "closing"]

    # ID-form output, and output cut before any entity, stay failures.
    assert recover_truncated_graph(GRAPH_TEXT + "p1 - cutting - f1; t1; l1\np1 - cut") is None
    assert recover_truncated_graph("[Context]\nmedium: live_action\nformat: narrative\ntopics: a; b") is None
    assert recover_truncated_graph(head + "man1; wea") is None


def test_progress_counts_each_scene_in_exactly_one_outcome():
    from extraction.progress import InferenceProgress
    from tqdm import tqdm
    import io

    with InferenceProgress(total=4, desc="x", unit="scene",
                           progress_factory=lambda **kw: tqdm(file=io.StringIO(), **kw)) as progress:
        progress.complete(task_id="a")
        progress.complete(task_id="b", warning=True)
        progress.complete(task_id="c", recovered=True)
        progress.complete(task_id="d", failed=True, warning=True, recovered=True)
        progress.complete(task_id="d", recovered=True)  # a later pass replaces the outcome
        counts = (progress.success, progress.warning, progress.recovered, progress.failed)
    assert counts == (1, 1, 2, 0) and sum(counts) == 4


def test_truncated_graph_scenes_are_recovered_counted_and_reused(context, monkeypatch):
    from artifact_io import read_json
    from extraction.graph_warnings import TRUNCATION_RECOVERED
    from extraction.steps import extract_graph_scenes
    from helpers import install_doubles
    from preparation.input_data import prepare_input_data

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    generator = FakeGenerator(LOOPED_GRAPH_TEXT, finish_reason="length")
    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    options = dict(model="qwen3vl_2b", arm="graph_qwen3vl_2b", schema="prompts/scene_graph_v9_2b.md")
    assert extract_graph_scenes(context, **options)["failure_count"] == 0
    directory = context.scene_arm_dir(options["arm"])
    stats = read_json(directory / "generation_stats.json")
    assert stats["recovered"] == stats["total"] == len(generator.tasks) > 0
    assert stats["success"] + stats["warning"] + stats["recovered"] + stats["failed"] == stats["total"]
    path = next(directory.glob("*.jsonl"))
    rows = read_jsonl(path)
    assert all(row["warning"] == [TRUNCATION_RECOVERED] and len(row["scene_graph"]["actions"]) == 2
               for row in rows)

    # Recovered scenes are final: a rerun generates nothing, also when --arm is omitted.
    generator.tasks.clear()
    extract_graph_scenes(context, **options)
    extract_graph_scenes(context, model="qwen3vl_2b", schema=options["schema"])
    assert not generator.tasks
    with pytest.raises(ValueError, match="invalid graph generation arm"):
        extract_graph_scenes(context, model="qwen3vl_2b", arm="desc_qwen3vl_2b", schema=options["schema"])

    # A stored truncation failure is salvaged in place instead of being generated again.
    rows[0] = {"content_id": path.stem, "scene_idx": rows[0]["scene_idx"], "tokens": 384,
               "error": "graph: output truncated at token limit", "raw_output": LOOPED_GRAPH_TEXT}
    from artifact_io import atomic_write_jsonl
    atomic_write_jsonl(path, rows, durable=True, sort_keys=False)
    assert extract_graph_scenes(context, **options)["failure_count"] == 0
    assert not generator.tasks
    assert read_jsonl(path)[0]["warning"] == [TRUNCATION_RECOVERED]
    assert read_json(directory / "generation_stats.json")["recovered"] == stats["total"]
