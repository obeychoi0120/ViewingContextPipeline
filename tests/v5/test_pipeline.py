from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest

from arm_registry import select_arms
from pipeline_runtime import read_json, write_json
from extraction.steps import extract_graph_scenes, extract_description_scenes, summarize_graph
from extraction.structured_output import validate_graph_structure, OutputValidationError
from extraction.summary_validation import inspect_summary
from validation.steps import embed_representations
from validation.representation_provenance import read_state
from validation.representation_checks import verify_representations

# pytest's repository conftest supplies fixtures; constants are intentionally local.
GRAPH = {
    "entities": [
        {"id": "p1", "name": "person", "attributes": ["red jacket"]},
        {"id": "p2", "name": "person", "attributes": ["blue shirt"]},
    ],
    "relations": [{"subject_id": "p1", "predicate": "looking at", "object_id": "p2"}],
    "context": [],
}


def generate_all(context):
    from extraction.steps import summarize_description

    for model in ("qwen", "gemini"):
        extract_description_scenes(context, model=model, schema="prompts/scene_description_v2.md")
        extract_graph_scenes(context, model=model, schema="prompts/scene_graph_v3.md")
        summarize_description(context, model="qwen", source=model, schema="prompts/summary_description_v4.md")
        summarize_graph(context, model="qwen", source=model, schema="prompts/summary_graph_v4.md")


def test_default_flow_and_artifact_lifecycle(ready_context, fake_models):
    context = ready_context
    generate_all(context)
    result = embed_representations(context, summary_source="qwen")
    assert set(result["generated_arms"]) == set(select_arms(context.config))
    assert len(result["generated_arms"]) == 5
    assert {p.name for p in context.run_root.iterdir()} == {"extraction", "validation"}
    assert context.keyframes_dir.parent == context.run_root.parent.parent / "preparation"
    for pattern in (
        ".recovery",
        ".pending",
        ".checkpoints",
        ".dirty",
        ".changed",
        "media_preflight.json",
        "metadata_missing.json",
    ):
        assert not list(context.run_root.rglob(pattern))
    assert not list((context.run_root / "extraction").rglob(".inputs"))
    assert {p.name for p in (context.run_root / "extraction").iterdir()} == {"description", "graph"}
    calls = len(fake_models)
    generate_all(context)
    assert len(fake_models) == calls
    assert embed_representations(context, summary_source="qwen")["generated_arms"] == []
    with np.load(context.representations_dir / "metadata_embeddings.npz") as values:
        assert np.all(values["values"][1] == 0)
    for name in select_arms(context.config):
        state = read_state(context, name)
        assert state["sources"] and state["truncation"]["truncated_count"] == 0


def test_scene_prompt_changes_reuse_success_without_changing_arm(ready_context, fake_models):
    context = ready_context
    schema = "prompts/scene_graph_v3.md"
    extract_graph_scenes(context, model="qwen", schema=schema)
    count = len(fake_models)
    extract_graph_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count
    path = context.prompt_path(schema)
    path.write_text(path.read_text() + "\nPreserve direction carefully.")
    extract_graph_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count
    summarize_graph(context, model="qwen", source="qwen", schema="prompts/summary_graph_v4.md")
    count = len(fake_models)
    prompt = context.prompt_path("prompts/summary_graph_v4.md")
    prompt.write_text(prompt.read_text() + "\nUse concise prose.")
    summarize_graph(context, model="qwen", source="qwen", schema=prompt)
    assert len(fake_models) == count
    name = "graph_qwen"
    context.config["protocol"]["arms"] = [name]
    embed_representations(context, summary_source="qwen", target=[name])
    replacement = context.root / "prompts/scene_graph_v99.md"
    replacement.write_text(path.read_text() + "\nPreserve all visible attributes.")
    extract_graph_scenes(context, model="qwen", schema=replacement)
    assert embed_representations(context, summary_source="qwen", target=[name])["generated_arms"] == []
    summarize_graph(context, model="qwen", source="qwen", schema=prompt)
    doc = read_json(next(context.graph_summary_dir("qwen").glob("*.json")))
    assert doc["arm"] == "graph_qwen"
    assert "graph_version" not in doc["provenance"]
    assert embed_representations(context, summary_source="qwen", target=[name])["generated_arms"] == []


def test_missing_and_raw_are_empty_but_corruption_is_an_error(ready_context, fake_models):
    from validation.selection import load_validation_cohort
    context = ready_context
    generate_all(context)
    path = sorted(context.description_summary_dir("gemini").glob("*.json"))[0]
    native = read_json(path)
    for contents in (None, {**native, "status": "raw_fallback", "violations": ["empty"]}, "{broken"):
        if contents is None:
            path.unlink()
        elif isinstance(contents, dict):
            write_json(path, contents)
        else:
            path.write_text(contents)
        if isinstance(contents, str):
            with pytest.raises((ValueError, RuntimeError)):
                embed_representations(context, summary_source="qwen", target=["desc_gemini"])
            continue
        embed_representations(context, summary_source="qwen", target=["desc_gemini"])
        cohort = load_validation_cohort(context)
        assert len(cohort["catalog"]) == 4
        assert not cohort["excluded"]
        assert all(r["actual_arm"] == "desc_gemini" for r in read_state(context, "desc_gemini")["sources"])
    write_json(path, native)
    with pytest.raises(RuntimeError, match="stale"):
        verify_representations(context, arms={"desc_gemini": "desc_gemini"})
    embed_representations(context, summary_source="qwen", target=["desc_gemini"])
    assert len(load_validation_cohort(context)["catalog"]) == 4


@pytest.mark.parametrize("source", ["qwen", "gemini"])
@pytest.mark.parametrize("representation", ["graph", "description"])
@pytest.mark.parametrize("length", [20, 200, 201])
def test_summary_word_count_is_prompt_only_and_cached_resume(
    ready_context, fake_models, monkeypatch, source, representation, length
):
    import extraction.steps as steps

    context = ready_context
    extract = getattr(steps, f"extract_{representation}_scenes")
    extract(context, model=source,
            schema=f"prompts/scene_{representation}_v{'3' if representation == 'graph' else '2'}.md")
    summarize = getattr(steps, f"summarize_{representation}")
    options = {"source": source, "schema": f"prompts/summary_{representation}_v4.md"}
    directory = context.summary_dir(representation, source, "qwen")
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                assert task.structured_output is None
                assert task.max_new_tokens == context.config["extraction"][representation]["summary_max_new_tokens"]
                callback(task.task_id, "word " * length)
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    assert summarize(context, model="qwen", **options)["failure_count"] == 0
    assert len(calls) == 1
    for path in directory.glob("*.json"):
        doc = read_json(path)
        assert doc["status"] == "complete" and doc["word_count"] == length
        assert doc["text"] == " ".join(["word"] * length)
        assert doc["violations"] == [] and doc["correction_count"] == 0
    assert not (directory / "failures.jsonl").exists()
    assert summarize(context, model="qwen", **options)["failure_count"] == 0
    assert len(calls) == 1
    arm = f"{'graph' if representation == 'graph' else 'desc'}_{source}"
    context.config["protocol"]["arms"] = [arm]
    assert embed_representations(context, summary_source="qwen", target=[arm])["generated_arms"] == [arm]


def test_summary_accepts_multiple_paragraphs_without_correction(ready_context, fake_models, monkeypatch):
    context = ready_context
    extract_graph_scenes(context, model="qwen", schema="prompts/scene_graph_v3.md")
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                callback(task.task_id, "first " * 100 + "\n\n" + "second " * 101)
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    summarize_graph(context, model="qwen", source="qwen", schema="prompts/summary_graph_v4.md")
    assert len(calls) == 1
    doc = read_json(next(context.graph_summary_dir("qwen").glob("*.json")))
    assert doc["status"] == "complete" and doc["correction_count"] == 0
    assert doc["word_count"] == 201
    assert doc["violations"] == []
    assert "\n\n" in doc["text"]
    assert doc["text"].startswith("first")
    context.config["protocol"]["arms"] = ["graph_qwen"]
    embed_representations(context, summary_source="qwen", target=["graph_qwen"])


def test_graph_preserves_ids_and_only_validates_json_shape():
    validate_graph_structure(GRAPH)
    assert GRAPH["relations"][0]["subject_id"] == "p1"
    for mutate in (
        lambda g: g["entities"][1].update(id="p1"),
        lambda g: g["relations"][0].update(object_id="missing"),
    ):
        value = deepcopy(GRAPH)
        mutate(value)
        before = deepcopy(value)
        validate_graph_structure(value)
        assert value == before
    for mutate in (
        lambda g: g["entities"][0].update(attributes="red"),
        lambda g: g["relations"][0].update(object_id=None),
        lambda g: g["entities"][0].update(id=""),
        lambda g: g.pop("entities"),
    ):
        value = deepcopy(GRAPH)
        mutate(value)
        with pytest.raises(OutputValidationError):
            validate_graph_structure(value)
    assert inspect_summary("A person walks.\nAnother waves.")[1] == []
    assert inspect_summary("A person walks.\n\nAnother waves.")[1] == []


@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_graph_id_mismatches_do_not_retry_or_block_downstream(
    ready_context, fake_models, monkeypatch, model,
):
    import json
    from types import SimpleNamespace
    from extraction.scene_storage import read_scene_records
    from validation.diagnosis_scenes import _success_scene_row_issues

    context = ready_context
    graph = deepcopy(GRAPH)
    graph["entities"][1]["id"] = "p1"
    graph["relations"][0].update(subject_id="unregistered", object_id="missing")
    generated = []

    def generate(tasks, callback):
        for task in tasks:
            generated.append(task.task_id)
            callback(task.task_id, json.dumps(graph))
        return {}

    @contextmanager
    def generator(**kwargs):
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            return generate(tasks, lambda task_id, text: callback(SimpleNamespace(
                task_id=task_id, text=text, error=None, response_diagnostics=None)))

    # Restore the ordinary summary fake after exercising scene generation.
    with monkeypatch.context() as patch:
        patch.setattr("extraction.steps.qwen_generator", generator)
        patch.setattr("extraction.steps.GeminiWorkerPool", Pool)
        for _ in range(2):
            result = extract_graph_scenes(context, model=model, schema="prompts/scene_graph_v3.md")
            assert result["failure_count"] == 0
    assert len(generated) == len(set(generated)) == 4
    for path in context.graph_scene_dir(model).glob("*.jsonl"):
        for row in read_scene_records(path):
            assert row["graph"] == graph
            assert "generation" not in row and row["provenance"]["prompt_hash"]
            assert row["semantic_warnings"] == []
            assert _success_scene_row_issues(f"graph_{model}", row, path.stem) == []
    summarize_graph(context, model="qwen", source=model, schema="prompts/summary_graph_v4.md")
    context.config["protocol"]["arms"] = [f"graph_{model}"]
    assert embed_representations(context, summary_source="qwen", target=[f"graph_{model}"])["generated_arms"] == [
        f"graph_{model}"]


def test_changed_model_settings_and_frame_bytes_reuse_success_until_force(ready_context, fake_models):
    context = ready_context
    schema = "prompts/scene_description_v2.md"
    extract_description_scenes(context, model="qwen", schema=schema)
    count = len(fake_models)
    context.config["extraction"]["description"]["scene_max_new_tokens"] = 768
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count
    (context.path("models", "qwen") / "config.json").write_text('{"revision": 2}')
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count
    from PIL import Image

    frame = next(context.keyframes_dir.rglob("*.png"))
    Image.new("RGB", (16, 8), "red").save(frame)
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count
    extract_description_scenes(context, model="qwen", schema=schema, force=True)
    assert len(fake_models) == count + 1
