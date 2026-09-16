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
from validation.representation_provenance import read_state, recommendation_identity
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
        extract_description_scenes(context, model=model, schema="prompts/description_scene_v2.md")
        extract_graph_scenes(context, model=model, schema="prompts/graph_scene_v3.md")
        summarize_description(context, source=model, schema="prompts/description_summary_v4.md")
        summarize_graph(context, source=model, schema="prompts/graph_summary_v4.md")


def test_default_flow_and_artifact_lifecycle(ready_context, fake_models):
    context = ready_context
    generate_all(context)
    result = embed_representations(context)
    assert set(result["generated_arms"]) == set(select_arms(context.config))
    assert len(result["generated_arms"]) == 5
    assert {p.name for p in context.run_root.iterdir()} == {"cohort", "extraction", "validation"}
    assert context.keyframes_dir.parent == context.run_root.parent.parent
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
    assert embed_representations(context)["generated_arms"] == []
    with np.load(context.representations_dir / "metadata_embeddings.npz") as values:
        assert np.all(values["values"][1] == 0)
    for name in select_arms(context.config):
        state = read_state(context, name)
        assert state["sources"] and state["truncation"]["truncated_count"] == 0


def test_prompt_changes_invalidate_cache_without_changing_arm(ready_context, fake_models):
    context = ready_context
    schema = "prompts/graph_scene_v3.md"
    extract_graph_scenes(context, model="qwen", schema=schema)
    count = len(fake_models)
    extract_graph_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count
    path = context.prompt_path(schema)
    path.write_text(path.read_text() + "\nPreserve direction carefully.")
    extract_graph_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) > count
    summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md")
    count = len(fake_models)
    prompt = context.prompt_path("prompts/graph_summary_v4.md")
    prompt.write_text(prompt.read_text() + "\nUse concise prose.")
    summarize_graph(context, source="qwen", schema=prompt)
    assert len(fake_models) == count + 1
    name = "graph_qwen"
    embed_representations(context, target=[name])
    replacement = context.root / "prompts/graph_scene_v99.md"
    replacement.write_text(path.read_text() + "\nPreserve all visible attributes.")
    extract_graph_scenes(context, model="qwen", schema=replacement)
    with pytest.raises((ValueError, RuntimeError), match="changed|hash"):
        embed_representations(context, target=[name])
    summarize_graph(context, source="qwen", schema=prompt)
    doc = read_json(next(context.graph_summary_dir("qwen").glob("*.json")))
    assert doc["arm"] == "graph_qwen"
    assert "graph_version" not in doc["provenance"]
    assert embed_representations(context, target=[name])["generated_arms"] == [name]


def test_fallback_missing_only_raw_precedence_and_identical_vectors(ready_context, fake_models):
    context = ready_context
    generate_all(context)
    path = next(context.description_summary_dir("gemini").glob("*.json"))
    native = read_json(path)
    path.unlink()
    embed_representations(context, target=["desc_gemini"])
    before = recommendation_identity(context, "desc_gemini")
    state = read_state(context, "desc_gemini")
    assert any(row["actual_arm"] == "desc_qwen" for row in state["sources"])
    native.update(
        status="raw_fallback", violations=["over_200_words"], text="word " * 201, word_count=201
    )
    write_json(path, native)
    with pytest.raises(RuntimeError, match="stale"):
        verify_representations(context, arms={"desc_gemini": "desc_gemini"})
    assert embed_representations(context, target=["desc_gemini"])["generated_arms"] == [
        "desc_gemini"
    ]
    assert recommendation_identity(context, "desc_gemini") != before
    assert all(
        row["actual_arm"] == "desc_gemini" for row in read_state(context, "desc_gemini")["sources"]
    )
    path.write_text("{broken")
    with pytest.raises(ValueError):
        embed_representations(context, target=["desc_gemini"])
    path.unlink()
    (context.description_summary_dir("qwen") / path.name).unlink()
    with pytest.raises(ValueError, match="missing"):
        embed_representations(context, target=["desc_gemini"])


@pytest.mark.parametrize(
    "length,expected_calls,status",
    [(200, 1, "complete"), (201, 1, "raw_fallback"), (20, 1, "complete")],
)
def test_summary_length_failure_without_correction_and_cached_resume(
    ready_context, fake_models, monkeypatch, length, expected_calls, status
):
    context = ready_context
    extract_graph_scenes(context, model="qwen", schema="prompts/graph_scene_v3.md")
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                assert task.structured_output is None and task.max_new_tokens == 512
                callback(task.task_id, "word " * (length if len(calls) == 1 else 100))
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md")
    assert len(calls) == expected_calls
    for path in context.graph_summary_dir("qwen").glob("*.json"):
        doc = read_json(path)
        assert doc["status"] == status and doc["word_count"] == length
        assert doc["correction_count"] == expected_calls - 1
        assert "person" in calls[0][0].prompt and "relationship" in calls[0][0].prompt
    summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md")
    assert len(calls) == expected_calls


def test_summary_keeps_first_nonempty_raw_without_correction(ready_context, fake_models, monkeypatch):
    context = ready_context
    extract_graph_scenes(context, model="qwen", schema="prompts/graph_scene_v3.md")
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                callback(task.task_id, "first " * 201)
            return {}

        yield generate

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    summarize_graph(context, source="qwen", schema="prompts/graph_summary_v4.md")
    assert len(calls) == 1
    doc = read_json(next(context.graph_summary_dir("qwen").glob("*.json")))
    assert doc["status"] == "raw_fallback" and doc["correction_count"] == 0
    assert doc["word_count"] == 201
    assert doc["text"].startswith("first")
    embed_representations(context, target=["graph_qwen"])


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
        lambda g: g.pop("context"),
    ):
        value = deepcopy(GRAPH)
        mutate(value)
        with pytest.raises(OutputValidationError):
            validate_graph_structure(value)
    assert inspect_summary("A person walks.\nAnother waves.")[1] == []
    assert inspect_summary("A person walks.\n\nAnother waves.")[1] == ["multiple_paragraphs"]


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
            result = extract_graph_scenes(context, model=model, schema="prompts/graph_scene_v3.md")
            assert result["failure_count"] == 0
    assert len(generated) == len(set(generated)) == 4
    for path in context.graph_scene_dir(model).glob("*.jsonl"):
        for row in read_scene_records(path):
            assert row["graph"] == graph
            assert set(row["generation"]) == {"input_key"}
            assert row["semantic_warnings"] == []
            assert _success_scene_row_issues(f"graph_{model}", row, path.stem) == []
    summarize_graph(context, source=model, schema="prompts/graph_summary_v4.md")
    assert embed_representations(context, target=[f"graph_{model}"])["generated_arms"] == [
        f"graph_{model}"]


def test_changed_model_settings_refresh_but_prepared_frame_bytes_are_trusted(ready_context, fake_models):
    context = ready_context
    schema = "prompts/description_scene_v2.md"
    extract_description_scenes(context, model="qwen", schema=schema)
    count = len(fake_models)
    context.config["extraction"]["description"]["scene_max_new_tokens"] = 768
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count + 1
    assert all(t.max_new_tokens == 768 for t in fake_models[-1])
    (context.path("models", "qwen") / "config.json").write_text('{"revision": 2}')
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count + 2
    from PIL import Image

    frame = next(context.keyframes_dir.rglob("*.png"))
    Image.new("RGB", (16, 8), "red").save(frame)
    extract_description_scenes(context, model="qwen", schema=schema)
    assert len(fake_models) == count + 2
    extract_description_scenes(context, model="qwen", schema=schema, force=True)
    assert len(fake_models) == count + 3
