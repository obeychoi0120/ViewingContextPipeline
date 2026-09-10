from contextlib import contextmanager

import numpy as np
import pytest

import extraction.steps as steps
import validation.steps as validation_steps
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.input_tracking import marker
from extraction.raw_output import raw_graph_record, raw_summary_document
from extraction.step_support import write_scene_checkpoint
from extraction.summary_executor import reuse_summary_document
from extraction.summary_validation import SUMMARY_SECTIONS
from pipeline_runtime import RunContext, read_json, read_jsonl, write_json, write_jsonl
from validation.diagnosis_scenes import _scene_arm_contract
from validation.representation_provenance import recommendation_identity
from pipeline_fixtures import context as context


def summary(text="A person walks."):
    return "\n".join(f"{name}: {text if index == 0 else ''}"
                     for index, name in enumerate(SUMMARY_SECTIONS))


def setup_scene(context, monkeypatch):
    context.initialize()
    monkeypatch.setattr(steps, "_visual_rows", lambda _: [{"content_id": "c1"}])
    monkeypatch.setattr(steps, "_video_name_map", lambda _: {})
    rows = [{"scene_idx": 0, "keyframes": [5],
             "task": QwenGenerationTask("c1:0", (), "prompt", 32)}]
    monkeypatch.setattr(steps, "_scene_generation_rows", lambda *_, **__: rows)
    return rows[0]


def test_qwen_graph_raw_then_grammar_summary_and_content_refresh(context, monkeypatch):
    setup_scene(context, monkeypatch)
    context.config["extraction"]["graph_repetition_penalty"] = [1, 1.05]
    context.config["extraction"]["summary_repetition_penalty"] = [1, 1.05]
    calls = []

    @contextmanager
    def graph_generator(**_):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks[0].repetition_penalty)
            assert "json" in tasks[0].structured_output
            callback(tasks[0].task_id, "Raw visible observations")
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", graph_generator)
    steps.extract_graph_scenes(context, model="qwen")
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"
    raw = read_jsonl(scene_path)[0]
    assert raw["status"] == "raw_fallback" and "graph" not in raw
    assert calls == [1, 1.05]
    assert not (scene_path.parent / ".recovery").exists()
    assert not (context.graph_failure_dir("qwen") / "c1.jsonl").exists()
    errors = []
    coverage, successes, valid = _scene_arm_contract(
        "graph_qwen", context.run_root, ["c1"], {("c1", 0)}, errors)
    assert valid and not errors and not successes
    assert coverage["raw_fallback_scene_count"] == 1 and coverage["success_coverage"] == 0

    summary_calls = []

    @contextmanager
    def summary_generator(**_):
        def generate(tasks, callback):
            tasks = list(tasks)
            summary_calls.append(tasks[0])
            assert "grammar" in tasks[0].structured_output
            assert "Raw scene observation" in tasks[0].prompt
            # A fenced answer is repaired without consuming the second penalty.
            callback("c1", "```text\n" + summary() + "\n```")
            return {}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", summary_generator)
    steps.summarize_graph(context, source="qwen")
    output = context.graph_summary_dir("qwen") / "c1.json"
    assert read_json(output)["status"] == "complete" and len(summary_calls) == 1
    assert not (output.parent / ".recovery").exists()
    write_json(output.parent / ".recovery" / "stale.json", {})
    steps.summarize_graph(context, source="qwen")
    assert not (output.parent / ".recovery").exists()
    assert len(summary_calls) == 1
    raw["raw_response"] = "Changed visible observation"
    write_scene_checkpoint(scene_path, context.graph_failure_dir("qwen") / "c1.jsonl", [raw], [])
    steps.summarize_graph(context, source="qwen")
    assert len(summary_calls) == 2
    assert "Changed visible observation" in summary_calls[-1].prompt


def test_raw_summary_is_explicit_and_empty_summary_stays_failed(context, monkeypatch):
    row = setup_scene(context, monkeypatch)
    write_jsonl(context.graph_scene_dir("qwen") / "c1.jsonl", [raw_graph_record(row, "Visible scene")])
    context.config["extraction"]["summary_repetition_penalty"] = [1, 1.1]
    calls = []

    @contextmanager
    def generator(**_):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks[0].repetition_penalty)
            return {"c1": "incomplete text" if len(calls) == 1 else ""}
        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    steps.summarize_graph(context, source="qwen")
    output = context.graph_summary_dir("qwen") / "c1.json"
    document = reuse_summary_document(output, schema_version="graph-video-summary/v3",
                                     content_id="c1", arm="graph_qwen", scene_count=1)
    assert document["status"] == "raw_fallback" and document["text"] == "incomplete text"
    assert "sections" not in document
    steps.summarize_graph(context, source="qwen")
    assert calls == [1, 1.1]
    with pytest.raises(steps.ExtractionStepError):
        steps.summarize_graph(context, source="qwen", force=True)
    # Failed forced regeneration does not overwrite the previously usable result.
    assert read_json(output) == document
    assert (output.parent / ".recovery").is_dir()
    # A failed forced refresh remains pending; a normal rerun starts a new cycle.
    with pytest.raises(steps.ExtractionStepError):
        steps.summarize_graph(context, source="qwen")
    assert calls == [1, 1.1, 1, 1.1, 1, 1.1]


def test_embedding_raw_gemini_precedence_and_only_changed_arm_regenerates(context, monkeypatch):
    catalog = [{"item_id": "1", "content_id": "c1"}]
    monkeypatch.setattr(RunContext, "require_ready_cohort", lambda _: {"catalog": catalog})
    write_jsonl(context.cohort_dir / "metadata_titles.jsonl", [{**catalog[0], "title": "title"}])
    for source in ("qwen", "gemini"):
        write_json(context.graph_summary_dir(source) / "c1.json",
                   raw_summary_document("c1", f"graph_{source}", 1, source + " raw"))
    write_json(context.description_summary_dir / "c1.json",
               raw_summary_document("c1", "description", 1, "description raw"))
    encoded = []

    class Encoder:
        def __init__(self, settings):
            self.dimension = settings.embedding_dim

        def encode(self, texts):
            encoded.append(list(texts))
            return np.full((len(texts), self.dimension), len(texts[0]), dtype=np.float32)

    monkeypatch.setattr("validation.features.BGETextEncoder", Encoder)
    validation_steps.embed_representations(context)
    assert encoded[2] == ["gemini raw"]
    assert read_json(context.representations_dir / "graph_gemini_fallbacks.json") == {"fallbacks": []}
    before = {branch: recommendation_identity(context, branch)
              for branch in ("metadata", "graph_qwen", "graph_gemini", "desc")}
    encoded.clear()
    validation_steps.embed_representations(context)
    assert not encoded
    path = context.graph_summary_dir("qwen") / "c1.json"
    value = read_json(path)
    value["scene_count"] = 2
    write_json(path, value)
    from extraction.input_tracking import mark_changed
    mark_changed(path)
    validation_steps.embed_representations(context)
    assert not encoded and not marker(path).exists()  # Same BGE text, changed summary metadata.
    value["text"] = "changed qwen raw output"
    write_json(path, value)  # A hash detects an edit even without the normal writer's marker.
    validation_steps.embed_representations(context)
    assert encoded == [[value["text"]]]
    assert recommendation_identity(context, "graph_qwen") != before["graph_qwen"]
    for branch in ("metadata", "graph_gemini", "desc"):
        assert recommendation_identity(context, branch) == before[branch]
    assert not marker(path).exists()


def test_scene_checkpoint_recovers_success_failure_pair_after_partial_write(context, monkeypatch):
    import extraction.step_support as support
    path = context.graph_scene_dir("qwen") / "c1.jsonl"
    failure_path = context.graph_failure_dir("qwen") / "c1.jsonl"
    write_jsonl(path, [])
    write_jsonl(failure_path, [{"scene_idx": 0, "keyframes": [5], "error": "old"}])
    record = raw_graph_record({"scene_idx": 0, "keyframes": [5]}, "raw")
    original = support.write_failure_jsonl
    monkeypatch.setattr(support, "write_failure_jsonl", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        support.write_scene_checkpoint(path, failure_path, [record], [])
    monkeypatch.setattr(support, "write_failure_jsonl", original)
    support.restore_scene_checkpoint(path, failure_path)
    assert read_jsonl(path) == [record] and not failure_path.exists()


def test_mixed_scene_summary_and_recovery_diagnosis(context, monkeypatch):
    row = setup_scene(context, monkeypatch)
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"
    records = [raw_graph_record({**row, "scene_idx": 2, "keyframes": [65]}, "raw observation"),
               {"scene_idx": 0, "keyframes": [5], "graph": {"setting_context": "indoor"},
                "parse_mode": "native", "semantic_warnings": []}]
    write_jsonl(scene_path, records)

    @contextmanager
    def generator(**_):
        def generate(tasks, callback):
            tasks = list(tasks)
            assert tasks[0].prompt.index("Scene 0") < tasks[0].prompt.index("Scene 2")
            assert "raw observation" in tasks[0].prompt and "indoor" in tasks[0].prompt
            return {"c1": "```\n" + summary() + "\n```"}
        yield generate
    monkeypatch.setattr(steps, "qwen_generator", generator)
    steps.summarize_graph(context, source="qwen")
    document = read_json(context.graph_summary_dir("qwen") / "c1.json")
    assert document["scene_count"] == 2
    from extraction.recovery_report import recovery_report
    report = recovery_report(context.run_root)
    assert report["graph/qwen/summaries"]["counts"] == {}
    assert report["graph/qwen/summaries"]["current_cycle_attempts"] == 0
    assert report["graph/qwen/summaries"]["summary_inputs"] == {
        "scene_count": 2, "normal_scene_count": 1, "raw_scene_count": 1, "summaries_with_raw_input": 1}
    assert report["graph/qwen/scenes"]["artifact_counts"] == {"raw_fallback": 1, "legacy_unknown": 1}


def test_gemini_raw_preserves_original_and_errors_never_become_raw(context, monkeypatch):
    row = setup_scene(context, monkeypatch)
    diagnostic = {"candidates": [{"finish_reason": "MAX_TOKENS"}],
                  "usage_metadata": {"candidates_token_count": 8}}
    calls = []
    class Pool:
        def __init__(self, *_, **__):
            pass
        def generate(self, tasks, callback, **_):
            calls.append([t.task_id for t in tasks])
            callback(steps.GeminiGenerationOutcome(tasks[0].task_id, "  raw observation \n",
                                                  response_diagnostics=diagnostic))
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    steps.extract_graph_scenes(context, model="gemini")
    output = context.graph_scene_dir("gemini") / "c1.jsonl"
    assert read_jsonl(output) == [raw_graph_record(row, "  raw observation \n")]
    assert not (output.parent / ".recovery").exists()
    steps.extract_graph_scenes(context, model="gemini")
    assert calls == [["c1:0"]]
    def fail(self, tasks, callback, **_):
        callback(steps.GeminiGenerationOutcome(tasks[0].task_id, "partial error", error="HTTP error"))
    monkeypatch.setattr(Pool, "generate", fail)
    assert steps.extract_graph_scenes(context, model="gemini", force=True)["failure_count"] == 1
    assert read_jsonl(output) == []
    assert (output.parent / ".recovery").is_dir()


@pytest.mark.parametrize("source", ["qwen", "gemini", "description"])
def test_force_interrupted_before_first_response_resumes_instead_of_reusing_old_output(context, monkeypatch, source):
    row = setup_scene(context, monkeypatch)
    description = source == "description"
    output = (context.description_scene_dir if description else context.graph_scene_dir(source)) / "c1.jsonl"
    previous = (dict(schema_version="scene-description/v1", content_id="c1", scene_idx=0,
                     keyframes=[5], description="old") if description else raw_graph_record(row, "old"))
    write_jsonl(output, [previous])
    before = output.read_bytes()
    interrupted = True
    calls = []
    @contextmanager
    def generator(**_):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append([t.task_id for t in tasks])
            if interrupted:
                raise RuntimeError("worker failed before response")
            return {tasks[0].task_id: "new observation"}
        yield generate
    class Pool:
        def __init__(self, *_, **__):
            pass
        def generate(self, tasks, callback, **_):
            with generator() as generate:
                for task_id, text in generate(tasks, None).items():
                    callback(steps.GeminiGenerationOutcome(task_id, text))
    monkeypatch.setattr(steps, "qwen_generator", generator)
    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    def run(**kwargs):
        return (steps.extract_description_scenes(context, **kwargs) if description
                else steps.extract_graph_scenes(context, model=source, **kwargs))
    with pytest.raises(RuntimeError, match="worker failed"):
        run(force=True)
    assert output.read_bytes() == before
    assert (output.parent / ".recovery" / ".force-run").is_file()
    interrupted = False
    run()
    assert len(calls) == 2
    assert "new observation" in str(read_jsonl(output))
    assert not (output.parent / ".recovery").exists()


def test_qwen_pool_is_loaded_only_if_checkpoint_replay_needs_generation(monkeypatch):
    from extraction.summary_executor import qwen_generator
    monkeypatch.setattr("extraction.summary_executor.QwenWorkerPool",
                        lambda *_, **__: pytest.fail("replay should not load a GPU model"))
    with qwen_generator(model_path="unused", gpus=1):
        pass
