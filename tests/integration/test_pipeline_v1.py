from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml

import extraction.steps as extraction_steps
import validation.steps as validation_steps
from extraction.backends.qwen_workers import QwenGenerationTask
from pipeline_runtime import ConfigError, RunContext, read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def context(tmp_path: Path) -> RunContext:
    data = tmp_path / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    (data / "pairs.tsv").write_text("fixture\n", encoding="utf-8")
    models = tmp_path / "models"
    for name in ("qwen", "bge"):
        (models / name).mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    config["artifacts_root"] = str(tmp_path / "artifacts")
    config["data"] = {
        "videos_dir": str(videos),
        "pairs_tsv": str(data / "pairs.tsv"),
    }
    config["models"] = {
        "qwen": str(models / "qwen"),
        "bge": str(models / "bge"),
        "gemini": {
            "project_id": "test-project",
            "location": "global",
            "model_id": "test-gemini",
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "thinking_level": "low",
            "media_resolution": "MEDIA_RESOLUTION_MEDIUM",
        },
    }
    config["extraction"]["graph"]["summary_prompt"] = str(
        ROOT / config["extraction"]["graph"]["summary_prompt"]
    )
    for key in ("scene_prompt", "summary_prompt"):
        config["extraction"]["description"][key] = str(
            ROOT / config["extraction"]["description"][key]
        )
    (config_dir / "pipeline.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return RunContext.load("test_run", root=tmp_path)


def test_config_contract_remains_fixed(context: RunContext) -> None:
    assert context.config["schema_version"] == "viewing-context-config/v1"
    assert set(context.config["data"]) == {"videos_dir", "pairs_tsv"}
    assert "do_sample" not in context.config["extraction"]["graph"]
    assert "do_sample" not in context.config["extraction"]["description"]
    assert context.config["extraction"]["summary_retry"] == {
        "seeds": [42, 43, 44],
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert set(context.config["validation"]["encoder"]) == {
        "embedding_dim",
        "max_length",
        "batch_size",
    }
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["protocol"]["modality"] = "multimodal"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="protocol.modality"):
        RunContext.load("other", root=context.root)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("temperature", -0.1, "temperature"),
        ("max_output_tokens", 0, "max_output_tokens"),
        ("thinking_level", "minimal", "thinking_level"),
        ("media_resolution", "medium", "media_resolution"),
    ],
)
def test_config_rejects_invalid_gemini_generation_settings(
    context: RunContext,
    key: str,
    value: object,
    message: str,
) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["models"]["gemini"][key] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        RunContext.load("invalid", root=context.root)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("seeds", [42, 42], "seeds"),
        ("temperature", 0.0, "temperature"),
        ("top_p", 1.1, "top_p"),
        ("top_k", 0, "top_k"),
    ],
)
def test_config_rejects_invalid_summary_retry_settings(
    context: RunContext,
    key: str,
    value: object,
    message: str,
) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["extraction"]["summary_retry"][key] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        RunContext.load("invalid-summary-retry", root=context.root)


def test_summary_retry_uses_three_seeded_sampling_attempts() -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "affect_or_topic": "Neutral",
    }
    responses = ["not json", "still not json", "also not json", json.dumps(sections)]
    submitted: list[QwenGenerationTask] = []

    def generate(tasks, callback=None):
        task = tasks[0]
        submitted.append(task)
        text = responses[len(submitted) - 1]
        if callback is not None:
            callback(task.task_id, text)
        return {task.task_id: text}

    completed = []

    def complete(task_id, text):
        completed.append((task_id, extraction_steps.parse_summary_sections(text)))

    retry_count = extraction_steps._generate_summaries_with_retry(
        generate,
        [QwenGenerationTask("c1", (), "prompt", 512)],
        complete,
        {
            "seeds": [42, 43, 44],
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
        },
    )

    assert retry_count == 3
    assert completed == [("c1", sections)]
    assert [task.seed for task in submitted] == [None, 42, 43, 44]
    assert [task.do_sample for task in submitted] == [False, True, True, True]
    for task in submitted[1:]:
        assert (task.temperature, task.top_p, task.top_k) == (0.1, 0.8, 20)


def test_summary_is_completed_from_worker_callback_before_batch_returns() -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "affect_or_topic": "Neutral",
    }
    structured = json.dumps(sections)
    completed = []

    def generate(tasks, callback=None):
        assert callback is not None
        callback("c2", structured)
        assert completed == ["c2"]
        callback("c1", structured)
        return {task.task_id: structured for task in tasks}

    def complete(task_id, text):
        extraction_steps.parse_summary_sections(text)
        completed.append(task_id)

    retry_count = extraction_steps._generate_summaries_with_retry(
        generate,
        [
            QwenGenerationTask("c1", (), "prompt", 512),
            QwenGenerationTask("c2", (), "prompt", 512),
        ],
        complete,
        {
            "seeds": [42, 43, 44],
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
        },
    )

    assert retry_count == 0
    assert completed == ["c2", "c1"]


def test_reused_summary_normalizes_single_line_text_to_section_lines(
    tmp_path: Path,
) -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "affect_or_topic": "Neutral",
    }
    path = tmp_path / "summary.json"
    write_json(
        path,
        {
            "schema_version": "graph-video-summary/v2",
            "content_id": "c1",
            "arm": "graph_qwen",
            "status": "complete",
            "sections": sections,
            "text": "legacy single-line text",
            "scene_count": 1,
        },
    )

    document = extraction_steps._reuse_summary_document(
        path,
        schema_version="graph-video-summary/v2",
        content_id="c1",
        arm="graph_qwen",
        scene_count=1,
    )

    assert document["text"].splitlines() == [
        "Setting and environments: An indoor room.",
        "Main characters and objects: A person.",
        "Chronological events: The person walks.",
        "Relations: The person is inside the room.",
        "Affect or topic: Neutral.",
    ]
    assert json.loads(path.read_text(encoding="utf-8")) == document


def test_runtime_has_no_orchestration_manifest_paths(context: RunContext) -> None:
    assert not hasattr(context, "pipeline_manifest")
    assert not hasattr(context, "stage_manifest")
    assert not hasattr(context, "visual_manifest")
    assert not hasattr(context, "representations_manifest")
    assert not hasattr(context, "recommendations_manifest")


def test_graph_scene_failure_is_recorded_and_stage_continues(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", ("unused.png",), "prompt", 10),
            "scene_idx": index,
            "scene_start_seconds": index * 30,
            "scene_end_seconds": (index + 1) * 30,
            "keyframes": [index * 30 + 5],
            "image_paths": ["unused.png"],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )
    graph = {
        "setting_context": "indoor",
        "entities": [],
        "events": [],
        "static_relations": [],
        "semantic_topics": [],
        "affect": {"subject_ids": [], "valence": "neutral", "arousal": "medium"},
    }

    @contextmanager
    def fake_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {tasks[0].task_id: json.dumps(graph), tasks[1].task_id: "not json"}

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 1
    scenes = read_jsonl(context.graph_scene_dir("qwen") / "c1.jsonl")
    assert len(scenes) == 1
    assert set(scenes[0]) == {
        "scene_idx",
        "keyframes",
        "graph",
        "parse_mode",
        "semantic_warnings",
    }
    assert scenes[0]["parse_mode"] == "native"
    failures = read_jsonl(context.graph_failure_dir("qwen") / "c1.jsonl")
    assert failures[0]["scene_idx"] == 1
    assert failures[0]["failure_kind"] == "json_repair"

    @contextmanager
    def successful_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {task.task_id: json.dumps(graph) for task in tasks}

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", successful_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen", force=True)

    assert result["failure_count"] == 0
    failure_path = context.graph_failure_dir("qwen") / "c1.jsonl"
    assert not failure_path.exists()

    write_jsonl(failure_path, [])

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("completed scenes must be reused")
        yield

    monkeypatch.setattr(extraction_steps, "_qwen_generator", unexpected_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 0
    assert not failure_path.exists()


def test_graph_summary_trusts_directory_and_compacts_legacy_scene(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    monkeypatch.setattr(
        extraction_steps,
        "_visual_rows",
        lambda _context: [{"content_id": "c1"}],
    )
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"
    write_jsonl(
        scene_path,
        [
            {
                "schema_version": "legacy",
                "content_id": "c1",
                "scene_idx": 0,
                "keyframes": [5, 15, 25],
                "image_paths": ["a.png", "b.png", "c.png"],
                "graph_source": "gemini",
                "input_fingerprint": "legacy",
                "graph": {"entities": []},
            }
        ],
    )

    @contextmanager
    def fake_generator(**_kwargs):
        def generate(tasks, callback=None):
            structured = json.dumps(
                {
                    "setting_and_environments": "An indoor setting",
                    "main_characters_and_objects": "A person and an object",
                    "chronological_events": "The person approaches the object",
                    "relations": "The person stands beside the object",
                    "affect_or_topic": "A neutral visible affect",
                }
            )
            results = {task.task_id: structured for task in tasks}
            if callback is not None:
                for task_id, text in results.items():
                    callback(task_id, text)
            return results

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)
    extraction_steps.summarize_graph(context, source="qwen")

    compacted = read_jsonl(scene_path)[0]
    assert set(compacted) == {
        "scene_idx",
        "keyframes",
        "graph",
        "parse_mode",
        "semantic_warnings",
    }
    assert compacted["parse_mode"] == "unknown"
    summary = json.loads(
        (context.graph_summary_dir("qwen") / "c1.json").read_text(encoding="utf-8")
    )
    assert summary == {
        "schema_version": "graph-video-summary/v2",
        "content_id": "c1",
        "arm": "graph_qwen",
        "status": "complete",
        "sections": {
            "setting_and_environments": "An indoor setting",
            "main_characters_and_objects": "A person and an object",
            "chronological_events": "The person approaches the object",
            "relations": "The person stands beside the object",
            "affect_or_topic": "A neutral visible affect",
        },
        "scene_count": 1,
        "text": (
            "Setting and environments: An indoor setting.\n"
            "Main characters and objects: A person and an object.\n"
            "Chronological events: The person approaches the object.\n"
            "Relations: The person stands beside the object.\n"
            "Affect or topic: A neutral visible affect."
        ),
    }


def test_embedding_uses_fixed_files_and_no_manifest(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    content_id = "c1"
    write_jsonl(context.cohort_dir / "catalog.jsonl", [{"content_id": content_id, "item_id": "1"}])
    for source in ("qwen", "gemini"):
        write_json(
            context.graph_summary_dir(source) / f"{content_id}.json",
            {
                "schema_version": "graph-video-summary/v2",
                "content_id": content_id,
                "status": "complete",
                "text": f"{source} graph",
            },
        )
    write_json(
        context.description_summary_dir / f"{content_id}.json",
        {
            "schema_version": "description-video-summary/v2",
            "content_id": content_id,
            "status": "complete",
            "text": "description",
        },
    )
    dimension = validation_steps.validation_config(context).encoder.embedding_dim
    loads: list[object] = []
    encoded_texts: list[list[str]] = []

    class FakeEncoder:
        def __init__(self, settings):
            loads.append(settings)

        def encode(self, texts):
            encoded_texts.append(list(texts))
            return np.ones((len(texts), dimension), dtype=np.float32)

    monkeypatch.setattr(validation_steps, "BGETextEncoder", FakeEncoder)

    validation_steps.embed_representations(context)

    assert (context.representations_dir / "item_index.json").is_file()
    for branch in ("graph_qwen", "graph_gemini", "desc"):
        assert (context.representations_dir / f"{branch}_embeddings.npz").is_file()
    assert len(loads) == 1
    assert len(encoded_texts) == 3

    (context.representations_dir / "desc_embeddings.npz").unlink()
    validation_steps.embed_representations(context)

    assert len(loads) == 2
    assert len(encoded_texts) == 4
    assert encoded_texts[-1] == ["description"]

    stable = {
        branch: np.load(
            context.representations_dir / f"{branch}_embeddings.npz"
        )["values"].copy()
        for branch in ("graph_qwen", "graph_gemini", "desc")
    }

    class FailingEncoder:
        def __init__(self, _settings):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated second-branch failure")
            return np.full((len(texts), dimension), 2.0, dtype=np.float32)

    monkeypatch.setattr(validation_steps, "BGETextEncoder", FailingEncoder)
    with pytest.raises(RuntimeError, match="second-branch failure"):
        validation_steps.embed_representations(context, force=True)
    for branch, expected in stable.items():
        actual = np.load(
            context.representations_dir / f"{branch}_embeddings.npz"
        )["values"]
        assert np.array_equal(actual, expected)
    assert not (context.representations_dir / "manifest.json").exists()


def test_missing_summary_error_names_the_actual_path(context: RunContext) -> None:
    context.initialize()
    write_jsonl(context.cohort_dir / "catalog.jsonl", [{"content_id": "c1", "item_id": "1"}])
    expected = context.graph_summary_dir("qwen")
    with pytest.raises(validation_steps.ValidationStepError, match="missing graph_qwen summary directory") as raised:
        validation_steps.embed_representations(context)
    assert str(expected) in str(raised.value)


def test_diagnosis_recomputes_runtime_data_and_overwrites_stale_pass(
    context: RunContext,
) -> None:
    context.initialize()
    write_json(
        context.diagnosis_path,
        {
            "schema_version": "diagnosis/v2",
            "runtime_decision": {"status": "pass", "checks": {}, "errors": []},
        },
    )

    with pytest.raises(validation_steps.ValidationStepError, match="runtime diagnosis failed"):
        validation_steps.run_diagnosis(context)

    diagnosis = json.loads(context.diagnosis_path.read_text(encoding="utf-8"))
    assert diagnosis["schema_version"] == "diagnosis/v2"
    assert diagnosis["runtime_decision"]["status"] == "fail"
    assert "report_ready" not in diagnosis
