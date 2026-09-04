from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import extraction.steps as extraction_steps
import extraction.summary_executor as summary_executor
import validation.features as validation_features
import validation.steps as validation_steps
from validation.cohort import prepare_cohort
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.summary_validation import SUMMARY_SECTIONS, parse_summary_sections
from pipeline_runtime import ConfigError, RunContext, read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def _summary_lines(sections: dict[str, str]) -> str:
    return "\n".join(f"{name}: {sections[name]}" for name in SUMMARY_SECTIONS)


def _ready_cohort(context: RunContext) -> str:
    context.config["validation"]["cohort"]["user_count"] = 1
    context.path("data", "pairs_tsv").write_text("fixture\t1 1 1 1 1\n", encoding="utf-8")
    (context.path("data", "videos_dir") / "1.mp4").write_bytes(b"video")
    prepare_cohort(
        validation_steps.validation_config(context), output_dir=context.cohort_dir,
        probe=lambda _: 10.0,
    )
    return "microlens_100k_00001"


@pytest.fixture()
def context(tmp_path: Path) -> RunContext:
    data = tmp_path / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    (data / "pairs.tsv").write_text("fixture\n", encoding="utf-8")
    (data / "titles.csv").write_text("1,Fixture title\n", encoding="utf-8")
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
        "titles_csv": str(data / "titles.csv"),
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
    for arm in ("graph", "description"):
        for key in ("scene_prompt", "summary_prompt"):
            config["extraction"][arm][key] = str(ROOT / config["extraction"][arm][key])
    (config_dir / "pipeline.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return RunContext.load("test_run", root=tmp_path)


def test_config_contract_remains_fixed(context: RunContext) -> None:
    assert context.config["schema_version"] == "viewing-context-config/v3"
    assert context.config["protocol"]["cohort_sampling"] == "user_first_nested_stratified"
    assert context.config["protocol"]["catalog_scope"] == "selected_user_sequence_union"
    assert set(context.config["data"]) == {"videos_dir", "pairs_tsv", "titles_csv"}
    assert context.config["protocol"]["arms"] == [
        "metadata",
        "graph_qwen",
        "graph_gemini",
        "description",
    ]
    assert "do_sample" not in context.config["extraction"]["graph"]
    assert "do_sample" not in context.config["extraction"]["description"]
    assert context.config["extraction"]["greedy_decoding"] is True
    assert context.config["extraction"]["graph_repetition_penalty"] == 1.05
    assert context.config["extraction"]["description_repetition_penalty"] == 1.0
    assert context.config["extraction"]["summary_repetition_penalty"] == 1.05
    assert context.config["extraction"]["summary_sampling"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert context.config["extraction"]["graph"]["summary_max_new_tokens"] == 512
    assert context.config["extraction"]["description"]["summary_max_new_tokens"] == 512
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
    ("change", "message"),
    [
        (lambda value: value.update(schema_version="viewing-context-config/v1"), "schema_version"),
        (lambda value: value.update(schema_version="viewing-context-config/v2"), "schema_version"),
        (
            lambda value: value["protocol"].update(
                arms=["id", "graph_qwen", "graph_gemini", "description"]
            ),
            "protocol.arms",
        ),
        (lambda value: value["protocol"].update(legacy=True), "protocol must contain exactly"),
        (lambda value: value["data"].pop("titles_csv"), "data must contain exactly"),
        (
            lambda value: value["validation"]["model"].update(embedding_dim=8),
            "invalid validation config",
        ),
    ],
)
def test_pipeline_config_rejects_legacy_and_non_exact_v3_contract(
    context: RunContext,
    change,
    message: str,
) -> None:
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        RunContext.load("legacy", root=context.root)


def test_greedy_decoding_switches_summary_generation_mode(
    context: RunContext,
) -> None:
    assert extraction_steps._summary_generation_settings(context) == {
        "repetition_penalty": 1.05,
    }
    context.config["extraction"]["greedy_decoding"] = False
    assert extraction_steps._summary_generation_settings(context) == {
        "repetition_penalty": 1.05,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "config/prompts/graph_summary_v3.md",
        "config/prompts/description_summary_v3.md",
    ],
)
def test_summary_prompts_require_bounded_single_line_values(
    relative_path: str,
) -> None:
    prompt = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "Output exactly seven physical lines in the required order." in prompt
    assert "Every non-empty value must be one natural, complete English sentence." in prompt
    assert "not an exhaustive inventory" in prompt
    assert "Never create combinations by pairing every person" in prompt
    assert "Use at most 20 English words per field." in prompt
    assert "Stop immediately after the semantic_topics line." in prompt
    for name in SUMMARY_SECTIONS:
        assert f"{name}: <one complete sentence or empty>" in prompt


def test_config_rejects_non_boolean_greedy_decoding(context: RunContext) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["extraction"]["greedy_decoding"] = "true"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="greedy_decoding"):
        RunContext.load("invalid-greedy", root=context.root)


@pytest.mark.parametrize("stage", ["graph", "description", "summary"])
@pytest.mark.parametrize("value", [0.9, 2.1, True, "1.05", None, float("nan"), float("inf")])
def test_config_rejects_invalid_repetition_penalty(
    context: RunContext,
    stage: str,
    value: object,
) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["extraction"][f"{stage}_repetition_penalty"] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=f"{stage}_repetition_penalty"):
        RunContext.load("invalid-repetition", root=context.root)


@pytest.mark.parametrize("stage", ["graph", "description", "summary"])
@pytest.mark.parametrize("value", [1, 1.15, 2])
def test_config_accepts_independent_repetition_penalty(context, stage, value) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["extraction"][f"{stage}_repetition_penalty"] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = RunContext.load("valid-repetition", root=context.root)
    assert loaded.config["extraction"] == config["extraction"]


@pytest.mark.parametrize("stage", ["graph", "description", "summary"])
def test_config_requires_each_repetition_penalty(context, stage) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    del config["extraction"][f"{stage}_repetition_penalty"]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=f"{stage}_repetition_penalty"):
        RunContext.load("missing-repetition", root=context.root)


@pytest.mark.parametrize(
    ("stage", "source", "expected_penalty"),
    [
        ("graph", "qwen", 1.11),
        ("graph", "gemini", 1.0),
        ("description", None, 1.22),
        ("summary", "qwen", 1.33),
        ("summary", "gemini", 1.33),
        ("summary", None, 1.33),
    ],
)
@pytest.mark.parametrize("greedy", [True, False])
def test_repetition_penalty_reaches_only_its_generation_stage(
    context, monkeypatch, stage, source, expected_penalty, greedy
) -> None:
    context.config["extraction"].update(
        graph_repetition_penalty=1.11,
        description_repetition_penalty=1.22,
        summary_repetition_penalty=1.33,
        greedy_decoding=greedy,
    )
    content_id = _ready_cohort(context)
    frames = context.evidence_dir / "resized_keyframes" / content_id
    frames.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(frames / "0005.png")
    write_json(
        context.cohort_dir / "source_assets" / content_id / "assets/timestamp_fixed_30s.json",
        [{"scene_start": 0, "scene_end": 10, "keyframe_timestamps": [5]}],
    )
    graph_record = {
        "scene_idx": 0,
        "keyframes": [5],
        "graph": {"setting_context": "indoor"},
        "parse_mode": "native",
        "semantic_warnings": [],
    }
    description_record = {
        "schema_version": "scene-description/v1",
        "content_id": content_id,
        "scene_idx": 0,
        "keyframes": [5],
        "description": "A person walks.",
    }
    if stage == "summary":
        scene_dir = context.graph_scene_dir(source) if source else context.description_scene_dir
        write_jsonl(scene_dir / f"{content_id}.jsonl", [graph_record if source else description_record])
        response = _summary_lines({name: "A person walks." for name in SUMMARY_SECTIONS})
    else:
        response = json.dumps(graph_record["graph"]) if stage == "graph" else "A person walks."
    captured = []

    def generate(tasks, callback):
        captured.extend(tasks)
        for task in tasks:
            callback(task.task_id, response)
        return {}

    @contextmanager
    def fake_generator(**_kwargs):
        yield generate

    class FakeGeminiPool:
        def __init__(self, *_args, **kwargs):
            assert not any("penalty" in key for key in kwargs)

        def generate(self, tasks, callback):
            captured.extend(tasks)
            for task in tasks:
                callback(extraction_steps.GeminiGenerationOutcome(task.task_id, response))

    monkeypatch.setattr(extraction_steps, "qwen_generator", fake_generator)
    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", FakeGeminiPool)
    if stage == "graph":
        result = extraction_steps.extract_graph_scenes(context, model=source)
    elif stage == "description":
        result = extraction_steps.extract_description_scenes(context)
    elif source:
        result = extraction_steps.summarize_graph(context, source=source)
    else:
        result = extraction_steps.summarize_description(context)
    assert result["failure_count"] == 0
    assert len(captured) == 1
    assert captured[0].repetition_penalty == expected_penalty
    assert captured[0].do_sample is (stage == "summary" and not greedy)


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
        ("temperature", 0.0, "temperature"),
        ("top_p", 1.1, "top_p"),
        ("top_k", 0, "top_k"),
    ],
)
def test_config_rejects_invalid_summary_sampling_settings(
    context: RunContext,
    key: str,
    value: object,
    message: str,
) -> None:
    path = context.root / "config/pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["extraction"]["summary_sampling"][key] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        RunContext.load("invalid-summary-sampling", root=context.root)


def test_summary_batch_saves_valid_callbacks_and_reports_failure_once() -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "visual_atmosphere": "A calm indoor atmosphere",
        "visible_affect": "Neutral visible affect",
        "semantic_topics": "Indoor activity",
    }
    structured = _summary_lines(sections)
    completed = []
    failures = []
    submissions = []

    def generate(tasks, callback=None):
        assert callback is not None
        submissions.append([task.task_id for task in tasks])
        callback("c2", structured)
        assert completed == ["c2"]
        callback("c1", "not labeled text")
        return {}

    def complete(task_id, text):
        parse_summary_sections(text)
        completed.append(task_id)

    with pytest.raises(
        extraction_steps.ExtractionStepError,
        match=r"task_ids=\['c1'\]",
    ):
        summary_executor.generate_summaries_once(
            generate,
            [
                QwenGenerationTask("c1", (), "prompt", 512),
                QwenGenerationTask("c2", (), "prompt", 512),
            ],
            complete,
            lambda task_id, attempt, seed, raw_response, error: failures.append(
                (task_id, attempt, seed, raw_response, str(error))
            ),
        )

    assert submissions == [["c1", "c2"]]
    assert completed == ["c2"]
    assert len(failures) == 1
    assert failures[0][:4] == ("c1", 1, None, "not labeled text")


def test_reused_summary_rejects_noncanonical_text(
    tmp_path: Path,
) -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "visual_atmosphere": "A calm indoor atmosphere",
        "visible_affect": "Neutral visible affect",
        "semantic_topics": "Indoor activity",
    }
    path = tmp_path / "summary.json"
    write_json(
        path,
        {
            "schema_version": "graph-video-summary/v3",
            "content_id": "c1",
            "arm": "graph_qwen",
            "status": "complete",
            "sections": sections,
            "text": "legacy single-line text",
            "scene_count": 1,
        },
    )

    with pytest.raises(
        summary_executor.ExtractionStepError,
        match="use --force or a new run_id",
    ):
        summary_executor.reuse_summary_document(
            path,
            schema_version="graph-video-summary/v3",
            content_id="c1",
            arm="graph_qwen",
            scene_count=1,
        )


def test_reused_summary_rejects_v2_and_requires_new_run_or_force(tmp_path: Path) -> None:
    sections = {
        "setting_and_environments": "An indoor room",
        "main_characters_and_objects": "A person",
        "chronological_events": "The person walks",
        "relations": "The person is inside the room",
        "visual_atmosphere": "A calm indoor atmosphere",
        "visible_affect": "Neutral visible affect",
        "semantic_topics": "Indoor activity",
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
            "text": "legacy",
            "scene_count": 1,
        },
    )

    with pytest.raises(
        summary_executor.ExtractionStepError,
        match="use --force or a new run_id",
    ):
        summary_executor.reuse_summary_document(
            path,
            schema_version="graph-video-summary/v3",
            content_id="c1",
            arm="graph_qwen",
            scene_count=1,
        )


def test_runtime_has_no_orchestration_manifest_paths(context: RunContext) -> None:
    assert not hasattr(context, "pipeline_manifest")
    assert not hasattr(context, "stage_manifest")
    assert not hasattr(context, "visual_manifest")
    assert not hasattr(context, "representations_manifest")
    assert not hasattr(context, "recommendations_manifest")


def test_failure_directories_are_nested_under_their_artifact_stage(
    context: RunContext,
) -> None:
    assert context.graph_failure_dir("qwen") == (context.graph_scene_dir("qwen") / "failures")
    assert context.graph_summary_failure_dir("gemini") == (
        context.graph_summary_dir("gemini") / "failures"
    )
    assert context.description_failure_dir == (context.description_scene_dir / "failures")
    assert context.description_summary_failure_dir == (context.description_summary_dir / "failures")


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

    monkeypatch.setattr(extraction_steps, "qwen_generator", fake_generator)
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

    monkeypatch.setattr(extraction_steps, "qwen_generator", successful_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen", force=True)

    assert result["failure_count"] == 0
    failure_path = context.graph_failure_dir("qwen") / "c1.jsonl"
    assert not failure_path.exists()

    write_jsonl(failure_path, [])

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("completed scenes must be reused")
        yield

    monkeypatch.setattr(extraction_steps, "qwen_generator", unexpected_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 0
    assert not failure_path.exists()


def test_description_failure_force_retry_and_cache_reuse(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
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
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )

    @contextmanager
    def failing_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {tasks[0].task_id: "visible action", tasks[1].task_id: ""}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", failing_generator)
    result = extraction_steps.extract_description_scenes(context)
    assert result["failure_count"] == 1
    failure_path = context.description_failure_dir / "c1.jsonl"
    assert read_jsonl(failure_path)[0]["failure_kind"] == "empty_response"

    @contextmanager
    def successful_generator(**_kwargs):
        def generate(tasks, _callback=None):
            return {task.task_id: f"visible action {task.task_id}" for task in tasks}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", successful_generator)
    result = extraction_steps.extract_description_scenes(context, force=True)
    assert result["failure_count"] == 0
    assert not failure_path.exists()
    assert len(read_jsonl(context.description_scene_dir / "c1.jsonl")) == 2

    write_jsonl(failure_path, [])

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("completed description scenes must be reused")
        yield

    monkeypatch.setattr(extraction_steps, "qwen_generator", unexpected_generator)
    result = extraction_steps.extract_description_scenes(context)
    assert result["failure_count"] == 0
    assert not failure_path.exists()


def test_qwen_graph_scene_is_checkpointed_before_content_finishes(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", ("unused.png",), "prompt", 10),
            "scene_idx": index,
            "keyframes": [index * 30 + 5],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
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
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"

    @contextmanager
    def streaming_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[1].task_id, json.dumps(graph))
            assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [1]
            callback(tasks[0].task_id, json.dumps(graph))
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", streaming_generator)
    result = extraction_steps.extract_graph_scenes(context, model="qwen")

    assert result["failure_count"] == 0
    assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [0, 1]
    assert "each completed scene is checkpointed immediately" in capsys.readouterr().err


def test_qwen_description_scene_is_checkpointed_before_content_finishes(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
    scene_rows = [
        {
            "task": QwenGenerationTask(f"c1:{index}", ("unused.png",), "prompt", 10),
            "scene_idx": index,
            "keyframes": [index * 30 + 5],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
    monkeypatch.setattr(
        extraction_steps,
        "_scene_generation_rows",
        lambda *_args, **_kwargs: scene_rows,
    )
    scene_path = context.description_scene_dir / "c1.jsonl"
    failure_path = context.description_failure_dir / "c1.jsonl"

    @contextmanager
    def streaming_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[1].task_id, "visible action")
            assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [1]
            callback(tasks[0].task_id, "")
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", streaming_generator)
    result = extraction_steps.extract_description_scenes(context)

    assert result["failure_count"] == 1
    assert [row["scene_idx"] for row in read_jsonl(scene_path)] == [1]
    assert [row["scene_idx"] for row in read_jsonl(failure_path)] == [0]
    assert "each completed scene is checkpointed immediately" in capsys.readouterr().err


def test_gemini_scene_stage_aggregates_out_of_order_errors(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "1", "source_video_path": "1.mp4"}],
    )
    visual = {"content_id": "c1", "frames_dir": "unused", "timestamp_json": "unused"}
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
    monkeypatch.setattr(extraction_steps, "_visual_rows", lambda _context: [visual])
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

    class Pool:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, tasks, callback):
            outcomes = {
                tasks[1].task_id: extraction_steps.GeminiGenerationOutcome(
                    tasks[1].task_id, "", "RuntimeError: quota"
                ),
                tasks[0].task_id: extraction_steps.GeminiGenerationOutcome(
                    tasks[0].task_id, json.dumps(graph), None
                ),
            }
            for task_id in (tasks[1].task_id, tasks[0].task_id):
                callback(outcomes[task_id])
            return outcomes

    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", Pool)
    result = extraction_steps.extract_graph_scenes(context, model="gemini")
    assert result["failure_count"] == 1
    assert len(read_jsonl(context.graph_scene_dir("gemini") / "c1.jsonl")) == 1
    failure = read_jsonl(context.graph_failure_dir("gemini") / "c1.jsonl")[0]
    assert failure["scene_idx"] == 1
    assert failure["failure_kind"] == "generation"


def test_graph_summary_failure_is_saved_once_and_manual_resume_removes_it(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    write_jsonl(
        context.graph_scene_dir("qwen") / "c1.jsonl",
        [
            {
                "scene_idx": 0,
                "keyframes": [5, 15, 25],
                "graph": {"setting_context": "indoor"},
                "parse_mode": "native",
                "semantic_warnings": [],
            }
        ],
    )
    sections = {
        "setting_and_environments": "An indoor setting.",
        "main_characters_and_objects": "",
        "chronological_events": "",
        "relations": "",
        "visual_atmosphere": "",
        "visible_affect": "",
        "semantic_topics": "",
    }
    output_path = context.graph_summary_dir("qwen") / "c1.json"
    failure_path = context.graph_summary_failure_dir("qwen") / "c1.jsonl"

    @contextmanager
    def invalid_generator(**_kwargs):
        def generate(tasks, callback):
            assert tasks[0].do_sample is False
            assert tasks[0].repetition_penalty == 1.05
            assert tasks[0].max_new_tokens == 512
            callback(tasks[0].task_id, "not labeled text")
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", invalid_generator)
    with pytest.raises(
        extraction_steps.ExtractionStepError,
        match=r"structured summary failed: task_ids=\['c1'\]",
    ):
        extraction_steps.summarize_graph(context, source="qwen", gpus=1)

    failures = read_jsonl(failure_path)
    assert len(failures) == 1
    assert failures[0]["schema_version"] == "summary-generation-failure/v1"
    assert failures[0]["attempt"] == 1
    assert failures[0]["seed"] is None
    assert failures[0]["raw_response"] == "not labeled text"
    stderr = capsys.readouterr().err
    assert "[Qwen_summary_graph_qwen_fail]" in stderr
    assert "Raw output:\nnot labeled text" in stderr
    assert "generation started" not in stderr

    @contextmanager
    def valid_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[0].task_id, _summary_lines(sections))
            assert output_path.is_file()
            assert not failure_path.exists()
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", valid_generator)
    result = extraction_steps.summarize_graph(context, source="qwen", gpus=1)

    assert result["content_count"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["sections"] == sections
    assert not failure_path.exists()


def test_description_summary_failure_is_nested_and_success_retry_removes_it(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    write_jsonl(
        context.description_scene_dir / "c1.jsonl",
        [
            {
                "schema_version": "scene-description/v1",
                "content_id": "c1",
                "scene_idx": 0,
                "keyframes": [5, 15, 25],
                "description": "A person walks indoors.",
            }
        ],
    )
    failure_path = context.description_summary_failure_dir / "c1.jsonl"

    @contextmanager
    def invalid_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[0].task_id, "not json")
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", invalid_generator)
    with pytest.raises(
        extraction_steps.ExtractionStepError,
        match="structured summary failed",
    ):
        extraction_steps.summarize_description(context)

    failures = read_jsonl(failure_path)
    assert [row["attempt"] for row in failures] == [1]
    assert [row["seed"] for row in failures] == [None]
    assert all(row["failure_kind"] == "schema_validation" for row in failures)
    assert all(row["raw_response"] == "not json" for row in failures)
    stderr = capsys.readouterr().err
    assert "[Qwen_summary_description_fail]" in stderr
    assert "Raw output:\nnot json" in stderr

    sections = {
        "setting_and_environments": "An indoor setting.",
        "main_characters_and_objects": "A person.",
        "chronological_events": "The person walks.",
        "relations": "",
        "visual_atmosphere": "",
        "visible_affect": "",
        "semantic_topics": "Walking indoors.",
    }

    @contextmanager
    def valid_generator(**_kwargs):
        def generate(tasks, callback):
            callback(tasks[0].task_id, _summary_lines(sections))
            return {}

        yield generate

    monkeypatch.setattr(extraction_steps, "qwen_generator", valid_generator)
    result = extraction_steps.summarize_description(context)

    assert result["content_count"] == 1
    assert not failure_path.exists()
    assert (context.description_summary_dir / "c1.json").is_file()


def test_graph_summary_rejects_legacy_scene_shape(
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

    with pytest.raises(
        extraction_steps.ExtractionStepError,
        match="use --force or a new run_id",
    ):
        extraction_steps.summarize_graph(context, source="qwen")


def test_embedding_uses_fixed_files_and_no_manifest(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.initialize()
    content_id = _ready_cohort(context)
    for source in ("qwen", "gemini"):
        write_json(
            context.graph_summary_dir(source) / f"{content_id}.json",
            {
                "schema_version": "graph-video-summary/v3",
                "content_id": content_id,
                "status": "complete",
                "text": f"{source} graph",
            },
        )
    write_json(
        context.description_summary_dir / f"{content_id}.json",
        {
            "schema_version": "description-video-summary/v3",
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

    monkeypatch.setattr(validation_features, "BGETextEncoder", FakeEncoder)

    validation_steps.embed_representations(context)

    assert (context.representations_dir / "item_index.json").is_file()
    for branch in ("metadata", "graph_qwen", "graph_gemini", "desc"):
        assert (context.representations_dir / f"{branch}_embeddings.npz").is_file()
    assert len(loads) == 1
    assert len(encoded_texts) == 4
    assert encoded_texts[0] == ["Fixture title"]

    (context.representations_dir / "desc_embeddings.npz").unlink()
    validation_steps.embed_representations(context)

    assert len(loads) == 2
    assert len(encoded_texts) == 5
    assert encoded_texts[-1] == ["description"]

    stable = {
        branch: np.load(context.representations_dir / f"{branch}_embeddings.npz")["values"].copy()
        for branch in ("metadata", "graph_qwen", "graph_gemini", "desc")
    }

    class FailingEncoder:
        def __init__(self, _settings):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated second-branch failure")
            return np.full((len(texts), dimension), 2.0, dtype=np.float32)

    monkeypatch.setattr(validation_features, "BGETextEncoder", FailingEncoder)
    with pytest.raises(RuntimeError, match="second-branch failure"):
        validation_steps.embed_representations(context, force=True)
    for branch, expected in stable.items():
        actual = np.load(context.representations_dir / f"{branch}_embeddings.npz")["values"]
        assert np.array_equal(actual, expected)
    assert not (context.representations_dir / "manifest.json").exists()


def test_missing_summary_error_names_the_actual_path(context: RunContext) -> None:
    context.initialize()
    _ready_cohort(context)
    expected = context.graph_summary_dir("qwen")
    with pytest.raises(
        validation_steps.ValidationStepError, match="missing graph_qwen summary directory"
    ) as raised:
        validation_steps.embed_representations(context)
    assert str(expected) in str(raised.value)


def test_diagnosis_recomputes_runtime_data_and_overwrites_stale_pass(
    context: RunContext,
) -> None:
    context.initialize()
    write_json(
        context.diagnosis_path,
        {
            "schema_version": "diagnosis/v3",
            "runtime_decision": {"status": "pass", "checks": {}, "errors": []},
        },
    )

    with pytest.raises(validation_steps.ValidationStepError, match="runtime diagnosis failed"):
        validation_steps.run_diagnosis(context)

    diagnosis = json.loads(context.diagnosis_path.read_text(encoding="utf-8"))
    assert diagnosis["schema_version"] == "diagnosis/v4"
    assert diagnosis["runtime_decision"]["status"] == "fail"
    assert "report_ready" not in diagnosis
