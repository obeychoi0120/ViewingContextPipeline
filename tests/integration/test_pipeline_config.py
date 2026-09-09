from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import yaml
from PIL import Image

import extraction.steps as extraction_steps
from extraction.summary_validation import SUMMARY_SECTIONS
from pipeline_runtime import ConfigError, RunContext, write_json, write_jsonl


from pipeline_fixtures import ROOT, context as context, _ready_cohort, _summary_lines


def test_config_contract_remains_fixed(context: RunContext) -> None:
    assert context.config["protocol"]["sampling"] == "fixed_windows"
    assert context.config["extraction"]["visual_evidence"]["scene_duration"] == 30
    assert context.config["extraction"]["visual_evidence"]["num_keyframes"] == 6
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
    from extraction.structured_output import SUMMARY_GRAMMAR
    assert extraction_steps._summary_generation_settings(context) == {
        "repetition_penalty": 1.05,
        "structured_output": {"grammar": SUMMARY_GRAMMAR},
    }
    context.config["extraction"]["greedy_decoding"] = False
    assert extraction_steps._summary_generation_settings(context) == {
        "repetition_penalty": 1.05,
        "structured_output": {"grammar": SUMMARY_GRAMMAR},
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
    }


@pytest.mark.parametrize("stage", ["graph", "description", "summary"])
@pytest.mark.parametrize("value", [1.05, [1, 1.05, 1.1, 1.15, 1.2], [], [1, True], [float("inf")]])
def test_recovery_schedule_config_validation(context, stage, value):
    path = context.root / "config/pipeline.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["extraction"][f"{stage}_repetition_penalty"] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    if value == 1.05 or value == [1, 1.05, 1.1, 1.15, 1.2]:
        assert RunContext.load("recovery-config", root=context.root).config["extraction"][f"{stage}_repetition_penalty"] == value
    else:
        with pytest.raises(ConfigError, match=f"{stage}_repetition_penalty"):
            RunContext.load("recovery-config", root=context.root)


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


@pytest.mark.parametrize(
    "key,value",
    [
        ("scene_duration", 0),
        ("scene_duration", -1),
        ("scene_duration", True),
        ("scene_duration", 2.5),
        ("scene_duration", "30"),
        ("num_keyframes", 0),
        ("num_keyframes", -1),
        ("num_keyframes", True),
        ("num_keyframes", 6.5),
        ("num_keyframes", "6"),
        ("num_keyframes", 301),
    ],
)
def test_config_rejects_invalid_visual_sampling(context, key, value):
    config = context.config
    config["extraction"]["visual_evidence"][key] = value
    (context.root / "config/pipeline.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="visual_evidence"):
        RunContext.load("invalid-sampling", root=context.root)


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
        write_jsonl(
            scene_dir / f"{content_id}.jsonl", [graph_record if source else description_record]
        )
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

        def generate(self, tasks, callback, *, on_progress=None):
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
