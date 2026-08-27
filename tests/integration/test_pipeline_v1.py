from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml

import extraction.steps as extraction_steps
import viewing_context_pipeline.pipeline as pipeline_module
import validation.steps as validation_steps
from extraction.backends.qwen_workers import QwenGenerationTask
from viewing_context_pipeline.pipeline import STAGES, execute_stage, run_pipeline
from viewing_context_pipeline.runtime import ConfigError, RunContext, read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def context(tmp_path: Path) -> RunContext:
    data = tmp_path / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    for name in ("titles.csv", "tags.csv", "pairs.tsv"):
        (data / name).write_text("fixture\n", encoding="utf-8")
    models = tmp_path / "models"
    for name in ("qwen", "bge"):
        (models / name).mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    config["artifacts_root"] = str(tmp_path / "artifacts")
    config["data"] = {
        "videos_dir": str(videos),
        "titles_csv": str(data / "titles.csv"),
        "tags_csv": str(data / "tags.csv"),
        "pairs_tsv": str(data / "pairs.tsv"),
    }
    config["models"] = {
        "qwen": str(models / "qwen"),
        "bge": str(models / "bge"),
        "gemini": {
            "project_id": "test-project",
            "location": "global",
            "model_id": "test-gemini",
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


def test_public_stage_order_is_fixed() -> None:
    assert STAGES == (
        "prepare-cohort",
        "prepare-input-data",
        "extract-graph-scenes-qwen",
        "summarize-graph-qwen",
        "extract-graph-scenes-gemini",
        "summarize-graph-gemini",
        "extract-description-scenes",
        "summarize-description",
        "embed-representations",
        "run-recommendation",
        "run-diagnosis",
    )


def test_config_contract_remains_fixed(context: RunContext) -> None:
    assert context.config["schema_version"] == "viewing-context-config/v1"
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["protocol"]["modality"] = "multimodal"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="protocol.modality"):
        RunContext.load("other", root=context.root)


def test_runtime_has_no_orchestration_manifest_paths(context: RunContext) -> None:
    assert not hasattr(context, "pipeline_manifest")
    assert not hasattr(context, "stage_manifest")
    assert not hasattr(context, "visual_manifest")
    assert not hasattr(context, "representations_manifest")
    assert not hasattr(context, "recommendations_manifest")


def test_execute_stage_calls_handler_without_manifest_gate(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[RunContext, bool]] = []

    def handler(selected: RunContext, *, force: bool = False) -> dict:
        calls.append((selected, force))
        return {"stage": "prepare-cohort"}

    monkeypatch.setattr(pipeline_module, "handlers", lambda: {"prepare-cohort": handler})
    assert execute_stage(context, "prepare-cohort", force=True)["stage"] == "prepare-cohort"
    assert calls == [(context, True)]


def test_root_pipeline_runs_every_stage_in_order(
    context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "preflight",
        lambda _context: {"ready": True, "checks": {}},
    )
    monkeypatch.setattr(
        pipeline_module,
        "execute_stage",
        lambda _context, stage, **_kwargs: calls.append(stage) or {"stage": stage},
    )
    assert run_pipeline(context) == 0
    assert calls == list(STAGES)
    assert not (context.run_root / "pipeline_manifest.json").exists()
    assert not (context.run_root / "manifests").exists()


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
    assert set(scenes[0]) == {"scene_idx", "keyframes", "graph"}
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
            results = {task.task_id: "video summary" for task in tasks}
            if callback is not None:
                for task_id, text in results.items():
                    callback(task_id, text)
            return results

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)
    extraction_steps.summarize_graph(context, source="qwen")

    assert set(read_jsonl(scene_path)[0]) == {"scene_idx", "keyframes", "graph"}
    summary = json.loads(
        (context.graph_summary_dir("qwen") / "c1.json").read_text(encoding="utf-8")
    )
    assert summary == {
        "content_id": "c1",
        "scene_count": 1,
        "text": "video summary",
        "validation_warnings": [],
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
                "schema_version": "graph-video-summary/v1",
                "status": "complete",
                "text": f"{source} graph",
            },
        )
    write_json(
        context.description_summary_dir / f"{content_id}.json",
        {
            "schema_version": "description-video-summary/v1",
            "status": "complete",
            "text": "description",
        },
    )
    dimension = validation_steps.validation_config(context).encoder.embedding_dim
    monkeypatch.setattr(
        validation_steps,
        "encode_bge_texts",
        lambda _settings, texts: np.ones((len(texts), dimension), dtype=np.float32),
    )

    validation_steps.embed_representations(context)

    assert (context.representations_dir / "item_index.json").is_file()
    for branch in ("graph_qwen", "graph_gemini", "desc"):
        assert (context.representations_dir / f"{branch}_embeddings.npz").is_file()
    assert not (context.representations_dir / "manifest.json").exists()


def test_missing_summary_error_names_the_actual_path(context: RunContext) -> None:
    context.initialize()
    write_jsonl(context.cohort_dir / "catalog.jsonl", [{"content_id": "c1", "item_id": "1"}])
    expected = context.graph_summary_dir("qwen")
    with pytest.raises(validation_steps.ValidationStepError, match="missing graph_qwen summary directory") as raised:
        validation_steps.embed_representations(context)
    assert str(expected) in str(raised.value)
