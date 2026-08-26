from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml

import viewing_context_pipeline.pipeline as pipeline_module
import extraction.steps as extraction_steps
from extraction.steps import _write_stage as write_extraction_stage
import validation.steps as validation_steps
from viewing_context_pipeline.cli import main as pipeline_cli
from viewing_context_pipeline.pipeline import (
    STAGES,
    descendants,
    execute_stage,
    preflight,
    run_pipeline,
)
from viewing_context_pipeline.runtime import (
    ConfigError,
    RunContext,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


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
    config = yaml.safe_load(
        (ROOT / "config/pipeline.yaml").read_text(encoding="utf-8")
    )
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


def test_public_stage_order_matches_step_first_dag() -> None:
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


def test_single_pipeline_config_contract_starts_at_v1(context: RunContext) -> None:
    assert context.config["schema_version"] == "viewing-context-config/v1"
    assert set(context.config) == {
        "schema_version",
        "protocol",
        "artifacts_root",
        "data",
        "models",
        "extraction",
        "validation",
    }
    assert context.config["protocol"]["graph_extractors"] == ["qwen", "gemini"]
    assert context.config["protocol"]["graph_summarizer"] == "qwen"
    assert context.config["extraction"]["graph"]["gemini_concurrency"] == 4
    assert set(context.config["models"]["gemini"]) == {
        "project_id",
        "location",
        "model_id",
    }


def test_config_rejects_nonpositive_gemini_concurrency(context: RunContext) -> None:
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["extraction"]["graph"]["gemini_concurrency"] = 0
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="positive integer"):
        RunContext.load("other", root=context.root)


def test_packages_are_top_level_and_old_namespace_is_absent() -> None:
    assert importlib.util.find_spec("extraction") is not None
    assert importlib.util.find_spec("validation") is not None
    assert importlib.util.find_spec("viewing_context_pipeline.extraction") is None
    assert importlib.util.find_spec("viewing_context_pipeline.validation") is None


def test_fixed_config_rejects_protocol_drift(context: RunContext) -> None:
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["protocol"]["modality"] = "multimodal"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="protocol.modality"):
        RunContext.load("other", root=context.root)


def test_single_config_rejects_wrong_schema(context: RunContext) -> None:
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["schema_version"] = "viewing-context-config/v2"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="viewing-context-config/v1"):
        RunContext.load("other", root=context.root)


def test_initialize_preserves_existing_run_directory(context: RunContext) -> None:
    context.run_root.mkdir(parents=True)
    marker = context.run_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    context.initialize()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_resume_allows_changed_config_for_same_run(context: RunContext) -> None:
    context.initialize()
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["validation"]["model"]["dropout"] = 0.2
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    changed = RunContext.load(context.run_id, root=context.root)
    changed.initialize()
    assert changed.run_root == context.run_root


def test_graph_force_does_not_invalidate_description_sibling() -> None:
    affected = descendants({"extract-graph-scenes-qwen"})
    assert "summarize-graph-qwen" in affected
    assert "embed-representations" in affected
    assert "extract-graph-scenes-gemini" not in affected
    assert "summarize-graph-gemini" not in affected
    assert "extract-description-scenes" not in affected
    assert "summarize-description" not in affected


def test_changed_graph_fingerprint_marks_only_actual_downstream_stale(context: RunContext) -> None:
    context.initialize()
    for stage in STAGES:
        write_json(context.stage_manifest(stage), {
            "schema_version": "step-manifest/v1",
            "run_id": context.run_id,
            "stage": stage,
            "status": "complete",
            "source_fingerprints": {},
            "output_fingerprint": "a" * 64,
        })
    write_extraction_stage(
        context,
        "extract-graph-scenes-qwen",
        source_fingerprints={"taxonomy": "changed"},
        output_fingerprint="b" * 64,
    )
    assert context.stage_manifest("extract-description-scenes").is_file()
    assert context.stage_manifest("summarize-description").is_file()
    assert not context.stage_manifest("summarize-graph-qwen").exists()
    assert context.stage_manifest("summarize-graph-gemini").is_file()
    assert not context.stage_manifest("embed-representations").exists()


def test_embedding_reuses_unchanged_description_branch(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context.initialize()
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "i1"}],
    )
    for stage, output_fingerprint in (
        ("summarize-graph-qwen", "graph-qwen-v1"),
        ("summarize-graph-gemini", "graph-gemini-v1"),
        ("summarize-description", "description-v1"),
    ):
        write_json(
            context.stage_manifest(stage),
            {
                "schema_version": "step-manifest/v1",
                "run_id": context.run_id,
                "stage": stage,
                "status": "complete",
                "source_fingerprints": {},
                "output_fingerprint": output_fingerprint,
            },
        )
    common = {
        "content_id": "c1",
        "status": "complete",
        "evidence_fingerprint": "same-evidence",
    }
    write_json(
        context.graph_summary_dir("qwen") / "c1.json",
        {
            **common,
            "schema_version": "graph-video-summary/v1",
            "graph_source": "qwen",
            "summary_model": "qwen",
            "text": "graph qwen one",
        },
    )
    write_json(
        context.graph_summary_dir("gemini") / "c1.json",
        {
            **common,
            "schema_version": "graph-video-summary/v1",
            "graph_source": "gemini",
            "summary_model": "qwen",
            "text": "graph gemini one",
        },
    )
    write_json(
        context.description_summary_dir / "c1.json",
        {
            **common,
            "schema_version": "description-video-summary/v1",
            "text": "description one",
        },
    )
    encoded: list[list[str]] = []

    def fake_encode(config, texts):
        encoded.append(texts)
        return np.ones((len(texts), config.embedding_dim), dtype=np.float32)

    monkeypatch.setattr(validation_steps, "encode_bge_texts", fake_encode)
    validation_steps.embed_representations(context)
    first = read_json(context.representations_manifest)
    description_artifact = first["branches"]["desc"]["artifact_fingerprint"]
    assert encoded == [
        ["graph qwen one"],
        ["graph gemini one"],
        ["description one"],
    ]

    write_json(
        context.graph_summary_dir("qwen") / "c1.json",
        {
            **common,
            "schema_version": "graph-video-summary/v1",
            "graph_source": "qwen",
            "summary_model": "qwen",
            "text": "graph qwen two",
        },
    )
    graph_stage = read_json(context.stage_manifest("summarize-graph-qwen"))
    graph_stage["output_fingerprint"] = "graph-v2"
    write_json(context.stage_manifest("summarize-graph-qwen"), graph_stage)

    validation_steps.embed_representations(context)
    second = read_json(context.representations_manifest)

    assert encoded == [
        ["graph qwen one"],
        ["graph gemini one"],
        ["description one"],
        ["graph qwen two"],
    ]
    assert (
        second["branches"]["desc"]["artifact_fingerprint"]
        == description_artifact
    )


def test_graph_steps_write_minimal_graph_and_scene_provenance(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context.initialize()
    write_json(
        context.stage_manifest("prepare-input-data"),
        {
            "schema_version": "step-manifest/v1",
            "run_id": context.run_id,
            "stage": "prepare-input-data",
            "status": "complete",
            "source_fingerprints": {},
            "output_fingerprint": "visual-v1",
        },
    )
    frames = context.evidence_dir / "resized_keyframes" / "c1"
    frames.mkdir(parents=True)
    for timestamp in (5, 15, 25):
        (frames / f"{timestamp:04d}.png").write_bytes(b"fixture")
    timestamps = context.cohort_dir / "timestamp.json"
    timestamps.parent.mkdir(parents=True, exist_ok=True)
    timestamps.write_text(
        json.dumps(
            [
                {
                    "scene_start": 0,
                    "scene_end": 30,
                    "keyframe_timestamps": [5, 15, 25],
                }
            ]
        ),
        encoding="utf-8",
    )
    write_jsonl(
        context.visual_manifest,
        [
            {
                "content_id": "c1",
                "frames_dir": str(frames),
                "timestamp_json": str(timestamps),
                "evidence_fingerprint": "evidence-v1",
            }
        ],
    )
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [{"content_id": "c1", "item_id": "i1", "source_video_path": "1.mp4"}],
    )
    graph = {
        "setting_context": "unknown",
        "entities": [
            {"local_id": f"e{index}", "name": "player", "role": "secondary"}
            for index in range(1, 9)
        ],
        "events": [],
        "static_relations": [],
        "semantic_topics": [],
        "affect": {
            "subject_ids": [],
            "valence": "unknown",
            "arousal": "unknown",
        },
    }

    @contextmanager
    def fake_generator(**_kwargs):
        def generate(tasks, on_task_complete=None):
            results = {}
            for task in tasks:
                text = (
                    "short factual summary"
                    if not task.image_paths
                    else json.dumps(graph)
                )
                results[task.task_id] = text
                if on_task_complete is not None:
                    on_task_complete(task.task_id, text)
            return results

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)

    extraction_steps.extract_graph_scenes(context, model="qwen")
    scene_path = context.graph_scene_dir("qwen") / "c1.jsonl"
    scene = read_jsonl(scene_path)[0]
    assert scene["schema_version"] == "minimal-semantic-scene/v1"
    assert scene["scene_start_seconds"] == 0
    assert scene["scene_end_seconds"] == 30
    assert scene["keyframes"] == [5, 15, 25]
    assert scene["graph"] == graph
    assert len(scene["graph"]["entities"]) == 8
    assert scene["parse_status"] == "parsed"
    assert scene["validation_warnings"] == [
        "entity_guidance_exceeded: observed=8 guidance_max=6"
    ]
    assert {
        "taxonomy_fingerprint",
        "prompt_fingerprint",
        "extractor_model_fingerprint",
        "evidence_fingerprint",
        "input_fingerprint",
        "repair_fingerprint",
        "validation_fingerprint",
    }.issubset(scene)

    legacy = dict(scene)
    legacy.pop("parse_status")
    legacy.pop("repair_fingerprint")
    legacy.pop("validation_fingerprint")
    legacy.pop("validation_warnings")
    write_jsonl(scene_path, [legacy])
    context.stage_manifest("extract-graph-scenes-qwen").unlink()

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("legacy migration must not load the model")
        yield

    monkeypatch.setattr(extraction_steps, "_qwen_generator", unexpected_generator)
    extraction_steps.extract_graph_scenes(context, model="qwen")
    migrated = read_jsonl(scene_path)[0]
    assert migrated["parse_status"] == "legacy_parser"
    assert migrated["repair_fingerprint"]
    assert migrated["validation_fingerprint"]
    assert migrated["validation_warnings"] == [
        "entity_guidance_exceeded: observed=8 guidance_max=6"
    ]

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)
    extraction_steps.summarize_graph(context, source="qwen")
    summary = read_json(context.graph_summary_dir("qwen") / "c1.json")
    assert summary["text"] == "short factual summary"
    assert summary["graph_source"] == "qwen"
    assert summary["summary_model"] == "qwen"
    assert summary["scene_graph_path"] == "extraction/graph/qwen/scenes/c1.jsonl"
    assert summary["scene_graph_fingerprint"]


def test_graph_json_failure_is_persisted_resumed_and_retried_only_with_force(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context.initialize()
    write_json(
        context.stage_manifest("prepare-input-data"),
        {
            "schema_version": "step-manifest/v1",
            "run_id": context.run_id,
            "stage": "prepare-input-data",
            "status": "complete",
            "source_fingerprints": {},
            "output_fingerprint": "visual-v1",
        },
    )
    visual_rows = []
    catalog_rows = []
    for content_id, filename in (("c1", "1.mp4"), ("c2", "2.mp4")):
        frames = context.evidence_dir / "resized_keyframes" / content_id
        frames.mkdir(parents=True)
        for timestamp in (5, 15, 25):
            (frames / f"{timestamp:04d}.png").write_bytes(b"fixture")
        timestamp_path = context.cohort_dir / f"{content_id}.json"
        timestamp_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp_path.write_text(
            json.dumps([{
                "scene_start": 0,
                "scene_end": 30,
                "keyframe_timestamps": [5, 15, 25],
            }]),
            encoding="utf-8",
        )
        visual_rows.append({
            "content_id": content_id,
            "frames_dir": str(frames),
            "timestamp_json": str(timestamp_path),
            "evidence_fingerprint": f"evidence-{content_id}",
        })
        catalog_rows.append({
            "content_id": content_id,
            "item_id": content_id,
            "source_video_path": filename,
        })
    write_jsonl(context.visual_manifest, visual_rows)
    write_jsonl(context.cohort_dir / "catalog.jsonl", catalog_rows)
    for stage in ("extract-graph-scenes-qwen", "summarize-graph-qwen"):
        write_json(
            context.stage_manifest(stage),
            {
                "schema_version": "step-manifest/v1",
                "run_id": context.run_id,
                "stage": stage,
                "status": "complete",
                "source_fingerprints": {},
                "output_fingerprint": "old-output",
            },
        )
    valid_graph = {"entities": [], "static_relations": []}

    def generator_for(responses: dict[str, str], calls: list[str]):
        @contextmanager
        def fake_generator(**_kwargs):
            def generate(tasks, on_task_complete=None):
                results = {}
                for task in tasks:
                    calls.append(task.task_id)
                    text = responses[task.task_id]
                    results[task.task_id] = text
                    if on_task_complete is not None:
                        on_task_complete(task.task_id, text)
                return results

            yield generate

        return fake_generator

    first_calls: list[str] = []
    monkeypatch.setattr(
        extraction_steps,
        "_qwen_generator",
        generator_for(
            {"c1:0": "plain prose", "c2:0": json.dumps(valid_graph)},
            first_calls,
        ),
    )
    with pytest.raises(extraction_steps.ExtractionStepError, match="1 scene"):
        extraction_steps.extract_graph_scenes(context, model="qwen")

    assert first_calls == ["c1:0", "c2:0"]
    assert read_jsonl(context.graph_scene_dir("qwen") / "c1.jsonl") == []
    failure = read_jsonl(context.graph_failure_dir("qwen") / "c1.jsonl")[0]
    assert failure["parse_status"] == "failed"
    assert failure["raw_response"] == "plain prose"
    assert read_jsonl(context.graph_scene_dir("qwen") / "c2.jsonl")[0]["parse_status"] == "parsed"
    assert not context.stage_manifest("extract-graph-scenes-qwen").exists()
    assert not context.stage_manifest("summarize-graph-qwen").exists()
    output = capsys.readouterr()
    assert "[Graph_skip_qwen] 1.mp4 | scene #000" in output.err
    assert "[Graph_qwen] 2.mp4 | scene #000" in output.err

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("resume must not load the model")
        yield

    monkeypatch.setattr(extraction_steps, "_qwen_generator", unexpected_generator)
    with pytest.raises(extraction_steps.ExtractionStepError, match="1 scene"):
        extraction_steps.extract_graph_scenes(context, model="qwen")

    force_calls: list[str] = []
    monkeypatch.setattr(
        extraction_steps,
        "_qwen_generator",
        generator_for(
            {
                "c1:0": json.dumps(valid_graph),
                "c2:0": json.dumps(valid_graph),
            },
            force_calls,
        ),
    )
    result = extraction_steps.extract_graph_scenes(context, model="qwen", force=True)

    assert result["status"] == "complete"
    assert force_calls == ["c1:0", "c2:0"]
    assert read_jsonl(context.graph_failure_dir("qwen") / "c1.jsonl") == []
    assert context.stage_manifest("extract-graph-scenes-qwen").is_file()


def test_gemini_graph_branch_persists_api_failure_and_uses_qwen_summary(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context.initialize()
    write_json(
        context.stage_manifest("prepare-input-data"),
        {
            "schema_version": "step-manifest/v1",
            "run_id": context.run_id,
            "stage": "prepare-input-data",
            "status": "complete",
            "source_fingerprints": {},
            "output_fingerprint": "visual-v1",
        },
    )
    visual_rows = []
    for content_id in ("c1", "c2"):
        frames = context.evidence_dir / "resized_keyframes" / content_id
        frames.mkdir(parents=True)
        for timestamp in (5, 15, 25):
            (frames / f"{timestamp:04d}.png").write_bytes(b"fixture")
        timestamp_path = context.cohort_dir / f"{content_id}.json"
        timestamp_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp_path.write_text(
            json.dumps([{
                "scene_start": 0,
                "scene_end": 30,
                "keyframe_timestamps": [5, 15, 25],
            }]),
            encoding="utf-8",
        )
        visual_rows.append({
            "content_id": content_id,
            "frames_dir": str(frames),
            "timestamp_json": str(timestamp_path),
            "evidence_fingerprint": f"evidence-{content_id}",
        })
    write_jsonl(context.visual_manifest, visual_rows)
    write_jsonl(
        context.cohort_dir / "catalog.jsonl",
        [
            {"content_id": "c1", "item_id": "i1", "source_video_path": "1.mp4"},
            {"content_id": "c2", "item_id": "i2", "source_video_path": "2.mp4"},
        ],
    )
    graph = json.dumps({"entities": [], "events": []})
    pool_settings: list[tuple[int, str, str, str]] = []

    class FailingPool:
        def __init__(self, concurrency, *, project_id, location, model_id):
            pool_settings.append((concurrency, project_id, location, model_id))

        def generate(self, tasks, callback):
            for task in reversed(list(tasks)):
                assert [Path(path).name for path in task.image_paths] == [
                    "0005.png",
                    "0015.png",
                    "0025.png",
                ]
                error = "RuntimeError: api unavailable" if task.task_id == "c1:0" else None
                callback(extraction_steps.GeminiGenerationOutcome(
                    task.task_id,
                    "" if error else graph,
                    error,
                ))

    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", FailingPool)
    with pytest.raises(extraction_steps.ExtractionStepError, match="1 scene"):
        extraction_steps.extract_graph_scenes(context, model="gemini")

    failure = read_jsonl(context.graph_failure_dir("gemini") / "c1.jsonl")[0]
    assert failure["failure_kind"] == "generation"
    assert failure["raw_response"] == ""
    assert read_jsonl(context.graph_scene_dir("gemini") / "c2.jsonl")[0][
        "graph_source"
    ] == "gemini"
    assert pool_settings == [(4, "test-project", "global", "test-gemini")]
    output = capsys.readouterr().err
    assert output.index("[Graph_gemini] 2.mp4") < output.index(
        "[Graph_skip_gemini] 1.mp4"
    )

    class SuccessfulPool(FailingPool):
        def generate(self, tasks, callback):
            for task in reversed(list(tasks)):
                callback(extraction_steps.GeminiGenerationOutcome(task.task_id, graph))

    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", SuccessfulPool)
    extraction_steps.extract_graph_scenes(context, model="gemini", force=True)

    summary_calls: list[str] = []

    @contextmanager
    def fake_qwen_generator(**_kwargs):
        def generate(tasks, on_task_complete=None):
            results = {}
            for task in tasks:
                summary_calls.append(task.task_id)
                results[task.task_id] = "short qwen summary"
                if on_task_complete is not None:
                    on_task_complete(task.task_id, results[task.task_id])
            return results

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_qwen_generator)
    extraction_steps.summarize_graph(context, source="gemini")
    summary = read_json(context.graph_summary_dir("gemini") / "c1.json")
    assert summary_calls == ["c1", "c2"]
    assert summary["graph_source"] == "gemini"
    assert summary["summary_model"] == "qwen"
    assert context.stage_manifest("summarize-graph-gemini").is_file()


def test_independent_step_does_not_auto_run_prerequisites(context: RunContext) -> None:
    context.initialize()
    with pytest.raises(RuntimeError, match="requires completed stages"):
        execute_stage(context, "extract-graph-scenes-qwen")
    assert not context.stage_manifest("prepare-cohort").exists()


def test_root_cli_has_no_config_override_flags() -> None:
    with pytest.raises(SystemExit):
        pipeline_cli(["run", "--run-id", "demo", "--config", "other.yaml"])


def test_root_cli_has_no_resume_flag() -> None:
    with pytest.raises(SystemExit):
        pipeline_cli(["run", "--run-id", "demo", "--resume"])


def test_root_cli_forwards_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    received = {}
    context = object()
    monkeypatch.setattr(
        "viewing_context_pipeline.cli.RunContext.load",
        lambda _: context,
    )

    def fake_run_pipeline(ctx, **kwargs):
        received.update(context=ctx, **kwargs)
        return 0

    monkeypatch.setattr(
        "viewing_context_pipeline.cli.run_pipeline",
        fake_run_pipeline,
    )
    assert pipeline_cli(["run", "--run-id", "demo", "--gpus", "2"]) == 0
    assert received["context"] is context
    assert received["gpus"] == 2


def test_dry_run_writes_no_artifacts(context: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "preflight", lambda _: {"ready": True, "checks": {}})
    assert run_pipeline(context, dry_run=True) == 0
    assert not context.run_root.exists()


def test_preflight_includes_gemini_dependency_and_vertex_adc(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(pipeline_module, "_vertex_adc_available", lambda: True)
    result = preflight(context)

    assert result["checks"]["python.google_genai"] is True
    assert result["checks"]["gemini.vertex_adc"] is True


def test_synthetic_full_runner_records_exact_v1_dag(context: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "preflight", lambda _: {"ready": True, "checks": {}})

    def fake_handlers():
        def build(stage: str):
            def run(ctx: RunContext, *, force: bool = False):
                if ctx.stage_manifest(stage).is_file() and not force:
                    return json.loads(ctx.stage_manifest(stage).read_text(encoding="utf-8"))
                calls.append(stage)
                document = {
                    "schema_version": "step-manifest/v1",
                    "run_id": ctx.run_id,
                    "stage": stage,
                    "status": "complete",
                    "source_fingerprints": {},
                    "output_fingerprint": (str(len(calls)) * 64)[:64],
                }
                write_json(ctx.stage_manifest(stage), document)
                return document
            return run
        return {stage: build(stage) for stage in STAGES}

    monkeypatch.setattr(pipeline_module, "handlers", fake_handlers)
    assert run_pipeline(context) == 0
    assert calls == list(STAGES)
    assert run_pipeline(context) == 0
    assert calls == list(STAGES)
    manifest = json.loads(context.pipeline_manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "pipeline-run/v1"
    assert manifest["complete"] is True


def test_resume_runs_only_missing_downstream_stages(
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context.initialize()
    completed = STAGES[:3]
    for stage in completed:
        write_json(
            context.stage_manifest(stage),
            {
                "schema_version": "step-manifest/v1",
                "run_id": context.run_id,
                "stage": stage,
                "status": "complete",
                "source_fingerprints": {},
                "output_fingerprint": stage,
            },
        )
    calls: list[str] = []

    def fake_handlers():
        def build(stage: str):
            def run(ctx: RunContext, *, force: bool = False):
                path = ctx.stage_manifest(stage)
                if path.is_file() and not force:
                    return json.loads(path.read_text(encoding="utf-8"))
                calls.append(stage)
                document = {
                    "schema_version": "step-manifest/v1",
                    "run_id": ctx.run_id,
                    "stage": stage,
                    "status": "complete",
                    "source_fingerprints": {},
                    "output_fingerprint": stage,
                }
                write_json(path, document)
                return document

            return run

        return {stage: build(stage) for stage in STAGES}

    monkeypatch.setattr(pipeline_module, "preflight", lambda _: {"ready": True, "checks": {}})
    monkeypatch.setattr(pipeline_module, "handlers", fake_handlers)
    assert run_pipeline(context) == 0
    assert calls == list(STAGES[len(completed):])
