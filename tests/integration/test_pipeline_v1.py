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
from viewing_context_pipeline.pipeline import STAGES, descendants, execute_stage, run_pipeline
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
        (ROOT / "config/pipeline.example.yaml").read_text(encoding="utf-8")
    )
    config["artifacts_root"] = str(tmp_path / "artifacts")
    config["data"] = {
        "videos_dir": str(videos),
        "titles_csv": str(data / "titles.csv"),
        "tags_csv": str(data / "tags.csv"),
        "pairs_tsv": str(data / "pairs.tsv"),
    }
    config["models"] = {"qwen": str(models / "qwen"), "bge": str(models / "bge")}
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
        "extract-graph-scenes",
        "summarize-graph",
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


def test_resume_refuses_nonempty_directory_without_runtime_snapshot(context: RunContext) -> None:
    context.run_root.mkdir(parents=True)
    marker = context.run_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ConfigError, match="no v1 runtime snapshot"):
        context.initialize()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_resume_rejects_changed_config_for_same_run(context: RunContext) -> None:
    context.initialize()
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["validation"]["model"]["dropout"] = 0.2
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    changed = RunContext.load(context.run_id, root=context.root)
    with pytest.raises(ConfigError, match="does not match the run snapshot"):
        changed.initialize()


def test_graph_force_does_not_invalidate_description_sibling() -> None:
    affected = descendants({"extract-graph-scenes"})
    assert "summarize-graph" in affected
    assert "embed-representations" in affected
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
        "extract-graph-scenes",
        source_fingerprints={"taxonomy": "changed"},
        output_fingerprint="b" * 64,
    )
    assert context.stage_manifest("extract-description-scenes").is_file()
    assert context.stage_manifest("summarize-description").is_file()
    assert not context.stage_manifest("summarize-graph").exists()
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
        ("summarize-graph", "graph-v1"),
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
        context.graph_summary_dir / "c1.json",
        {**common, "schema_version": "graph-video-summary/v1", "text": "graph one"},
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
    assert encoded == [["graph one"], ["description one"]]

    write_json(
        context.graph_summary_dir / "c1.json",
        {**common, "schema_version": "graph-video-summary/v1", "text": "graph two"},
    )
    graph_stage = read_json(context.stage_manifest("summarize-graph"))
    graph_stage["output_fingerprint"] = "graph-v2"
    write_json(context.stage_manifest("summarize-graph"), graph_stage)

    validation_steps.embed_representations(context)
    second = read_json(context.representations_manifest)

    assert encoded == [["graph one"], ["description one"], ["graph two"]]
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
    empty_graph = {
        "setting_context": "unknown",
        "entities": [],
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
                    else json.dumps(empty_graph)
                )
                results[task.task_id] = text
                if on_task_complete is not None:
                    on_task_complete(task.task_id, text)
            return results

        yield generate

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)

    extraction_steps.extract_graph_scenes(context)
    scene_path = context.graph_scene_dir / "c1.jsonl"
    scene = read_jsonl(scene_path)[0]
    assert scene["schema_version"] == "minimal-semantic-scene/v1"
    assert scene["scene_start_seconds"] == 0
    assert scene["scene_end_seconds"] == 30
    assert scene["keyframes"] == [5, 15, 25]
    assert scene["graph"] == empty_graph
    assert scene["parse_status"] == "parsed"
    assert {
        "taxonomy_fingerprint",
        "prompt_fingerprint",
        "model_fingerprint",
        "evidence_fingerprint",
        "input_fingerprint",
        "repair_fingerprint",
    }.issubset(scene)

    legacy = dict(scene)
    legacy.pop("parse_status")
    legacy.pop("repair_fingerprint")
    write_jsonl(scene_path, [legacy])
    context.stage_manifest("extract-graph-scenes").unlink()

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("legacy migration must not load the model")
        yield

    monkeypatch.setattr(extraction_steps, "_qwen_generator", unexpected_generator)
    extraction_steps.extract_graph_scenes(context)
    migrated = read_jsonl(scene_path)[0]
    assert migrated["parse_status"] == "legacy_parser"
    assert migrated["repair_fingerprint"]

    monkeypatch.setattr(extraction_steps, "_qwen_generator", fake_generator)
    extraction_steps.summarize_graph(context)
    summary = read_json(context.graph_summary_dir / "c1.json")
    assert summary["text"] == "short factual summary"
    assert summary["scene_graph_path"] == "extraction/graph/scenes/c1.jsonl"
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
    for stage in ("extract-graph-scenes", "summarize-graph"):
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
    with pytest.raises(extraction_steps.ExtractionStepError, match="1 graph scene"):
        extraction_steps.extract_graph_scenes(context)

    assert first_calls == ["c1:0", "c2:0"]
    assert read_jsonl(context.graph_scene_dir / "c1.jsonl") == []
    failure = read_jsonl(context.graph_failure_dir / "c1.jsonl")[0]
    assert failure["parse_status"] == "failed"
    assert failure["raw_response"] == "plain prose"
    assert read_jsonl(context.graph_scene_dir / "c2.jsonl")[0]["parse_status"] == "parsed"
    assert not context.stage_manifest("extract-graph-scenes").exists()
    assert not context.stage_manifest("summarize-graph").exists()
    output = capsys.readouterr()
    assert "[Graph_skip] 1.mp4 | scene #000" in output.err
    assert "[Graph] 2.mp4 | scene #000" in output.err

    @contextmanager
    def unexpected_generator(**_kwargs):
        raise AssertionError("resume must not load the model")
        yield

    monkeypatch.setattr(extraction_steps, "_qwen_generator", unexpected_generator)
    with pytest.raises(extraction_steps.ExtractionStepError, match="1 graph scene"):
        extraction_steps.extract_graph_scenes(context)

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
    result = extraction_steps.extract_graph_scenes(context, force=True)

    assert result["status"] == "complete"
    assert force_calls == ["c1:0", "c2:0"]
    assert read_jsonl(context.graph_failure_dir / "c1.jsonl") == []
    assert context.stage_manifest("extract-graph-scenes").is_file()


def test_independent_step_does_not_auto_run_prerequisites(context: RunContext) -> None:
    context.initialize()
    with pytest.raises(RuntimeError, match="requires completed stages"):
        execute_stage(context, "extract-graph-scenes")
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
    snapshot = json.loads(context.runtime_path.read_text(encoding="utf-8"))
    assert set(snapshot) == {
        "schema_version",
        "run_id",
        "config_path",
        "config",
        "config_fingerprint",
    }


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
