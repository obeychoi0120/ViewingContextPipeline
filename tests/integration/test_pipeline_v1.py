from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

import viewing_context_pipeline.pipeline as pipeline_module
from extraction.steps import _write_stage as write_extraction_stage
from viewing_context_pipeline.cli import main as pipeline_cli
from viewing_context_pipeline.pipeline import STAGES, descendants, execute_stage, run_pipeline
from viewing_context_pipeline.runtime import ConfigError, RunContext, write_json


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
    config["extraction"]["graph"]["ontology"] = str(
        ROOT / "contracts/extraction/relational_graph_ontology_v1.json"
    )
    for arm in ("graph", "description"):
        for key in ("scene_prompt", "summary_prompt"):
            config["extraction"][arm][key] = str(
                ROOT / config["extraction"][arm][key]
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
        source_fingerprints={"ontology": "changed"},
        output_fingerprint="b" * 64,
    )
    assert context.stage_manifest("extract-description-scenes").is_file()
    assert context.stage_manifest("summarize-description").is_file()
    assert not context.stage_manifest("summarize-graph").exists()
    assert not context.stage_manifest("embed-representations").exists()


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
