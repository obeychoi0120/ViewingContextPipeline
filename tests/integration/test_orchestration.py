from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

import scripts._orchestration as orchestration
from scripts._orchestration import (
    STAGES,
    PipelineContext,
    PipelineError,
    command_for_stage,
    execute_stage,
    initialize_run,
    outputs_for_stage,
    runtime_documents,
    runner_main,
    write_runtime_configs,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def context(tmp_path: Path) -> PipelineContext:
    data = tmp_path / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    for name in ("pairs.tsv", "titles.csv", "tags.csv"):
        (data / name).write_text("fixture\n", encoding="utf-8")
    qwen = tmp_path / "models" / "qwen"
    bge = tmp_path / "models" / "bge"
    qwen.mkdir(parents=True)
    bge.mkdir(parents=True)
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({
        "schema_version": "viewing-context-local/v1",
        "data": {
            "videos_dir": str(videos),
            "titles_csv": str(data / "titles.csv"),
            "tags_csv": str(data / "tags.csv"),
            "pairs_tsv": str(data / "pairs.tsv"),
        },
        "models": {"qwen": str(qwen), "bge": str(bge)},
    }), encoding="utf-8")
    pipeline = yaml.safe_load((ROOT / "config/pipelines/microlens_graph_vs_desc_pilot.yaml").read_text(encoding="utf-8"))
    pipeline["artifacts_root"] = str(tmp_path / "artifacts")
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    return PipelineContext.load(config, local, root=ROOT)


def test_runtime_documents_pin_fresh_visual_only_contract(context: PipelineContext) -> None:
    documents = runtime_documents(context)
    assert documents["microlens"]["output_root"] == str(context.extraction_output)
    assert documents["microlens"]["source"]["selection_jsonl"] == str(context.selection_path)
    assert documents["processing"]["shot_interval"] == "fixed_30s"
    assert documents["processing"]["asr_config"]["enabled"] is False
    assert documents["processing"]["ocr_config"]["enabled"] is False
    assert documents["graph"]["multimodal"] is False
    assert documents["description"]["multimodal"] is False
    assert documents["validation"]["dataset"]["vp_graph_dir"] == str(context.graph_profile_dir)
    assert documents["validation"]["dataset"]["vp_desc_dir"] == str(context.description_profile_dir)
    serialized = json.dumps(documents)
    assert "ViewingContextExtraction/output" not in serialized


def test_root_and_direct_stage_use_one_command_builder(context: PipelineContext) -> None:
    command = command_for_stage(context, "extract-graph")
    assert command.cwd == ROOT / "extraction"
    assert "--settings" in command.argv
    assert str(context.runtime_paths["graph"]) in command.argv
    assert command.env["OUTPUT_SAVE_PATH"] == str(context.extraction_output)


def test_pipeline_stage_order_has_no_cache_resolution_step() -> None:
    assert STAGES == (
        "preflight",
        "prepare-cohort",
        "import-microlens",
        "extract-graph",
        "build-graph-profiles",
        "build-description-profiles",
        "materialize-representations",
        "run-experiment",
    )


def test_dry_run_does_not_create_artifacts(context: PipelineContext) -> None:
    result = execute_stage(context, "import-microlens", dry_run=True)
    assert result["dry_run"] is True
    assert not context.run_root.exists()


def test_independent_stage_rejects_missing_prerequisite(context: PipelineContext) -> None:
    context.run_root.mkdir(parents=True)
    write_runtime_configs(context)
    with pytest.raises(PipelineError, match="missing prerequisite"):
        execute_stage(context, "import-microlens")


def test_fresh_run_never_overwrites_nonempty_directory(context: PipelineContext) -> None:
    context.run_root.mkdir(parents=True)
    (context.run_root / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(PipelineError, match="not empty"):
        initialize_run(context)
    assert (context.run_root / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_resume_skips_completed_stage_without_subprocess(context: PipelineContext) -> None:
    initialize_run(context)
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["prepare-cohort"] = {"status": "complete"}
    context.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = execute_stage(context, "prepare-cohort", resume=True)
    assert result == {"stage": "prepare-cohort", "status": "skipped", "reason": "already complete"}


def test_synthetic_root_runner_handoff(
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_preflight(_: PipelineContext) -> dict[str, object]:
        return {"schema_version": "viewing-context-preflight/v1", "ready": True, "checks": {}}

    def fake_run(argv, *, cwd, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, stdout="test-head\n", stderr="")
        module = argv[2]
        command = argv[-1]
        if module == "vc_validation.cli" and command == "prepare-cohort":
            stage = "prepare-cohort"
            context.selection_path.parent.mkdir(parents=True, exist_ok=True)
            context.selection_path.write_text("{}\n", encoding="utf-8")
        elif module == "src.video_data_collection.cli":
            stage = "import-microlens"
            context.import_manifest.parent.mkdir(parents=True, exist_ok=True)
            context.import_manifest.write_text("content_id,url\n", encoding="utf-8")
        elif module == "src.scene_context_extraction.ondevice.cli":
            stage = "extract-graph"
            output = outputs_for_stage(context, stage)[0]
            output.mkdir(parents=True, exist_ok=True)
        elif module == "src.scene_description_generation.graph_profile_cli":
            stage = "build-graph-profiles"
            context.graph_profile_dir.mkdir(parents=True, exist_ok=True)
        elif module == "src.scene_description_generation.cli":
            stage = "build-description-profiles"
            context.description_profile_dir.mkdir(parents=True, exist_ok=True)
        elif module == "vc_validation.cli" and command == "materialize-representations":
            stage = "materialize-representations"
            output = outputs_for_stage(context, stage)[0]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
        elif module == "vc_validation.cli" and command == "run-experiment":
            stage = "run-experiment"
            for output in outputs_for_stage(context, stage):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("{}", encoding="utf-8")
        else:
            raise AssertionError(argv)
        calls.append(stage)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(orchestration, "preflight", fake_preflight)
    monkeypatch.setattr(orchestration.subprocess, "run", fake_run)
    assert runner_main([
        "--config", str(context.pipeline_path),
        "--local-config", str(context.local_path),
        "--stage", "all",
    ]) == 0
    capsys.readouterr()
    assert calls == list(STAGES[1:])
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert all(row["status"] == "complete" for row in manifest["stages"].values())
