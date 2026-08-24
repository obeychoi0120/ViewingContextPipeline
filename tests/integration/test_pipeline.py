from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess

import pytest
import yaml

import viewing_context_pipeline.pipeline as pipeline_module
from viewing_context_pipeline.pipeline import STAGES, PipelineContext, PipelineError, descendants, generate_run_id, initialize_run, outputs_for_stage, runner_main


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "260824_0938"


@pytest.fixture()
def context(tmp_path: Path) -> PipelineContext:
    data = tmp_path / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    for name in ("pairs.tsv", "titles.csv", "tags.csv"):
        (data / name).write_text("fixture\n", encoding="utf-8")
    models = tmp_path / "models"
    for name in ("qwen", "bge", "asr"):
        (models / name).mkdir(parents=True)
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({
        "schema_version": "viewing-context-local/v1",
        "data": {"videos_dir": str(videos), "titles_csv": str(data / "titles.csv"), "tags_csv": str(data / "tags.csv"), "pairs_tsv": str(data / "pairs.tsv")},
        "models": {name: str(models / name) for name in ("qwen", "bge", "asr")},
        "cloud": {"gcp_project_id": "fixture-project"},
    }), encoding="utf-8")
    pipeline = yaml.safe_load((ROOT / "config/pipelines/microlens_graph_vs_desc_pilot.yaml").read_text(encoding="utf-8"))
    pipeline["artifacts_root"] = str(tmp_path / "artifacts")
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    return PipelineContext.load(config, local, run_id=RUN_ID, allow_generate=False, root=ROOT)


def test_public_stage_order_is_exactly_v1_contract() -> None:
    assert STAGES == ("prepare_data", "extract_ondevice_graph_context", "extract_ondevice_desc_context", "extract_gemini_graph_context", "extract_gemini_desc_context", "embed_representations", "run_recommendation", "run_diagnosis")


def test_run_id_uses_asia_seoul_minute_format() -> None:
    assert generate_run_id(datetime(2026, 8, 24, 9, 38)) == RUN_ID


def test_run_id_accepts_user_defined_directory_name(context: PipelineContext) -> None:
    named = PipelineContext.load(
        context.pipeline_path,
        context.local_path,
        run_id="1k_pilot_260824",
        allow_generate=False,
        root=ROOT,
    )
    assert named.run_id == "1k_pilot_260824"


def test_run_id_rejects_paths(context: PipelineContext) -> None:
    with pytest.raises(PipelineError, match="single directory name"):
        PipelineContext.load(
            context.pipeline_path,
            context.local_path,
            run_id="../outside",
            allow_generate=False,
            root=ROOT,
        )


def test_run_sh_requires_explicit_positional_run_id() -> None:
    script = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert 'if [[ $# -eq 0 || -z "$1" ]]' in script
    assert 'RUN_ID="$1"' in script
    assert 'ARGS+=(--run-id "$RUN_ID")' in script


def test_downstream_context_requires_explicit_run_id(context: PipelineContext) -> None:
    with pytest.raises(PipelineError, match="--run-id is required"):
        PipelineContext.load(context.pipeline_path, context.local_path, run_id=None, allow_generate=False, root=ROOT)


def test_artifact_layout_and_v1_runtime(context: PipelineContext) -> None:
    initialize_run(context)
    runtime = json.loads(context.runtime_path.read_text(encoding="utf-8"))
    assert runtime["schema_version"] == "pipeline-runtime/v1"
    assert context.visual_manifest == context.run_root / "data/fixed_30s/visual_manifest.jsonl"
    assert context.multimodal_ref_dir == context.run_root / "data/fixed_30s/multimodal_ref"
    assert context.context_dir("ondevice_graph") == context.run_root / "extraction/contexts/visual_only/ondevice_graph"
    assert "ViewingContextExtraction/output" not in json.dumps(runtime)


def test_fresh_run_refuses_nonempty_collision(context: PipelineContext) -> None:
    context.run_root.mkdir(parents=True)
    marker = context.run_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(PipelineError, match="will not be overwritten"):
        initialize_run(context)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_force_branch_only_invalidates_actual_descendants(context: PipelineContext) -> None:
    affected = descendants(context, {"extract_ondevice_graph_context"})
    assert affected == {"extract_ondevice_graph_context", "embed_representations", "run_recommendation", "run_diagnosis"}
    assert "extract_ondevice_desc_context" not in affected
    assert "extract_gemini_graph_context" not in affected


def test_optional_gemini_description_is_not_in_enabled_dag(context: PipelineContext) -> None:
    assert "extract_gemini_desc_context" not in context.enabled_stages
    assert outputs_for_stage(context, "extract_gemini_desc_context")[0].parent.name == "gemini_desc"


def test_runner_rejects_forcing_disabled_optional_branch(context: PipelineContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "preflight", lambda _: {"ready": True, "checks": {}})
    with pytest.raises(SystemExit):
        runner_main([
            "--config", str(context.pipeline_path), "--local-config", str(context.local_path),
            "--run-id", RUN_ID, "--stage", "all", "--force-stage", "extract_gemini_desc_context",
        ])
    assert not context.run_root.exists()


def test_dry_run_creates_no_artifacts(context: PipelineContext) -> None:
    result = pipeline_module.execute_stage(context, "prepare_data", dry_run=True)
    assert result["dry_run"] is True
    assert not context.run_root.exists()


def test_stage_subprocess_can_import_src_package_without_install(context: PipelineContext) -> None:
    command = pipeline_module.command_for_stage(context, "prepare_data")
    assert command.env["PYTHONPATH"].split(pipeline_module.os.pathsep)[0] == str(ROOT / "src")
    completed = subprocess.run(
        [pipeline_module.sys.executable, "-c", "import viewing_context_pipeline"],
        cwd=ROOT,
        env=command.env,
        check=False,
    )
    assert completed.returncode == 0


def test_synthetic_runner_records_only_enabled_stages(context: PipelineContext, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "preflight", lambda _: {"schema_version": "pipeline-preflight/v1", "ready": True, "checks": {}})

    def fake_run(argv, *, cwd, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, stdout="test-head\n", stderr="")
        stage = argv[-1]
        calls.append(stage)
        schema = "video-context-manifest/v1" if stage.startswith("extract_") else {"prepare_data": "prepared-data/v1", "embed_representations": "representations/v1", "run_recommendation": "recommendations/v1", "run_diagnosis": "diagnosis/v1"}[stage]
        payload = {"schema_version": schema, "run_id": RUN_ID, "modality": "visual_only", "complete": True}
        if stage == "run_diagnosis":
            payload = {"schema_version": schema, "run_id": RUN_ID, "modality": "visual_only", "report_ready": True}
        output = outputs_for_stage(context, stage)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(pipeline_module.subprocess, "run", fake_run)
    assert runner_main(["--config", str(context.pipeline_path), "--local-config", str(context.local_path), "--run-id", RUN_ID, "--stage", "all"]) == 0
    assert calls == list(context.enabled_stages)
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert all(row["status"] == "complete" for row in manifest["stages"].values())


def test_synthetic_multimodal_runner_keeps_one_run_level_modality(context: PipelineContext, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = yaml.safe_load(context.pipeline_path.read_text(encoding="utf-8"))
    pipeline["protocol"]["modality"] = "multimodal"
    context.pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    multimodal = PipelineContext.load(context.pipeline_path, context.local_path, run_id=RUN_ID, allow_generate=False, root=ROOT)
    monkeypatch.setattr(pipeline_module, "preflight", lambda _: {"schema_version": "pipeline-preflight/v1", "ready": True, "checks": {}})

    def fake_run(argv, *, cwd, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, stdout="test-head\n", stderr="")
        stage = argv[-1]
        schema = "video-context-manifest/v1" if stage.startswith("extract_") else {"prepare_data": "prepared-data/v1", "embed_representations": "representations/v1", "run_recommendation": "recommendations/v1", "run_diagnosis": "diagnosis/v1"}[stage]
        payload = {"schema_version": schema, "run_id": RUN_ID, "modality": "multimodal", "complete": True}
        if stage == "run_diagnosis":
            payload = {"schema_version": schema, "run_id": RUN_ID, "modality": "multimodal", "report_ready": True}
        output = outputs_for_stage(multimodal, stage)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(pipeline_module.subprocess, "run", fake_run)
    assert runner_main(["--config", str(multimodal.pipeline_path), "--local-config", str(multimodal.local_path), "--run-id", RUN_ID, "--stage", "all"]) == 0
    manifest = json.loads(multimodal.manifest_path.read_text(encoding="utf-8"))
    assert manifest["modality"] == "multimodal"
    assert all(row["status"] == "complete" for row in manifest["stages"].values())
