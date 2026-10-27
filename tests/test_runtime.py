from pathlib import Path

import pytest

from arm_registry import generation_registry, registry, select_arms
from validation.steps import validation_config


def test_current_config_loads(context):
    settings = validation_config(context)
    assert settings.encoder.embedding_dim == 1024
    assert settings.model.embedding_dim == 512
    assert settings.model.seeds == [42, 43, 44]
    assert settings.model.max_sequence_length == 10
    assert context.run_root == context.root / "artifacts" / "runs" / "regression" / "microlens_100k"
    paths = [context.path("data", key) for key in context.data]
    paths += [context.path("models", key) for key in ("qwen3vl", "qwen3.5", "bge")]
    assert all(path.is_relative_to(context.root) for path in paths)
    assert not any(str(path).startswith("/home_nvme") for path in paths)
    assert Path(settings.dataset.pairs_csv).is_file()


def test_empty_run_directories_can_be_initialized(context):
    from pipeline_runtime import RunContext

    directory = context.artifact_root / "runs" / "empty-run" / context.dataset
    (directory / "extraction" / "scenes").mkdir(parents=True)
    (directory / "validation").mkdir()
    fresh = RunContext.load("empty-run", dataset=context.dataset, root=context.root)
    fresh.initialize()
    assert not (directory / "run_contract.json").exists()


def test_existing_results_without_contract_are_preserved(context):
    from pipeline_runtime import RunContext

    directory = context.artifact_root / "runs" / "261027_v14" / context.dataset
    (directory / "extraction").mkdir(parents=True)
    result = directory / "extraction" / "result.json"
    result.write_text('{"historical": true}')
    selected = RunContext.load("261027_v14", dataset=context.dataset, root=context.root)
    selected.initialize()
    assert result.read_text() == '{"historical": true}'
    assert not (directory / "run_contract.json").exists()


def test_legacy_run_contract_is_ignored_and_preserved(context):
    from pipeline_runtime import RunContext

    context.run_root.mkdir(parents=True)
    path = context.run_root / "run_contract.json"
    path.write_text("historical metadata; not read by the pipeline")
    selected = RunContext.load(context.run_id, dataset=context.dataset, root=context.root)
    selected.initialize()
    assert path.read_text() == "historical metadata; not read by the pipeline"


def test_arm_sources_and_title_policy(context):
    from arm_registry import RECOMMENDATION_ARM_CONTRACT, feature_source, target_set

    arms = registry(context.config)
    assert list(arms) == ["base", "d0", "d1", "g0", "g1", "a1", "a2", "a3"]
    assert all(arm.uses_title for arm in arms.values())
    assert [feature_source(arm) for arm in arms.values()] == [
        "metadata", "desc_qwen3vl", "desc_qwen3vl", *(["graph_qwen3vl"] * 5),
    ]
    sources = generation_registry(context.config)
    assert all(not arm.uses_title and arm.name == arm.scene_arm for arm in sources.values())
    assert "graph_minicpm" in sources and "graph_gemini" in sources
    assert list(select_arms(context.config, ["g1", "base"])) == ["base", "g1"]
    assert target_set(context.config, ["g1", "base"]) == "base_g1"
    assert RECOMMENDATION_ARM_CONTRACT == "gnn-attn-aggregation/v1"
    assert context.representations_dir == context.validation_dir / "representations"
    assert context.recommendations_dir == context.validation_dir / "recommendations"
    assert context.diagnosis_path(["g1", "base"]) == (
        context.validation_dir / "diagnosis" / "base_g1" / "diagnosis.json"
    )
    for targets in ([], ["base", "base"], ["meta"], ["graph_minicpm"], ["missing"]):
        with pytest.raises(ValueError):
            select_arms(context.config, targets)


def test_cli_mixed_arms_and_removed_options(context, monkeypatch):
    from pipeline_runtime import RunContext
    from validation.cli import STEP_HANDLERS, main

    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)
    calls = []
    monkeypatch.setitem(STEP_HANDLERS, "run-recommendation", lambda ctx, **kw: calls.append(kw))
    command = ["run-recommendation", "--run-id", context.run_id, "--dataset", context.dataset, "--target", "d1", "a3"]
    assert main(command) == 0
    assert calls == [{"force": False, "target": ["d1", "a3"]}]
    for option in (["--representation-mode", "graph"], ["--scene-aggregation", "mean"],
                   ["--step", "ablation"]):
        with pytest.raises(SystemExit):
            main(command + option)
    assert main(["run-recommendation", "--run-id", context.run_id, "--dataset", context.dataset, "--target", "graph_minicpm"]) == 1
