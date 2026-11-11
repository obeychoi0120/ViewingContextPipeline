from pathlib import Path

import pytest

from arm_registry import generation_registry, registry, select_arms
from validation.steps import validation_config


def test_current_config_and_extractor_protocol_load(context):
    from copy import deepcopy
    from pipeline_runtime import _validate_config

    settings = validation_config(context)
    assert settings.encoder.embedding_dim == 1024
    assert settings.model.embedding_dim == 512
    assert settings.model.seeds == [42, 43, 44]
    assert settings.model.max_sequence_length == 10
    assert context.run_root == context.root / "artifacts" / "runs" / "regression" / "microlens_100k"
    paths = [context.path("data", key) for key in context.data]
    paths += [context.path("models", key) for key in ("qwen3vl_2b", "qwen3vl_8b", "bge")]
    assert all(path.is_relative_to(context.root) for path in paths)
    assert not any(str(path).startswith("/home_nvme") for path in paths)
    assert Path(settings.dataset.pairs_csv).is_file()

    config = deepcopy(context.config)
    config.pop("_rec_experiments")
    config["protocol"]["graph_extractors"] = ["qwen3vl_2b", "qwen3vl_8b", "gemini"]
    config["protocol"]["description_extractors"] = ["qwen3vl_2b", "qwen3vl_8b", "gemini"]
    _validate_config(config)


def test_arm_sources_and_title_policy(context):
    from arm_registry import RECOMMENDATION_ARM_CONTRACT, feature_source, target_set

    arms = registry(context.config)
    assert list(arms) == [
        "base", "description", "graph", "graph-activity", "graph-role-activity",
    ]
    assert all(arm.uses_title for arm in arms.values())
    assert [feature_source(arm) for arm in arms.values()] == [
        "metadata", "desc_qwen3vl_2b", *(["graph_qwen3vl_2b"] * 3),
    ]
    sources = generation_registry(context.config)
    from validation.diagnosis_scenes import scene_arms

    for name, arm in sources.items():
        expected = context.run_root / "extraction" / name
        assert context.scene_arm_dir(name) == expected
        assert context.extraction_dir(arm.representation, arm.model, "scenes") == expected
    for name, (scene_path, failure_path) in scene_arms(context.config).items():
        assert context.run_root / scene_path == context.scene_arm_dir(arms[name].scene_arm)
        assert context.run_root / failure_path == context.scene_arm_dir(arms[name].scene_arm) / "failures"
    assert all(not arm.uses_title and arm.name == arm.scene_arm for arm in sources.values())
    assert set(sources) == {
        "graph_qwen3vl_2b", "desc_qwen3vl_2b", "graph_qwen3vl_8b", "desc_qwen3vl_8b",
        "graph_gemini", "desc_gemini",
    }
    assert list(select_arms(context.config, ["graph-activity", "base"])) == ["base", "graph-activity"]
    assert target_set(context.config, ["graph-activity", "base"]) == "base_graph-activity"
    assert RECOMMENDATION_ARM_CONTRACT == "gnn-attn-aggregation/v1"
    assert all(arms[name].aggregation == "mean" for name in ("description", "graph", "graph-activity", "graph-role-activity"))
    assert [arms[name].role_fusion for name in ("graph", "graph-activity", "graph-role-activity")] == [False, False, True]
    assert [arms[name].extensions for name in ("graph", "graph-activity", "graph-role-activity")] == [(), ("segments",), ("segments",)]
    assert context.representations_dir == context.validation_dir / "representations"
    assert context.recommendations_dir == context.validation_dir / "recommendations"
    assert context.diagnosis_path(["graph-activity", "base"]) == (
        context.validation_dir / "diagnosis" / "base_graph-activity" / "diagnosis.json"
    )
    for targets in ([], ["base", "base"], ["missing"]):
        with pytest.raises(ValueError):
            select_arms(context.config, targets)


def test_cli_mixed_arms_and_removed_options(context, monkeypatch):
    from pipeline_runtime import RunContext
    from validation.cli import STEP_HANDLERS, main

    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)
    calls = []
    monkeypatch.setitem(STEP_HANDLERS, "run-recommendation", lambda ctx, **kw: calls.append(kw))
    command = ["run-recommendation", "--run-id", context.run_id, "--dataset", context.dataset]
    assert main(command) == 0
    assert calls == [{"force": False}]
    with pytest.raises(SystemExit) as exc:
        main(["run-recommendation", "--run-id", context.run_id])
    assert exc.value.code == 2
    for option in (["--representation-mode", "graph"], ["--scene-aggregation", "mean"],
                   ["--step", "ablation"], ["--profile-every", "100"],
                   ["--profile-operators"], ["--target", "graph"],
                   ["--experiments", "another.yaml"]):
        with pytest.raises(SystemExit):
            main(command + option)



@pytest.mark.parametrize("configured, expected", [(False, "false"), (True, "true"), (None, "true")])
def test_recommendation_logs_effective_deterministic(context, monkeypatch, capsys, configured, expected):
    from validation import leave_one_out_recommendation
    from validation.steps import run_recommendation

    settings = context.config["validation"]["model"]
    if configured is None:
        settings.pop("deterministic", None)
    else:
        settings["deterministic"] = configured
    calls = []
    monkeypatch.setattr(
        leave_one_out_recommendation, "run_leave_one_out",
        lambda ctx, **kwargs: calls.append(kwargs),
    )
    run_recommendation(context, target=["base"], workers_per_gpu=4)
    output = capsys.readouterr().out
    assert f"  deterministic: {expected}\n" in output
    assert "profile_every" not in output
    assert "profile_operators" not in output
    assert calls == [{"force": False, "target": ["base"], "workers_per_gpu": 4}]
