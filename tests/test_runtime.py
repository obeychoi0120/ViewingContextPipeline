from pathlib import Path

import pytest

from arm_registry import generation_registry, registry, select_arms
from validation.graph_context import SCENE_AGGREGATIONS, graph_context, validate_mode
from validation.steps import validation_config


def test_current_config_loads(context):
    settings = validation_config(context)
    assert settings.encoder.embedding_dim == 1024
    assert settings.model.embedding_dim == 512
    assert settings.model.seeds == [42, 43, 44]
    assert settings.model.max_sequence_length == 10
    assert context.run_root == context.root / "artifacts" / "runs" / "regression"
    paths = [context.path("data", key) for key in context.config["data"]]
    paths += [context.path("models", key) for key in ("qwen3vl", "qwen3.5", "bge")]
    assert all(path.is_relative_to(context.root) for path in paths)
    assert not any(str(path).startswith("/home_nvme") for path in paths)
    assert Path(settings.dataset.pairs_csv).is_file()


def test_arm_sources_and_title_policy(context):
    arms = registry(context.config)
    expected = {
        "meta": (None, True),
        "graph_qwen3vl": ("graph_qwen3vl", False),
        "graph_qwen3vl_meta": ("graph_qwen3vl", True),
        "graph_qwen3.5": ("graph_qwen3.5", False),
        "graph_qwen3.5_meta": ("graph_qwen3.5", True),
        "desc_qwen3.5_meta": ("desc_qwen3.5", True),
        "graph_gemini_meta": ("graph_gemini", True),
        "desc_qwen3vl_meta": ("desc_qwen3vl", True),
        "desc_gemini_meta": ("desc_gemini", True),
        "graph_minicpm": ("graph_minicpm", False),
        "graph_minicpm_meta": ("graph_minicpm", True),
    }
    assert {name: (arm.scene_arm, arm.uses_title) for name, arm in arms.items()} == expected
    sources = generation_registry(context.config)
    assert all(not arm.uses_title and arm.name == arm.scene_arm for arm in sources.values())
    selected = select_arms(context.config, ["graph_qwen3vl_meta", "meta"])
    assert list(selected) == ["meta", "graph_qwen3vl_meta"]
    for targets in ([], ["meta", "meta"], ["missing"]):
        with pytest.raises(ValueError):
            select_arms(context.config, targets)


def test_representation_mode_constraints(context):
    validate_mode("text", None, "run-recommendation", ["meta"])
    validate_mode("graph", None, "embed-representations", ["graph_qwen3vl"])
    for aggregation in SCENE_AGGREGATIONS:
        validate_mode("graph", aggregation, "run-recommendation", ["graph_qwen3vl"])
        ctx = graph_context(context, aggregation)
        assert ctx.recommendations_dir.name == aggregation
    invalid = [
        ("text", "gnn", "run-recommendation", ["meta"]),
        ("graph", None, "run-recommendation", ["meta"]),
        ("graph", "mean", "run-recommendation", ["desc_qwen3vl_meta"]),
        ("graph", "attention", "run-diagnosis", ["meta"]),
        ("graph", "mean", "embed-representations", ["meta"]),
    ]
    for args in invalid:
        with pytest.raises(ValueError):
            validate_mode(*args)
