import json

import numpy as np
import pytest

from artifact_io import read_json, write_json
from arm_registry import registry
from helpers import prepare_inputs
from pipeline_runtime import RunContext
from validation.cli import main
from validation.graph_inputs import load_scene_segments, scene_segments_identity
from validation.representation_provenance import recommendation_identity


def test_prepare_module_cli_builds_segments_without_reencoding(context, monkeypatch, capsys):
    targets = ["graph-activity", "graph-role-activity"]
    prepare_inputs(context, monkeypatch, modules=False)
    directory = context.representations_dir / "graph_qwen3vl_2b"
    originals = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="python -m validation prepare-module"):
        load_scene_segments(directory)

    def forbidden(*args, **kwargs):
        raise AssertionError("module preparation attempted extraction or BGE encoding")

    monkeypatch.setattr("validation.features.BGETextEncoder", forbidden)
    monkeypatch.setattr("extraction.steps.qwen_generator", forbidden)
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)
    from validation.scene_segments import attach

    calls = []

    def tracked_attach(ctx, arm):
        calls.append(arm)
        return attach(ctx, arm)

    monkeypatch.setattr("validation.scene_segments.attach", tracked_attach)
    command = [
        "prepare-module", "--run-id", context.run_id, "--dataset", context.dataset,
    ]
    assert main(command) == 0
    assert calls == [targets[0]]
    output = capsys.readouterr().out
    assert "[STEP] prepare-module" in output
    assert f"selected_arms: {json.dumps(list(registry(context.config)))}" in output
    assert f"output_dir: {context.representations_dir}" in output
    manifest = read_json(directory / "manifest.json")
    pairs = load_scene_segments(directory)
    assert pairs.shape == (48, 2)
    np.testing.assert_array_equal(pairs[:, 0], np.arange(48))
    assert len(np.unique(pairs[:, 1])) == 24
    meta = scene_segments_identity(directory)
    assert meta["input_hash"] == manifest["input_hash"]
    for arm in ("graph", "graph-activity", "graph-role-activity"):
        identity = recommendation_identity(context, arm)
        assert json.loads(json.dumps(identity)) == identity
    assert recommendation_identity(context, "graph-activity")["scene_segments"] == recommendation_identity(context, "graph-role-activity")["scene_segments"]

    path = directory / "scene_segments.json"
    stale = read_json(path)
    stale["input_hash"] = "stale"
    write_json(path, stale)
    with pytest.raises(ValueError, match="python -m validation prepare-module"):
        load_scene_segments(directory)
    assert main(command + ["--force"]) == 0
    assert "force: true" in capsys.readouterr().out
    load_scene_segments(directory)
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_prepare_module_cli_with_no_activity_modules_is_noop(context, monkeypatch, capsys):
    from rec_experiments import CONFIG_KEY, parse_experiments

    context.config[CONFIG_KEY] = parse_experiments({"version": 1, "experiments": [
        {"label": "only-description", "source": "desc_qwen3vl_2b"},
    ]})

    def forbidden(*args, **kwargs):
        raise AssertionError("no activity module should be prepared")

    monkeypatch.setattr("validation.scene_segments.attach", forbidden)
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: context)
    assert main([
        "prepare-module", "--run-id", context.run_id, "--dataset", context.dataset,
    ]) == 0
    assert "[STEP] prepare-module" in capsys.readouterr().out
    assert not context.representations_dir.exists()
