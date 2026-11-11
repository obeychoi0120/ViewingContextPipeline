from copy import deepcopy

import pytest
import torch

from arm_registry import Arm, registry
from rec_experiments import CONFIG_KEY, load_experiments, parse_experiments


def test_explicit_sources_defaults_modules_and_order():
    value = {"version": 1, "defaults": {"readout": "mean"}, "experiments": [
        {"label": "custom-role", "source": "graph_gemini", "modules": ["role_fusion"]},
        {"label": "combined", "source": "graph_qwen3vl_2b",
         "modules": ["activity_hypergraph", "role_fusion"], "readout": "mean"},
        {"label": "title-only", "source": "metadata"},
        {"label": "description-only", "source": "desc_gemini"},
    ]}
    resolved = parse_experiments(value)
    assert list(registry({CONFIG_KEY: resolved})) == [r["label"] for r in value["experiments"]]
    assert resolved["experiments"][1]["modules"] == ["role_fusion", "activity_hypergraph"]
    arms = registry({CONFIG_KEY: resolved})
    assert arms["custom-role"].role_fusion and arms["custom-role"].extensions == ()
    assert arms["title-only"].representation == "metadata"
    assert arms["description-only"].model == "gemini"
    assert all(arm.uses_title for arm in arms.values())


def test_config_module_names_resolve_to_result_identity_names():
    def modules(names):
        value = {"version": 1, "experiments": [
            {"label": "trial", "source": "graph_qwen3vl_2b", "modules": names}]}
        return parse_experiments(value)["experiments"][0]["modules"]

    assert modules(["ASHA", "REAF"]) == ["role_fusion", "activity_hypergraph"]
    assert modules(["REAF"]) == ["role_fusion"]
    # Earlier names stay readable; an alias and its target are the same module.
    assert modules(["activity_hypergraph"]) == ["activity_hypergraph"]
    for duplicate in (["ASHA", "activity_hypergraph"], ["REAF", "REAF"], ["asha"]):
        with pytest.raises(ValueError):
            modules(duplicate)


def test_invalid_settings_are_rejected_before_outputs(context, monkeypatch):
    value = {"version": 1, "experiments": [{"label": "trial", "source": "graph_qwen3vl_2b"}]}
    invalid = [None, {}, {**value, "version": True}, {**value, "experiments": []},
               {**value, "defaults": {"source": "graph_qwen3vl_2b"}},
               {**value, "defaults": {"readout": "attention"}},
               {**value, "use_title": True}]
    for change in ({"source": None}, {"label": "../bad"}, {"label": "A"},
                   {"label": "has_space"}, {"readout": "sum"}, {"source": "missing"},
                   {"modules": "role_fusion"}, {"modules": [None]}, {"modules": ["unknown"]},
                   {"modules": ["role_fusion", "role_fusion"]}, {"use_title": True},
                   {"source": "metadata", "modules": ["role_fusion"]},
                   {"source": "desc_qwen3vl_2b", "modules": ["activity_hypergraph"]}):
        invalid.append({**value, "experiments": [{**value["experiments"][0], **change}]})
    invalid += [{**value, "experiments": value["experiments"] * 2},
                {**value, "experiments": [{"label": n, "source": "metadata"} for n in ["a", "b"]]}]
    for item in invalid:
        with pytest.raises(ValueError):
            parse_experiments(item)
    # Invalid configuration fails before feature preparation or directory creation.
    context.config.pop(CONFIG_KEY)
    (context.root / "rec_experiments_config.yaml").write_text("version: 1\nexperiments: []\n")
    from validation.steps import embed_representations

    monkeypatch.setattr("validation.graph_inputs.prepare", lambda *a, **k: pytest.fail("prepared invalid config"))
    with pytest.raises(ValueError, match="rec_experiments_config.yaml"):
        embed_representations(context)
    assert not context.run_root.exists()


def test_snapshot_survives_file_changes_and_extraction_does_not_require_file(context):
    import pickle
    from pipeline_runtime import RunContext

    snapshot = deepcopy(load_experiments(context))
    (context.root / "rec_experiments_config.yaml").unlink()
    assert load_experiments(pickle.loads(pickle.dumps(context))) == snapshot
    fresh = RunContext.load(context.run_id, dataset=context.dataset, root=context.root)
    assert fresh.scene_arm_dir("graph_gemini").name == "graph_gemini"
    assert CONFIG_KEY not in fresh.config
    with pytest.raises(ValueError, match="rec_experiments_config.yaml"):
        load_experiments(fresh)


def test_initial_definitions_preserve_legacy_weights_and_outputs(context, monkeypatch):
    from helpers import prepare_inputs
    from validation.graph_model import new_graph_model
    from validation.steps import embed_representations, prepare_module, validation_config

    prepare_inputs(context, monkeypatch)
    embed_representations(context)
    prepare_module(context)
    config = validation_config(context)
    legacy = [
        Arm("base", "metadata", uses_title=True),
        Arm("d", "description", "qwen3vl_2b", True, "desc_qwen3vl_2b", "mean"),
        Arm("g0", "graph", "qwen3vl_2b", True, "graph_qwen3vl_2b", "mean"),
        Arm("g1", "graph", "qwen3vl_2b", True, "graph_qwen3vl_2b", "mean", extensions=("segments",)),
        Arm("g2", "graph", "qwen3vl_2b", True, "graph_qwen3vl_2b", "mean", True, ("segments",)),
    ]
    for current, previous in zip(registry(context.config), legacy, strict=True):
        torch.manual_seed(42)
        actual = new_graph_model(context, config, current, "cpu").eval()
        with monkeypatch.context() as patch:
            patch.setattr("arm_registry.registry", lambda config: {previous.name: previous})
            torch.manual_seed(42)
            expected = new_graph_model(context, config, previous.name, "cpu").eval()
        assert actual.state_dict().keys() == expected.state_dict().keys()
        assert all(torch.equal(t, expected.state_dict()[k]) for k, t in actual.state_dict().items())
        with torch.no_grad():
            assert torch.equal(actual.catalog_vectors(), expected.catalog_vectors())


def replace_experiments(context, rows):
    import yaml

    path = context.root / "rec_experiments_config.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "experiments": rows}))
    context.config.pop(CONFIG_KEY, None)
    load_experiments(context)


@pytest.mark.parametrize("rows, baseline", [
    ([{"label": "just-description", "source": "desc_qwen3vl_2b"}], None),
    ([{"label": "title-only", "source": "metadata"},
      {"label": "base", "source": "graph_qwen3vl_2b", "modules": ["role_fusion"]}], "title-only"),
])
def test_custom_labels_and_sources_complete_the_pipeline(context, monkeypatch, rows, baseline):
    from artifact_io import read_json
    from helpers import prepare_inputs
    from validation.steps import embed_representations, prepare_module, run_recommendation, run_diagnosis
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config

    prepare_inputs(context, monkeypatch)
    replace_experiments(context, rows)
    embed_representations(context)
    assert prepare_module(context)["statistics"] == {}
    result = run_recommendation(context)
    assert result["completed"] == 3 * len(rows)
    run_diagnosis(context)
    document = read_json(context.diagnosis_path(None))
    assert document["recommendations"]["baseline"] == baseline
    assert (document["recommendations"]["base_relative_changes"] is None) == (baseline is None)
    assert document["statistics"]["comparison_count"] == len(rows) * (len(rows) - 1) // 2
    assert document["targets"] == [r["label"] for r in rows]
    if baseline:
        model = new_graph_model(context, validation_config(context), "base", "cpu")
        assert model.graph_encoder.role_fusion is not None
        assert not hasattr(model, "item_projection")


def test_editing_experiment_list_reuses_results_and_preserves_conflicts(context, monkeypatch):
    from artifact_io import read_json
    from helpers import train_inputs
    from validation.steps import embed_representations, prepare_module, run_recommendation

    train_inputs(context, monkeypatch)
    saved = {p: p.read_bytes() for p in context.recommendations_dir.glob("seed_*/*/*") if p.is_file()}
    rows = [r for r in load_experiments(context)["experiments"] if r["label"] in {"base", "graph-activity"}]
    added = {"label": "role-only", "source": "graph_qwen3vl_2b", "modules": ["role_fusion"]}
    replace_experiments(context, rows + [added])
    assert embed_representations(context)["generated_sources"] == []
    prepare_module(context)
    result = run_recommendation(context)
    assert result["skipped"] == 6 and result["completed"] == 3
    assert all(p.read_bytes() == value for p, value in saved.items())
    original = read_json(context.recommendations_dir / "seed_42" / "graph-activity" / "identity.json")
    replace_experiments(context, [rows[1]])
    assert run_recommendation(context)["skipped"] == 3
    assert all(p.read_bytes() == value for p, value in saved.items())
    # Changed definitions cannot overwrite a label's previous architecture, including with force.
    replace_experiments(context, [{**rows[1], "modules": []}])
    with pytest.raises(ValueError, match="new Run ID"):
        run_recommendation(context, force=True)
    assert all(p.read_bytes() == value for p, value in saved.items())
    assert original["experiment_definition"]["modules"] == ["activity_hypergraph"]


def test_run_comparison_rejects_same_label_with_different_definition(context, monkeypatch):
    from dataclasses import replace
    from artifact_io import write_json
    from rec_experiments import experiment_definition
    from validation.run_comparison import compare_runs

    reference = replace(context, run_id="reference", run_root=context.run_root.parent / "reference")
    definitions = {name: experiment_definition(arm) for name, arm in registry(context.config).items()}
    for ctx in (context, reference):
        for seed in ctx.config["validation"]["model"]["seeds"]:
            for name, definition in definitions.items():
                row = deepcopy(definition)
                if ctx is reference and name == "graph-role-activity":
                    row["modules"] = ["activity_hypergraph"]
                write_json(ctx.recommendations_dir / f"seed_{seed}" / name / "training.json",
                           {"experiment_definition": row})
    monkeypatch.setattr("validation.run_comparison.RunContext.load", lambda *a, **kw: reference)
    with pytest.raises(ValueError, match="identical experiment definitions: graph-role-activity"):
        compare_runs(context, reference.run_id)
