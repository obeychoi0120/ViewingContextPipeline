from copy import deepcopy
from dataclasses import replace

import pytest

from artifact_io import read_json, write_json
from helpers import prepare_inputs
from rec_experiments import CONFIG_KEY
from validation.legacy_diagnosis import (
    LEGACY_ARMS,
    diagnosis_context,
    diagnosis_identity,
    stored_arm,
)
from validation.steps import embed_representations, prepare_module, run_diagnosis, run_recommendation


def legacy_context(context):
    config = deepcopy(context.config)
    for row, label in zip(config[CONFIG_KEY]["experiments"], LEGACY_ARMS, strict=True):
        row["label"] = label
    return replace(context, config=config)


def test_legacy_resolution_rejects_conflicts_and_missing_seeds(context):
    for seed in context.config["validation"]["model"]["seeds"]:
        (context.recommendations_dir / f"seed_{seed}" / "d").mkdir(parents=True)
    resolved, labels = diagnosis_context(context, ["description"])
    assert labels == ["d"]
    assert resolved.config[CONFIG_KEY]["experiments"][0]["label"] == "d"
    assert context.config[CONFIG_KEY]["experiments"][1]["label"] == "description"
    current = context.recommendations_dir / "seed_42" / "description"
    current.mkdir()
    with pytest.raises(ValueError, match="ambiguous diagnosis"):
        diagnosis_context(context, ["description"])
    current.rmdir()
    (context.recommendations_dir / "seed_44" / "d").rmdir()
    with pytest.raises(ValueError, match="missing legacy recommendation seed"):
        diagnosis_context(context, ["description"])


@pytest.mark.parametrize("mismatch", ["architecture_version", "experiment_arm", "partial_definition"])
def test_legacy_identity_rejects_other_structures(tmp_path, mismatch):
    identity = {
        "arm": "g2", "architecture_version": "sasrec-scene-content/v4",
        "experiment_arm": stored_arm(LEGACY_ARMS["g2"]),
        "experiment_definition": {},
    }
    training = deepcopy(identity)
    training.pop("experiment_definition")
    complete = {"identity": deepcopy(training)}
    if mismatch == "architecture_version":
        identity["architecture_version"] = "sasrec-scene-content/v3"
    elif mismatch == "experiment_arm":
        training["experiment_arm"]["aggregation"] = "gnn"
    else:
        complete["identity"]["experiment_definition"] = {}
    write_json(tmp_path / "training.json", training)
    write_json(tmp_path / "complete.json", complete)
    with pytest.raises(ValueError, match="unrecognized legacy experiment definition"):
        diagnosis_identity(tmp_path, identity)


@pytest.mark.torch
def test_legacy_results_roundtrip_without_retraining(context, monkeypatch):
    from validation.representation_provenance import recommendation_identity

    prepare_inputs(context, monkeypatch)
    old = legacy_context(context)
    embed_representations(old)
    prepare_module(old)

    # Generate the previous persisted format, including checkpoint and event identities.
    def old_identity(ctx, branch):
        value = recommendation_identity(ctx, branch)
        value.pop("experiment_definition")
        return value

    with monkeypatch.context() as training_patch:
        training_patch.setattr("validation.representation_provenance.recommendation_identity", old_identity)
        run_recommendation(old)
    original = {path: path.read_bytes() for path in context.recommendations_dir.rglob("*")
                if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnosis attempted training or preparation")

    monkeypatch.setattr("validation.leave_one_out_recommendation.run_leave_one_out", forbidden)
    monkeypatch.setattr("validation.graph_inputs.prepare", forbidden)
    monkeypatch.setattr("validation.features.BGETextEncoder", forbidden)
    run_diagnosis(context)
    document = read_json(old.diagnosis_path(list(LEGACY_ARMS)))
    assert document["runtime_decision"]["status"] == "pass"
    assert document["targets"] == list(LEGACY_ARMS)
    assert [row["label"] for row in document["experiments"]] == list(LEGACY_ARMS)
    assert document["statistics"]["status"] == "computed"
    assert len(document["statistics"]["comparisons"]) == 10
    assert document["recommendations"]["actual_event_count"] == 45
    assert all(row["definition_source"] == "validated_legacy_experiment_arm"
               for row in document["recommendations"]["model_reports"].values())
    report = (context.run_root / "reports" / "base_d_g0_g1_g2" / "report.md").read_text()
    assert "저장된 `experiment_arm`" in report
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert not (context.recommendations_dir / "seed_42" / "description").exists()

    # Compatibility cannot hide changed training inputs or corrupt result files.
    altered = replace(context, config=deepcopy(context.config))
    altered.config["validation"]["model"]["learning_rate"] *= 2
    with pytest.raises(RuntimeError, match="diagnosis failed"):
        run_diagnosis(altered)
    path = context.recommendations_dir / "seed_42" / "base" / "catalog_vectors.npy"
    path.write_bytes(original[path] + b"corrupt")
    with pytest.raises(RuntimeError, match="diagnosis failed"):
        run_diagnosis(context)
