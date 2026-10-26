from copy import deepcopy

import numpy as np
import pytest

from artifact_io import read_json, write_json, write_jsonl
from helpers import (
    prepare_inputs,
    refresh_checksums,
    result_identity,
    selected_table,
    train_inputs,
)
from validation.leave_one_out_recommendation import combination_complete
from validation.representation_checks import verify_representations
from validation.selection import training_signature
from validation.shared_cache import SharedCache
from validation.steps import validation_config


def test_training_identity_invalidates_changed_inputs(context, monkeypatch):
    from validation.recommendation_contracts import EVALUATION_CONTRACT

    prepare_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    config = validation_config(context)
    original = training_signature(context, cohort, config)
    modified = deepcopy(cohort)
    modified["events"][0]["timestamp"] += 1
    assert training_signature(context, modified, config) != original
    changed_config = config.model_copy(deep=True)
    changed_config.model.learning_rate *= 2
    assert training_signature(context, cohort, changed_config) != original
    monkeypatch.setitem(EVALUATION_CONTRACT, "history_mask", "different-contract")
    assert training_signature(context, cohort, config) != original


@pytest.mark.torch
def test_completion_requires_valid_bundle(context, monkeypatch):
    train_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    identity = result_identity(context, cohort)
    directory = context.recommendations_dir / "seed_42" / "base"
    assert combination_complete(directory, identity, 3)
    assert not combination_complete(directory, {**identity, "seed": 999}, 3)
    assert not combination_complete(directory, identity, 2)
    path = directory / "sasrec.pt"
    original = path.read_bytes()
    path.write_bytes(original + b"corruption")
    assert not combination_complete(directory, identity, 3)
    path.write_bytes(original)
    assert combination_complete(directory, identity, 3)
    path.unlink()
    assert not combination_complete(directory, identity, 3)
    path.write_bytes(original)
    # A semantically invalid metric row must also fail after checksums are repaired.
    metrics = directory / "per_event_metrics.jsonl"
    import json

    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[1]["event_id"] = rows[0]["event_id"]
    write_jsonl(metrics, rows)
    refresh_checksums(directory)
    assert not combination_complete(directory, identity, 3)
    complete = read_json(directory / "complete.json")
    complete["identity"]["evaluation_contract"] = {"version": "unsupported"}
    write_json(directory / "complete.json", complete)
    assert not combination_complete(directory, identity, 3)


def test_shared_cache_roundtrip_and_corruption(context):
    source = context.root / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"immutable features")
    cache = SharedCache(context, "regression", "key")
    cache.publish(source, ["data.bin"], origin={"run_id": "original"})
    assert cache.valid()
    target = context.root / "restored"
    manifest = cache.restore(target)
    assert manifest["origin"] == {"run_id": "original"}
    assert (target / "data.bin").read_bytes() == b"immutable features"
    (cache.path / "data.bin").write_bytes(b"corrupt")
    assert not cache.valid()
    rejected = context.root / "rejected"
    assert cache.restore(rejected) is None
    assert not rejected.exists()
    cache.publish(source, ["data.bin"], origin={"run_id": "repaired"})
    assert cache.valid()
    assert cache.restore(target)["origin"]["run_id"] == "repaired"


def test_unfinished_embeddings_are_rejected(context, monkeypatch):
    prepare_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    verify_representations(context, cohort, arms=["base"])
    root = context.representations_dir / "metadata"
    manifest = (root / "manifest.json").read_bytes()
    (root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing/stale/corrupt"):
        verify_representations(context, cohort, arms=["base"])

    (root / "manifest.json").write_bytes(manifest)
    path = root / "features.npy"
    values = np.load(path)
    values[1, 0] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match="missing/stale/corrupt"):
        verify_representations(context, cohort, arms=["base"])


@pytest.mark.torch
def test_cross_run_cache_and_paired_comparison(context, monkeypatch):
    from pipeline_runtime import RunContext
    from validation.run_comparison import compare_runs
    from validation.steps import run_recommendation, run_diagnosis

    train_inputs(context, monkeypatch)
    reference = RunContext.load("reference", root=context.root)
    prepare_inputs(reference, monkeypatch)
    reused = run_recommendation(reference, target=["base", "g0"])
    assert reused["completed"] == 0 and reused["skipped"] == 6
    cohort, _, _ = selected_table(reference)
    for seed in (42, 43, 44):
        for arm in ("base", "g0"):
            directory = reference.recommendations_dir / f"seed_{seed}" / arm
            assert combination_complete(directory, result_identity(reference, cohort, seed, arm), 3)
            assert read_json(directory / "training.json")["reused_from"]["source_run_id"] == context.run_id
    compared = compare_runs(context, reference.run_id, target=["g0", "base"])
    assert compared["targets"] == ["base", "g0"]
    assert all(row["difference"] == 0 for row in compared["comparisons"].values())
    assert run_diagnosis(context, target=["base", "g0"], compare_run_id=reference.run_id)["status"] == "pass"
    reference.config["validation"]["model"]["seeds"] = [42, 43, 45]
    import yaml

    # A changed seed contract must not be mixed with the previously trained grid.
    (reference.root / "config.yaml").write_text(yaml.safe_dump(reference.config))
    with pytest.raises(ValueError, match="identical model seeds"):
        compare_runs(context, reference.run_id, target=["base"])
