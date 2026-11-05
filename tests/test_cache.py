from copy import deepcopy

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


def test_preflight_protects_entire_grid_before_restoring_or_training(context, monkeypatch):
    from validation.steps import run_recommendation
    from validation import recommendation_cache

    prepare_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    last = context.recommendations_dir / "seed_44" / "graph-activity"
    last.mkdir(parents=True)
    old = {**result_identity(context, cohort, 44, "graph-activity"), "architecture_version": "sasrec-scene-content/v1"}
    write_json(last / "training.json", old)
    before = {p: p.read_bytes() for p in context.recommendations_dir.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("must preflight every selected result before writing any combination")

    monkeypatch.setattr(recommendation_cache, "restore", forbidden)
    monkeypatch.setattr("validation.leave_one_out_recommendation.run_combination", forbidden)
    with pytest.raises(ValueError, match="new Run ID"):
        run_recommendation(context, target=["base", "graph-activity"], force=True)
    assert before == {p: p.read_bytes() for p in context.recommendations_dir.rglob("*") if p.is_file()}
