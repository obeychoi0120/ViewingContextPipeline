from __future__ import annotations

import json

from validation.recommendation import (
    TRAINING_RUN_SCHEMA_VERSION,
    _persist_training_runs,
    _training_run_record,
)


def test_training_run_metadata_is_persisted_with_epoch_history(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "sasrec.pt"
    history = [
        {"epoch": 1, "loss": 1.25, "NDCG@10": 0.1},
        {"epoch": 2, "loss": 1.0, "NDCG@10": 0.2},
    ]
    record = _training_run_record(
        run_id="pilot",
        seed=42,
        arm="SASRec_ID",
        branch=None,
        history=history,
        best_ndcg=0.2,
        best_epoch=2,
        checkpoint=checkpoint,
        candidate_count=1000,
        elapsed_seconds=3.5,
        max_epochs=50,
    )
    path = tmp_path / "training_runs.jsonl"

    _persist_training_runs(path, [record])

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == TRAINING_RUN_SCHEMA_VERSION
    assert stored["best_validation"] == {
        "epoch": 2,
        "metric": "NDCG@10",
        "value": 0.2,
    }
    assert stored["epochs"] == history
    assert stored["epochs_completed"] == 2
    assert stored["early_stopped"] is True
