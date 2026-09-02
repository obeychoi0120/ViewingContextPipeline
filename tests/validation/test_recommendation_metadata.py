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
    refit_history = [
        {"epoch": 1, "loss": 1.1},
        {"epoch": 2, "loss": 0.9},
    ]
    record = _training_run_record(
        run_id="pilot",
        seed=42,
        arm="SASRec_METADATA",
        branch="metadata",
        selection_history=history,
        best_ndcg=0.2,
        best_epoch=2,
        refit_history=refit_history,
        checkpoint=checkpoint,
        candidate_count=1000,
        elapsed_seconds=3.5,
        max_epochs=50,
    )
    path = tmp_path / "training_runs.jsonl"

    _persist_training_runs(path, [record])

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == TRAINING_RUN_SCHEMA_VERSION
    assert stored["selection"]["best_validation"] == {
        "epoch": 2,
        "metric": "NDCG@10",
        "value": 0.2,
    }
    assert stored["selection"]["epochs"] == history
    assert stored["selection"]["epochs_completed"] == 2
    assert stored["selection"]["early_stopped"] is True
    assert stored["refit"] == {
        "data": "train+valid_target",
        "epochs_completed": 2,
        "epochs": refit_history,
    }
