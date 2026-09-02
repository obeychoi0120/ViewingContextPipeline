import numpy as np

from validation.recommendation_contracts import (
    ARCHITECTURE_VERSION,
    RECOMMENDATION_ARMS,
    TRAINING_RUN_SCHEMA_VERSION,
)
from validation.steps import _representations_match_catalog, _training_runs_complete
from pipeline_runtime import write_json, write_jsonl


def test_representation_cache_must_match_full_catalog(tmp_path) -> None:
    catalog = [{"item_id": "1"}, {"item_id": "2"}]
    item_index_path = tmp_path / "item_index.json"
    outputs = [
        tmp_path / f"{branch}.npz" for branch in ("metadata", "graph_qwen", "graph_gemini", "desc")
    ]
    write_json(item_index_path, {"1": 0, "2": 1})
    for path in outputs:
        np.savez_compressed(path, values=np.ones((2, 1024), dtype=np.float32))

    assert _representations_match_catalog(item_index_path, outputs, catalog, 1024)

    write_json(item_index_path, {"1": 0})
    assert not _representations_match_catalog(item_index_path, outputs, catalog, 1024)


def test_training_cache_requires_every_seed_and_arm_record(tmp_path) -> None:
    path = tmp_path / "training_runs.jsonl"
    seeds = [42, 43, 44]
    rows = [
        {
            "schema_version": TRAINING_RUN_SCHEMA_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "run_id": "pilot",
            "seed": seed,
            "arm": arm,
            "selection": {
                "best_validation": {"epoch": 1},
                "epochs": [{"epoch": 1}],
            },
            "refit": {
                "epochs_completed": 1,
                "epochs": [{"epoch": 1}],
            },
        }
        for seed in seeds
        for arm in RECOMMENDATION_ARMS
    ]

    write_jsonl(path, rows[:-1])
    assert not _training_runs_complete(path, run_id="pilot", seeds=seeds)

    write_jsonl(path, rows)
    assert _training_runs_complete(path, run_id="pilot", seeds=seeds)
    assert not _training_runs_complete(path, run_id="other", seeds=seeds)
