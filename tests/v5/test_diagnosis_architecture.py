import numpy as np
import pytest

from validation.selection import prepare_validation_cohort, training_signature
from pipeline_runtime import read_json, write_json, write_jsonl
from validation.metrics import metrics_from_rank
from validation.representation_provenance import state_path
from validation.rolling_data import EventTable, iter_jsonl
from validation.rolling_diagnosis import collect_metrics, diagnose, diagnosis_training
from validation.rolling_recommendation import LEGACY_SCHEMA as SCHEMA, combination_complete, combination_dir, phase_ids
from validation.steps import validation_config


@pytest.fixture
def historical_results(ready_context, monkeypatch):
    context = ready_context
    config = validation_config(context)
    context.config["protocol"]["arms"] = ["metadata"]
    cohort = prepare_validation_cohort(context, "qwen")
    table = EventTable(cohort["events"])
    signature = training_signature(context, cohort, config)
    write_json(state_path(context, "metadata"), {"recommendation_hash": "embedding"})
    monkeypatch.setattr("validation.rolling_diagnosis.verify_representations", lambda *a, **k: None)
    monkeypatch.setattr("validation.metadata.verify_missing_metadata", lambda *a: {})
    directories = []
    for split in cohort["plan"]["splits"]:
        ids = phase_ids(table, split, "test")
        raw_refit = table.select(end=split["phases"]["refit"]["end_ms"], eligible=False)
        frequencies = np.bincount(table.targets[raw_refit], minlength=len(table.items) + 1)
        for seed in config.model.seeds:
            identity = {
                "run_id": context.run_id, "evaluation_date": split["evaluation_date"],
                "seed": seed, "arm": "metadata", "training_input_hash": signature,
                "embedding_hash": "embedding",
            }
            directory = combination_dir(context, split["evaluation_date"], seed, "metadata")
            directory.mkdir(parents=True)
            directories.append(directory)
            training = {
                **identity, "schema_version": SCHEMA, "architecture_version": "sasrec-content-v2",
                "catalog_size": len(table.items), "best_epoch": 1, "split": split,
            }
            for phase in ("selection", "refit"):
                count = split["phases"][phase]["eligible_count"]
                training[phase] = [{"positive_count": count,
                                    "optimizer_updates": (count + config.model.batch_size - 1)
                                    // config.model.batch_size}]
            write_json(directory / "training.json", training)
            write_json(directory / "complete.json", {
                "schema_version": SCHEMA, "identity": identity, "event_count": len(ids),
            })
            # Diagnosis only checks checkpoint presence; it never loads the model.
            (directory / "sasrec.pt").write_bytes(b"fixture checkpoint")
            write_jsonl(directory / "per_event_metrics.jsonl", [{
                **table.rows[int(event)], **identity, "schema_version": "sasrec-per-event-metrics/v1",
                "rank": 1, "candidate_count": len(table.items),
                "refit_item_frequency": int(frequencies[table.targets[event]]),
                **metrics_from_rank(1, config.evaluation.cutoffs),
            } for event in ids])
    return context, config, cohort, directories


@pytest.mark.parametrize("version", ["sasrec-content-v2", "sasrec-content-v3"])
def test_historical_diagnosis_and_strict_current_resume(historical_results, version):
    context, _, _, directories = historical_results
    for directory in directories:
        path = directory / "training.json"
        training = read_json(path)
        training["architecture_version"] = version
        write_json(path, training)
    original = [(directory / "training.json").read_bytes() for directory in directories]
    assert diagnose(context, target=["metadata"])["status"] == "pass"
    report = read_json(context.diagnosis_path)
    assert report["recommendations"]["architecture_version"] == version
    assert report["recommendations"]["combination_count"] == 21
    assert report["statistics"]["status"] == "computed"
    assert report["recommendations"]["means"]["metadata"]["NDCG@10"] == 1
    for directory, before in zip(directories, original, strict=True):
        complete = read_json(directory / "complete.json")
        assert combination_complete(directory, complete["identity"], complete["event_count"]) is False
        assert (directory / "training.json").read_bytes() == before


@pytest.mark.parametrize("version,message", [
    ("sasrec-content-v3", "mixed recommendation architectures"),
    ("sasrec-content-v999", "unsupported diagnosis architecture"),
    (None, "unsupported diagnosis architecture"),
])
def test_mixed_unknown_or_missing_versions_fail(historical_results, version, message):
    context, config, cohort, directories = historical_results
    path = directories[-1] / "training.json"
    training = read_json(path)
    training["architecture_version"] = version
    write_json(path, training)
    with pytest.raises(ValueError, match=message):
        collect_metrics(context, config, cohort, arms={"metadata": "metadata"})


@pytest.mark.parametrize("corruption", ["embedding", "rank", "missing_checkpoint", "metric"])
def test_historical_results_still_require_valid_evidence(historical_results, corruption):
    context, config, cohort, directories = historical_results
    directory = directories[0]
    complete = read_json(directory / "complete.json")
    if corruption == "embedding":
        training = read_json(directory / "training.json")
        training["embedding_hash"] = "changed"
        write_json(directory / "training.json", training)
    elif corruption == "missing_checkpoint":
        (directory / "sasrec.pt").unlink()
    else:
        rows = list(iter_jsonl(directory / "per_event_metrics.jsonl"))
        rows[0]["rank" if corruption == "rank" else "NDCG@10"] = 0
        write_jsonl(directory / "per_event_metrics.jsonl", rows)
    if corruption == "metric":
        with pytest.raises(ValueError, match="event metric/rank mismatch"):
            collect_metrics(context, config, cohort, arms={"metadata": "metadata"})
    else:
        with pytest.raises(ValueError, match="incomplete/corrupt combination"):
            diagnosis_training(directory, complete["identity"], complete["event_count"])
