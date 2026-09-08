from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline_runtime import RunContext, read_json, write_json
from validation.rolling_data import DAY, EventTable, load_csv, prepare_full_cohort
from validation.rolling_diagnosis import cluster_bootstrap, collect_metrics, weighted_day_mean
from validation.rolling_recommendation import run_rolling
from validation.steps import validation_config

ROOT = Path(__file__).resolve().parents[2]


def event_table(records):
    return EventTable(
        [
            dict(event_id=i, user_id=str(u), item_id=str(item), timestamp=t)
            for i, (u, item, t) in enumerate(records)
        ]
    )


def test_all_transitions_strict_ties_and_full_history():
    table = event_table([(1, i + 1, i) for i in range(30)] + [(1, 31, 29), (2, 1, 29)])
    assert table.history(30) == list(range(1, 30))
    assert table.history(29, 10) == list(range(20, 30))
    assert table.history(31) == []
    assert len(table.select()) == 30
    assert len(table.select(end=15)) == 14
    # An earlier event appearing later in the source file is still in context.
    table = event_table([(1, 3, 3), (1, 1, 1), (1, 2, 2), (1, 4, 3)])
    assert table.history(0) == [1, 2] == table.history(3)


def test_csv_retains_duplicates_and_rejects_fractional_timestamps(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("user,item,timestamp\n1,2,100\n1,2,100\n1,3,101\n")
    table, duplicates = load_csv(path)
    assert duplicates == 1 and len(table.rows) == 3
    assert table.history(2) == [1, 1]
    path.write_text("user,item,timestamp\n1,2,100.0\n")
    with pytest.raises(ValueError, match="integer milliseconds"):
        load_csv(path)
    path.write_text("user,item,timestamp\n1,2,100\n\n1,3,101\n")
    with pytest.raises(ValueError, match="malformed CSV"):
        load_csv(path)


def test_utc_rolling_boundaries_and_same_day_context():
    origin = int(datetime(2022, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    table = event_table([(1, i % 4 + 1, origin + i * DAY // 2) for i in range(24)])
    splits = table.splits()
    assert [s["evaluation_date"] for s in splits] == [f"2022-09-{i:02}" for i in range(5, 12)]
    first = splits[0]["phases"]
    assert first["selection"]["end_ms"] == origin + 3 * DAY
    assert first["test"]["raw_count"] == 2
    assert table.history(9)[-1] == int(table.targets[8])
    assert all(
        table.timestamps[i] < first["refit"]["end_ms"]
        for i in table.select(end=first["refit"]["end_ms"])
    )


def test_equal_date_mean_and_paired_user_multiplicity():
    counts = np.array([[1, 3], [1, 1]], dtype=float)
    sums = np.array([[[1, 0.5], [0, 0]], [[0, 0], [1, 0.5]]])
    np.testing.assert_allclose(weighted_day_mean(sums, counts, np.ones((1, 2))), [[0.375, 0.1875]])
    observed, draws, report = cluster_bootstrap(sums, counts, samples=200, seed=4)
    np.testing.assert_allclose(observed, [0.375, 0.1875])
    np.testing.assert_allclose(draws[:, 1], draws[:, 0] / 2)
    # Draws select both events of a user as a cluster, across both dates.
    assert set(draws[:, 0]) <= {0.375, 0.5}
    assert report["working_bytes_upper_bound"] <= 128 * 1024**2
    with pytest.raises(ValueError, match="empty evaluation date"):
        weighted_day_mean(sums, np.zeros_like(counts), np.ones((1, 2)))


def test_100k_bootstrap_arrays_are_bounded():
    counts = np.ones((100_000, 7))
    sums = np.full((100_000, 7, 4), 0.2)
    observed, draws, report = cluster_bootstrap(sums, counts, samples=70)
    np.testing.assert_allclose(observed, 0.2)
    np.testing.assert_allclose(draws, 0.2)
    assert report["working_bytes_upper_bound"] <= report["memory_limit_bytes"]


@pytest.fixture
def full_context(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    config["artifacts_root"] = str(tmp_path / "artifacts")
    for key in config["data"]:
        config["data"][key] = str(tmp_path / key)
    videos = Path(config["data"]["videos_dir"])
    videos.mkdir()
    for item in range(1, 5):
        (videos / f"{item}.mp4").write_bytes(b"fixture video")
    Path(config["data"]["titles_csv"]).write_text("".join(f"{i},title {i}\n" for i in range(1, 5)))
    origin = int(datetime(2022, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [(user, i % 4 + 1, origin + i * DAY // 2) for user in range(1, 4) for i in range(24)]
    Path(config["data"]["pairs_csv"]).write_text(
        "user,item,timestamp\n" + "".join(f"{u},{i},{t}\n" for u, i, t in rows)
    )
    config["validation"]["cohort"].update(user_count=3, interaction_count=72, item_count=4)
    config["validation"]["model"].update(max_epochs=1, patience=1)
    config["validation"]["evaluation"]["bootstrap_samples"] = 20
    for branch in ("graph", "description"):
        for kind in ("scene_prompt", "summary_prompt"):
            config["extraction"][branch][kind] = str(ROOT / config["extraction"][branch][kind])
    for model in ("qwen", "bge"):
        config["models"][model] = str(tmp_path / model)
        Path(config["models"][model]).mkdir()
        (Path(config["models"][model]) / "config.json").write_text("{}")
    context = RunContext(tmp_path, "full", config, tmp_path / "artifacts" / "full")
    from validation.cohort import build_item_inventory

    monkeypatch.setattr(
        "validation.rolling_data.build_item_inventory",
        lambda items, path: build_item_inventory(items, path, probe=lambda _: 31.0),
    )
    context.initialize()
    prepare_full_cohort(context)
    return context


def test_full_preparation_and_portable_fingerprints(full_context):
    context = full_context
    cohort = context.require_ready_cohort()
    assert cohort["plan"]["interaction_count"] == 72
    assert cohort["plan"]["eligible_test_count"] == 42
    preflight = read_json(context.cohort_dir / "media_preflight.json")
    assert preflight["scene_count"] == 8 and preflight["keyframe_count"] == 28
    before = read_json(context.run_root / "experiment.json")["fingerprint"]
    context.config["data"]["videos_dir"] = "D:/host-specific/videos"
    context.initialize()
    assert read_json(context.run_root / "experiment.json")["fingerprint"] == before
    context.config["extraction"]["visual_evidence"]["num_keyframes"] = 3
    with pytest.raises(RuntimeError, match="new run ID"):
        context.initialize()


def test_cardinality_and_tampered_source_fail(full_context):
    context = full_context
    path = context.cohort_dir / "events.jsonl"
    with path.open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="changed"):
        context.require_ready_cohort()
    context.config["validation"]["cohort"]["interaction_count"] = 719405
    with pytest.raises(ValueError, match="cardinality"):
        prepare_full_cohort(context, plan_only=True)


def test_gemini_fallback_and_invalid_present_summary(full_context, monkeypatch):
    from extraction.summary_validation import SUMMARY_SECTIONS, serialize_summary_sections
    from validation.steps import embed_representations
    from validation.provenance import verify_representations

    context = full_context
    cohort = context.require_ready_cohort()
    sections = {field: "visible evidence" for field in SUMMARY_SECTIONS}
    for row in cohort["catalog"]:
        for arm, directory, schema in (
            ("graph_qwen", context.graph_summary_dir("qwen"), "graph-video-summary/v3"),
            ("description", context.description_summary_dir, "description-video-summary/v3"),
        ):
            write_json(
                directory / f"{row['content_id']}.json",
                {
                    "schema_version": schema,
                    "arm": arm,
                    "content_id": row["content_id"],
                    "status": "complete",
                    "sections": sections,
                    "text": serialize_summary_sections(sections),
                    "scene_count": 2,
                },
            )

    class Encoder:
        def __init__(self, config):
            pass

        def encode(self, texts):
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr("validation.features.BGETextEncoder", Encoder)
    from validation.provenance import bind_stage

    for stage in ("summarize-graph-qwen", "summarize-description"):
        bind_stage(context, stage, {"fixture": "structured summary"})
    embed_representations(context)
    fallbacks = read_json(context.representations_dir / "graph_gemini_fallbacks.json")["fallbacks"]
    assert len(fallbacks) == 4
    verify_representations(context)
    # A missing branch must not cause another corrupt, finite-shaped matrix to be re-signed.
    np.savez(context.representations_dir / "metadata_embeddings.npz", values=np.zeros((4, 1024)))
    (context.representations_dir / "graph_qwen_embeddings.npz").unlink()
    embed_representations(context)
    with np.load(context.representations_dir / "metadata_embeddings.npz") as data:
        assert np.all(data["values"] == 1)
    verify_representations(context)
    # Cached representations still validate a present malformed Gemini summary.
    content = cohort["catalog"][0]["content_id"]
    write_json(
        context.graph_summary_dir("gemini") / f"{content}.json",
        {"content_id": content, "text": "invalid seven-section document", "scene_count": 2},
    )
    with pytest.raises(RuntimeError, match="incompatible structured summary"):
        embed_representations(context)


def test_zero_relative_denominator_cannot_produce_a_decision():
    from types import SimpleNamespace
    from validation.rolling_diagnosis import comparisons

    settings = SimpleNamespace(familywise_alpha=0.05, non_inferiority_margin=0.05)
    with pytest.raises(ValueError, match="denominator"):
        comparisons(np.zeros(4), np.zeros((20, 4)), settings)


def test_invalid_gemini_summary_is_not_automatically_replaced(full_context, monkeypatch):
    from pipeline_runtime import write_jsonl
    from validation.provenance import bind_extraction
    from extraction.steps import summarize_graph

    context = full_context
    catalog = context.require_ready_cohort()["catalog"]
    for row in catalog:
        write_jsonl(context.graph_scene_dir("gemini") / f"{row['content_id']}.jsonl", [])
    bind_extraction(context, "summarize-graph-gemini", scene_dir=context.graph_scene_dir("gemini"))
    path = context.graph_summary_dir("gemini") / f"{catalog[0]['content_id']}.json"
    write_json(path, {"text": "malformed existing Gemini summary"})
    before = path.read_bytes()
    monkeypatch.setattr("extraction.steps._visual_rows", lambda _: catalog)
    with pytest.raises(RuntimeError, match="incompatible structured summary"):
        summarize_graph(context, source="gemini")
    assert path.read_bytes() == before


@pytest.mark.torch
def test_evaluation_masks_history_older_than_the_ten_item_context():
    import torch
    from types import SimpleNamespace
    from validation.rolling_recommendation import evaluate

    table = event_table([(1, i + 1, i) for i in range(31)])

    class FixedScores(torch.nn.Module):
        max_length = 10

        def catalog_vectors(self):
            return torch.arange(31, 0, -1, dtype=torch.float32).reshape(-1, 1)

        def user_vectors(self, sequences):
            assert sequences.shape[1] == 10
            return torch.ones((len(sequences), 1))

    config = SimpleNamespace(
        model=SimpleNamespace(batch_size=2), evaluation=SimpleNamespace(cutoffs=[10, 30])
    )
    row = next(evaluate(FixedScores(), table, np.array([30]), config, torch.device("cpu")))
    assert row["rank"] == 1 and row["candidate_count"] == 31


@pytest.mark.torch
def test_84_combinations_real_cpu_training_resume_and_diagnosis(full_context, monkeypatch):
    import torch
    from validation.model import SASRec
    from validation.recommendation_contracts import RECOMMENDATION_ARMS

    torch.set_num_threads(1)
    context = full_context
    directory = context.representations_dir
    directory.mkdir(parents=True)
    write_json(directory / "item_index.json", {str(i): i - 1 for i in range(1, 5)})
    write_json(directory / "graph_gemini_fallbacks.json", {"fallbacks": []})
    for branch in RECOMMENDATION_ARMS.values():
        np.savez(
            directory / f"{branch}_embeddings.npz", values=np.ones((4, 1024), dtype=np.float32)
        )
    from validation.provenance import bind_stage, complete_representations

    bind_stage(context, "representations", {"fixture": "synthetic fixed features"})
    complete_representations(context)

    def tiny_model(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm="metadata", item_features=features).to(device)

    monkeypatch.setattr("validation.rolling_recommendation._new_model", tiny_model)
    from validation.rolling_recommendation import evaluate as original_evaluate

    calls = []

    def interrupted_evaluate(*args, **kwargs):
        calls.append(1)
        for index, row in enumerate(original_evaluate(*args, **kwargs)):
            if len(calls) == 2 and index == 1:
                raise KeyboardInterrupt
            yield row

    monkeypatch.setattr("validation.rolling_recommendation.evaluate", interrupted_evaluate)
    with pytest.raises(KeyboardInterrupt):
        run_rolling(context)
    assert not list(context.recommendations_dir.rglob("complete.json"))
    assert len(list(context.recommendations_dir.rglob("per_event_metrics.jsonl.tmp"))) == 1
    monkeypatch.setattr("validation.rolling_recommendation.evaluate", original_evaluate)
    result = run_rolling(context)
    assert result == {"stage": "run-recommendation", "completed": 84, "skipped": 0}
    completions = list(context.recommendations_dir.rglob("complete.json"))
    assert len(completions) == 84
    protected = completions[-1].read_bytes()
    assert run_rolling(context)["skipped"] == 84
    # A corrupt combination restarts; other completed combinations stay byte-identical.
    completions[0].with_name("per_event_metrics.jsonl").write_text("{}\n")
    result = run_rolling(context)
    assert result["completed"] == 1 and result["skipped"] == 83
    assert completions[-1].read_bytes() == protected
    config = validation_config(context)
    sums, counts, report = collect_metrics(context, config, context.require_ready_cohort())
    assert report["actual_event_count"] == 42 * 12
    assert report["combination_count"] == 84
    observed, _, _ = cluster_bootstrap(sums, counts, samples=20)
    np.testing.assert_allclose(
        observed, [report["means"][a]["NDCG@10"] for a in RECOMMENDATION_ARMS]
    )
    # Exercise the public diagnosis stage, including original scene coverage.
    import json
    from pipeline_runtime import write_jsonl
    from validation.rolling_diagnosis import diagnose

    for item in context.require_ready_cohort()["catalog"]:
        content = item["content_id"]
        timestamp = (
            context.cohort_dir / "source_assets" / content / "assets" / "timestamp_fixed_30s.json"
        )
        timestamp.parent.mkdir(parents=True)
        timestamp.write_text(json.dumps([{"scene_idx": 0}, {"scene_idx": 1}]))
        for source in ("qwen", "gemini"):
            write_jsonl(
                context.graph_scene_dir(source) / f"{content}.jsonl",
                [
                    dict(
                        scene_idx=i,
                        keyframes=[2.5 + 30 * i],
                        graph={},
                        parse_mode="native",
                        semantic_warnings=[],
                    )
                    for i in range(2)
                ],
            )
        write_jsonl(
            context.description_scene_dir / f"{content}.jsonl",
            [
                dict(
                    schema_version="scene-description/v1",
                    content_id=content,
                    scene_idx=i,
                    keyframes=[2.5 + 30 * i],
                    description="fixture",
                )
                for i in range(2)
            ],
        )
    assert diagnose(context)["status"] == "pass"
    diagnosis = read_json(context.diagnosis_path)
    assert diagnosis["statistics"]["status"] == "computed"
    assert diagnosis["scene_coverage"]["arms"]["graph_gemini"]["success_coverage"] == 1
    # Even if a writer updates the checksum, repeated events cannot pass validation.
    from validation.provenance import file_hash

    path = completions[0].with_name("per_event_metrics.jsonl")
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    document = read_json(completions[0])
    document["hashes"][path.name] = file_hash(path)
    write_json(completions[0], document)
    with pytest.raises(ValueError, match="duplicate"):
        collect_metrics(context, config, context.require_ready_cohort())


@pytest.mark.torch
def test_evaluation_keeps_parameters_and_caches_catalog():
    import torch
    from types import SimpleNamespace
    from validation.model import SASRec
    from validation.rolling_recommendation import evaluate

    table = event_table([(1, i % 4 + 1, i) for i in range(15)])
    model = SASRec(4, 10, 8, 1, 2, 0, arm="metadata", item_features=np.ones((4, 8)))
    before = deepcopy(model.state_dict())
    calls = []
    original = model.catalog_vectors

    def catalog():
        calls.append(1)
        return original()

    model.catalog_vectors = catalog
    config = SimpleNamespace(
        model=SimpleNamespace(batch_size=2), evaluation=SimpleNamespace(cutoffs=[10, 30])
    )
    rows = list(evaluate(model, table, table.select(), config, torch.device("cpu")))
    assert len(calls) == 1 and len(rows) == 14
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
    assert all(r["rank"] == 1 for r in rows[3:])
