from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml
from PIL import Image

import extraction.cli as extraction_cli
import extraction.data_preparation.fixed30 as fixed30
import extraction.steps as extraction_steps
import validation.cli as validation_cli
import validation.cohort as cohort_module
import validation.features as features
import validation.recommendation as recommendation
import validation.steps as validation_steps
from extraction.evidence_reuse import copy_matching_evidence, evidence_paths
from extraction.preparation import prepare_input_data
from extraction.summary_validation import SUMMARY_SECTIONS
from pipeline_runtime import RunContext, read_json, read_jsonl, write_json, write_jsonl
from validation.metrics import metrics_from_rank
from validation.recommendation_contracts import RECOMMENDATION_ARMS


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    config["artifacts_root"] = str(tmp_path / "artifacts")
    config["data"] = {
        "pairs_tsv": str(tmp_path / "pairs.tsv"),
        "titles_csv": str(tmp_path / "titles.csv"),
        "videos_dir": str(tmp_path / "videos"),
    }
    Path(config["data"]["videos_dir"]).mkdir()
    Path(config["data"]["pairs_tsv"]).write_text(
        "".join(
            f"u{user}\t" + " ".join(str(user + step) for step in range(1, 7)) + "\n"
            for user in range(3)
        ),
        encoding="utf-8",
    )
    Path(config["data"]["titles_csv"]).write_text(
        "".join(f"{item},Title {item}\n" for item in range(1, 9)), encoding="utf-8"
    )
    for item in range(1, 9):
        (Path(config["data"]["videos_dir"]) / f"{item}.mp4").write_bytes(b"video")
    for model in ("qwen", "bge"):
        config["models"][model] = str(tmp_path / model)
        (tmp_path / model).mkdir()
    for arm in ("graph", "description"):
        for field in ("scene_prompt", "summary_prompt"):
            config["extraction"][arm][field] = str(ROOT / config["extraction"][arm][field])
    config["extraction"]["visual_evidence"]["image_resolution"] = [16, 8]
    config["validation"]["cohort"]["user_count"] = 1
    config["validation"]["model"].update(max_epochs=1, patience=1)
    config["validation"]["evaluation"]["bootstrap_samples"] = 20
    (tmp_path / "config").mkdir()
    (tmp_path / "config/pipeline.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(cohort_module, "probe_duration", lambda _: 30.1)
    return RunContext.load("target", root=tmp_path)


@pytest.fixture
def extracted(monkeypatch):
    calls = []

    def extract(source, timestamps, output, size):
        calls.append(Path(source).stem)
        output = Path(output)
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        for timestamp in timestamps:
            Image.new("RGB", size, "white").save(output / f"{timestamp:04d}.png")

    monkeypatch.setattr(fixed30, "extract_resized_keyframes", extract)
    return calls


def _donor_and_target(context, extracted):
    config = deepcopy(context.config)
    config["validation"]["cohort"]["user_count"] = 3
    donor = replace(
        context, run_id="donor", run_root=context.run_root.parent / "donor", config=config
    )
    validation_steps.prepare_cohort_step(donor)
    prepare_input_data(donor)
    # A legacy run is allowed only as an evidence donor, not as a cohort cache.
    write_json(
        donor.cohort_dir / "eligibility_summary.json",
        {
            "schema_version": "microlens-cohort-eligibility/v2",
        },
    )
    write_jsonl(donor.graph_scene_dir("qwen") / "never-copy.jsonl", [{"graph": {}}])
    validation_steps.prepare_cohort_step(context)
    extracted.clear()
    return donor, context.require_ready_cohort()


def _snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_reuse_copies_only_required_verified_evidence_and_leaves_donor_unchanged(
    context, extracted
):
    donor, cohort = _donor_and_target(context, extracted)
    before = _snapshot(donor.run_root)
    result = prepare_input_data(context, reuse_run_id=donor.run_id)
    assert result["reused_donor"] == len(cohort["catalog"]) == 6
    assert result["extracted"] == 0 and extracted == []
    assert not (context.run_root / "extraction").exists()
    assert {path.name for path in (context.evidence_dir / "resized_keyframes").iterdir()} == {
        row["content_id"] for row in cohort["catalog"]
    }
    for row in cohort["catalog"]:
        target_stamp, target_frames = evidence_paths(context.run_root, row["content_id"])
        source_stamp, source_frames = evidence_paths(donor.run_root, row["content_id"])
        assert target_stamp.read_bytes() == source_stamp.read_bytes()
        assert _snapshot(target_frames) == _snapshot(source_frames)
    assert _snapshot(donor.run_root) == before
    write_jsonl(context.cohort_dir / "preparation_failures.jsonl", [{"error": "stale"}])
    second = prepare_input_data(context, reuse_run_id=donor.run_id)
    assert second["reused_target"] == 6 and second["reused_donor"] == 0
    assert not (context.cohort_dir / "preparation_failures.jsonl").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "corrupt",
        "empty",
        "truncated",
        "wrong_size",
        "extra_png",
        "missing_timestamp",
        "wrong_timestamp",
        "source_video_path",
        "source_file_size",
        "source_mtime_ns",
        "duration_seconds",
        "content_id",
        "missing_metadata",
        "missing_inventory",
        "duplicate_inventory",
    ],
)
def test_invalid_donor_falls_back_per_content_without_changing_selection(
    context, extracted, problem
):
    donor, cohort = _donor_and_target(context, extracted)
    row = cohort["catalog"][0]
    timestamp, frames = evidence_paths(donor.run_root, row["content_id"])
    image = frames / "0005.png"
    inventory_path = donor.cohort_dir / "item_inventory.jsonl"
    inventory = read_jsonl(inventory_path)
    record = next(value for value in inventory if value["item_id"] == row["item_id"])
    if problem == "corrupt":
        image.write_bytes(b"invalid PNG")
    elif problem == "empty":
        image.write_bytes(b"")
    elif problem == "truncated":
        image.write_bytes(image.read_bytes()[:16])
    elif problem == "wrong_size":
        Image.new("RGB", (32, 8)).save(image)
    elif problem == "extra_png":
        Image.new("RGB", (16, 8)).save(frames / "9999.png")
    elif problem == "missing_timestamp":
        timestamp.unlink()
    elif problem == "wrong_timestamp":
        timestamp.write_text('[{"keyframe_timestamps": [5]}]', encoding="utf-8")
    elif problem == "missing_inventory":
        inventory_path.unlink()
    else:
        if problem == "missing_metadata":
            record.pop("source_mtime_ns")
        elif problem == "duplicate_inventory":
            inventory.append(dict(record))
        elif problem in {"source_video_path", "content_id"}:
            record[problem] += "-different"
        else:
            record[problem] += 1
        write_jsonl(inventory_path, inventory)
    before = _snapshot(donor.run_root)
    selected = (context.cohort_dir / "selected_users.jsonl").read_bytes()
    result = prepare_input_data(context, reuse_run_id=donor.run_id)
    expected = 6 if problem in {"missing_inventory", "duplicate_inventory"} else 1
    assert result["extracted"] == len(extracted) == expected
    assert result["reused_donor"] == 6 - expected
    assert (context.cohort_dir / "selected_users.jsonl").read_bytes() == selected
    assert _snapshot(donor.run_root) == before


def test_same_run_corrupt_evidence_is_not_reused_by_exists_only_fast_path(context, extracted):
    validation_steps.prepare_cohort_step(context)
    prepare_input_data(context)
    row = context.require_ready_cohort()["catalog"][0]
    _, frames = evidence_paths(context.run_root, row["content_id"])
    (frames / "0005.png").write_bytes(b"")
    extracted.clear()
    result = prepare_input_data(context)
    assert result["extracted"] == 1
    assert extracted == [row["item_id"]]


def test_failed_copy_does_not_commit_a_partial_content(context, extracted, monkeypatch):
    donor, cohort = _donor_and_target(context, extracted)
    row = cohort["inventory"][0]
    donor_row = next(
        value
        for value in read_jsonl(donor.cohort_dir / "item_inventory.jsonl")
        if value["item_id"] == row["item_id"]
    )

    def failed_copy(*_args, **_kwargs):
        raise OSError("simulated copy interruption")

    monkeypatch.setattr(shutil, "copy2", failed_copy)
    assert not copy_matching_evidence(
        target_root=context.run_root,
        donor_root=donor.run_root,
        current=row,
        donor=donor_row,
        image_size=(16, 8),
    )
    stamp, frames = evidence_paths(context.run_root, row["content_id"])
    assert not stamp.exists() and not frames.exists()


def test_copy_interrupt_restores_the_previous_target_content(context, extracted, monkeypatch):
    donor, cohort = _donor_and_target(context, extracted)
    row = cohort["inventory"][0]
    donor_row = next(
        value
        for value in read_jsonl(donor.cohort_dir / "item_inventory.jsonl")
        if value["item_id"] == row["item_id"]
    )
    timestamp, frames = evidence_paths(context.run_root, row["content_id"])
    frames.mkdir(parents=True)
    (frames / "old-partial.txt").write_bytes(b"preserve me until a successful commit")
    timestamp.parent.mkdir(parents=True)
    timestamp.write_bytes(b"old timestamp")
    original_replace = Path.replace

    def interrupt_timestamp_commit(path, target):
        if path.name == "timestamp.json":
            raise KeyboardInterrupt
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_timestamp_commit)
    with pytest.raises(KeyboardInterrupt):
        copy_matching_evidence(
            target_root=context.run_root,
            donor_root=donor.run_root,
            current=row,
            donor=donor_row,
            image_size=(16, 8),
        )
    assert _snapshot(frames) == {"old-partial.txt": b"preserve me until a successful commit"}
    assert timestamp.read_bytes() == b"old timestamp"


def test_source_changed_after_preparation_is_not_mistaken_for_valid_cached_evidence(
    context, extracted
):
    validation_steps.prepare_cohort_step(context)
    prepare_input_data(context)
    row = context.require_ready_cohort()["inventory"][0]
    Path(row["source_video_path"]).write_bytes(b"a different source video")
    extracted.clear()
    with pytest.raises(RuntimeError, match="source video changed or is missing"):
        prepare_input_data(context)
    assert extracted == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reuse_run_id": "target"},
        {"reuse_run_id": "../donor"},
        {"reuse_run_id": "donor", "force": True},
    ],
)
def test_invalid_reuse_arguments_are_rejected(context, kwargs):
    with pytest.raises(RuntimeError):
        prepare_input_data(context, **kwargs)


@pytest.mark.parametrize("state", ["planned", "blocked"])
def test_all_downstream_commands_require_ready_cohort_before_backend_work(context, state):
    validation_steps.prepare_cohort_step(context, plan_only=True)
    if state == "blocked":
        Path(context.config["data"]["titles_csv"]).unlink()
        with pytest.raises(RuntimeError):
            validation_steps.prepare_cohort_step(context)
    for step, kwargs in [
        (prepare_input_data, {}),
        (extraction_steps.extract_graph_scenes, {"model": "qwen"}),
        (extraction_steps.extract_graph_scenes, {"model": "gemini"}),
        (extraction_steps.summarize_graph, {"source": "qwen"}),
        (extraction_steps.summarize_graph, {"source": "gemini"}),
        (extraction_steps.extract_description_scenes, {}),
        (extraction_steps.summarize_description, {}),
        (validation_steps.embed_representations, {}),
        (validation_steps.run_recommendation, {}),
    ]:
        with pytest.raises(RuntimeError, match="not ready"):
            step(context, **kwargs)
    with pytest.raises(RuntimeError, match="runtime diagnosis failed"):
        validation_steps.run_diagnosis(context)
    report = read_json(context.diagnosis_path)
    assert report["schema_version"] == "diagnosis/v4"
    assert report["runtime_decision"]["checks"]["cohort_plan_matches_artifacts"] is False
    assert report["statistical_analysis"]["status"] == "not_computed"


def _mock_recommendation(config, runtime):
    cohort = config.output_dir / "data/cohort"
    catalog = read_jsonl(cohort / "catalog.jsonl")
    users = read_jsonl(cohort / "sequences.jsonl")
    mapping = {row["item_id"]: row["content_id"] for row in catalog}
    frequency = Counter(item for row in users for item in row["train"])
    median = sorted(frequency.values())[len(frequency) // 2]
    output = Path(runtime["paths"]["recommendations_dir"])
    runs, metrics = [], []
    for seed in config.model.seeds:
        for arm, branch in RECOMMENDATION_ARMS.items():
            checkpoint = output / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"mock checkpoint; core profile does not train")
            runs.append(
                recommendation._training_run_record(
                    run_id=config.run_id,
                    seed=seed,
                    arm=arm,
                    branch=branch,
                    selection_history=[{"epoch": 1, "loss": 1.0, "NDCG@10": 1.0}],
                    best_ndcg=1.0,
                    best_epoch=1,
                    refit_history=[{"epoch": 1, "loss": 1.0}],
                    checkpoint=checkpoint,
                    candidate_count=len(catalog),
                    elapsed_seconds=1.0,
                    max_epochs=1,
                )
            )
            for row in users:
                target = row["test_target"]
                top = [target] + [item for item in mapping if item != target]
                count = frequency[target]
                metrics.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "branch": branch,
                        "user_id": row["user_id"],
                        "candidate_count": len(catalog),
                        "target_item_id": target,
                        "target_content_id": mapping[target],
                        "rank": 1,
                        "top_item_ids": top,
                        "top_content_ids": [mapping[item] for item in top],
                        "target_frequency_bucket": "cold"
                        if count == 0
                        else "low"
                        if count <= median
                        else "warm",
                        **metrics_from_rank(1, config.evaluation.cutoffs),
                    }
                )
    write_jsonl(output / "training_runs.jsonl", runs)
    write_jsonl(output / "per_user_metrics.jsonl", metrics)


@pytest.mark.parametrize("user_count,expected_catalog", [(1, 6), (3, 8)])
def test_eleven_stage_cli_runs_on_selected_user_catalog(
    context, extracted, monkeypatch, user_count, expected_catalog
):
    context.config["validation"]["cohort"]["user_count"] = user_count
    monkeypatch.setattr(RunContext, "load", lambda _run_id: context)
    graph_prompt = context.config_path("extraction", "graph", "scene_prompt").read_text(
        encoding="utf-8"
    )

    @contextmanager
    def generator(**_kwargs):
        def generate(tasks, callback):
            for task in tasks:
                if not task.image_paths:
                    response = "\n".join(f"{field}: A person walks." for field in SUMMARY_SECTIONS)
                elif task.prompt.startswith(graph_prompt.strip()):
                    response = "{}"
                else:
                    response = "A person walks."
                callback(task.task_id, response)
            return {}

        yield generate

    class Gemini:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, tasks, callback):
            for task in reversed(tasks):
                callback(extraction_steps.GeminiGenerationOutcome(task.task_id, "{}"))

    class Encoder:
        def __init__(self, _settings):
            pass

        def encode(self, texts):
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(extraction_steps, "qwen_generator", generator)
    monkeypatch.setattr(extraction_steps, "GeminiWorkerPool", Gemini)
    monkeypatch.setattr(features, "BGETextEncoder", Encoder)
    monkeypatch.setattr(recommendation, "train_recommendation_arms", _mock_recommendation)
    assert validation_cli.main(["prepare-cohort", "--run-id", context.run_id, "--plan-only"]) == 0
    assert validation_cli.main(["prepare-cohort", "--run-id", context.run_id]) == 0
    for args in [
        ["prepare-input-data"],
        ["extract-graph-scenes", "--model", "qwen"],
        ["summarize-graph", "--source", "qwen"],
        ["extract-graph-scenes", "--model", "gemini"],
        ["summarize-graph", "--source", "gemini"],
        ["extract-description-scenes"],
        ["summarize-description"],
    ]:
        assert extraction_cli.main([*args, "--run-id", context.run_id]) == 0
    for step in ("embed-representations", "run-recommendation", "run-diagnosis"):
        assert validation_cli.main([step, "--run-id", context.run_id]) == 0
    report = read_json(context.diagnosis_path)
    assert report["schema_version"] == "diagnosis/v4"
    assert report["runtime_decision"]["status"] == "pass"
    assert report["cohort"]["catalog_scope"] == "selected_user_sequence_union"
    assert report["cohort"]["selected_user_count"] == user_count
    assert report["artifact_integrity"]["catalog_size"] == expected_catalog
    assert report["artifact_integrity"]["training_runs"]["observed_row_count"] == 12
    assert report["statistical_analysis"]["computed_comparison_count"] == 6
    for branch in RECOMMENDATION_ARMS.values():
        with np.load(context.representations_dir / f"{branch}_embeddings.npz") as arrays:
            assert arrays["values"].shape == (expected_catalog, 1024)
    assert not list((context.run_root / "extraction").glob("**/failures/*.jsonl"))
