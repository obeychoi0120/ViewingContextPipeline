"""Dataset isolation, raw M3L semantics and lossless legacy layout relocation."""

from pathlib import Path

import pytest

from artifact_io import (
    file_fingerprint,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
)
from helpers import ready_cohort, result_identity, train_inputs
from pipeline_runtime import ConfigError, RunContext
from preparation.cohort import load_cohort, prepare_full_cohort
from preparation.migrate_dataset_layout import JOURNAL, migrate
from validation.selection import (
    load_validation_cohort,
    prepare_validation_cohort,
    training_signature,
)
from validation.shared_cache import SharedCache
from validation.steps import validation_config


def make_m3l(context, monkeypatch):
    root = context.root / "m3l"
    meta, videos = root / "metadata", root / "videos"
    meta.mkdir(parents=True)
    videos.mkdir()
    movies = list(range(1, 19))
    (meta / "id_mmrec_to_grouplens.tsv").write_text(
        "newID\toldID\n" + "".join(f"{i}\t{movie}\n" for i, movie in enumerate(movies))
    )
    (meta / "movies.dat").write_text(
        "".join(f"{movie}::Movie, {movie} (2001)::Drama\n" for movie in movies)
    )
    rows = [
        f"1::{movie}::{0.5 if movie == 14 else 4}::{100000 + min(movie, 15)}\n"
        for movie in range(1, 17)
    ]
    rows += [f"2::{movie}::1::{200000 + i}\n" for i, movie in enumerate([1, 2, 3, 17, 18])]
    rows += [f"3::{movie}::2::{300000 + movie}\n" for movie in range(1, 6)]
    # Original MovieLens contains movies outside the M3L mapping.
    rows += ["4::99::3::400000\n"]
    (meta / "ratings.dat").write_text("".join(rows))
    for movie in movies:
        if movie != 17:
            (videos / f"{movie}.mp4").write_bytes(b"trailer")
    context.config["datasets"]["m3l_10m"].update(
        data={"root": str(root)},
        source_counts={"user_count": 3, "item_count": 18, "interaction_count": 26},
    )
    import yaml

    (context.root / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in context.config.items() if k != "_rec_experiments"}))
    monkeypatch.setattr("preparation.m3l.shutil.which", lambda tool: "/usr/bin/ffprobe")

    def probe(path):
        if path.stem == "18":
            raise ValueError("no video stream")
        return 32.2

    monkeypatch.setattr("preparation.m3l.probe_video", probe)
    return RunContext.load(context.run_id, dataset="m3l_10m", root=context.root)


def test_dataset_paths_contract_and_cache_are_isolated(context, monkeypatch):
    m3l = make_m3l(context, monkeypatch)
    for ctx in (context, m3l):
        ctx.initialize()
        assert ctx.run_root == ctx.artifact_root / "runs" / ctx.run_id / ctx.dataset
        assert ctx.preparation_dir == ctx.artifact_root / "preparation" / ctx.dataset
        assert ctx.cohort_dir.is_relative_to(ctx.preparation_dir)
        assert not (ctx.run_root / "run_contract.json").exists()
    assert context.keyframes_dir != m3l.keyframes_dir
    assert context.source_assets_dir != m3l.source_assets_dir
    assert context.scene_arm_dir("graph_qwen3vl") != m3l.scene_arm_dir("graph_qwen3vl")
    assert context.diagnosis_path(["base"]) != m3l.diagnosis_path(["base"])
    left, right = (SharedCache(ctx, "features", "same-key") for ctx in (context, m3l))
    source = context.root / "cache-input"
    source.mkdir()
    (source / "x").write_bytes(b"microlens")
    left.publish(source, ["x"], origin={"run_id": context.run_id})
    assert not right.valid()
    assert right.restore(context.root / "wrong-cache") is None
    with pytest.raises(ConfigError, match="unsupported dataset"):
        RunContext.load(context.run_id, dataset="../escape", root=context.root)


def test_m3l_filtering_timestamps_sequences_and_frozen_catalog(context, monkeypatch):
    ctx = make_m3l(context, monkeypatch)
    ctx.initialize()
    assert validation_config(ctx).dataset.root == ctx.path("data", "root")
    assert prepare_full_cohort(ctx)["status"] == "ready"
    source = ctx.require_ready_cohort()
    assert source["plan"]["source_counts"] == ctx.source_counts
    assert source["plan"]["interaction_count"] == 24
    assert source["plan"]["excluded_interaction_count"] == 2
    excluded = read_jsonl(ctx.cohort_dir / "excluded_items.jsonl")
    assert [(row["item_id"], row["reason"], row["interaction_count"]) for row in excluded] == [
        ("17", "missing_video", 1),
        ("18", "invalid_video", 1),
    ]
    assert source["catalog"][0]["content_id"] == "m3l_10m_00001"
    assert source["metadata_titles"][0]["title"] == "Movie, 1 (2001)"
    rows = read_jsonl(ctx.cohort_dir / "events.jsonl")
    assert rows[0]["timestamp"] == 100001000
    assert all("rating" not in row for row in rows)
    selected = prepare_validation_cohort(ctx)
    assert selected["plan"]["source_data_hash"] == fingerprint(rows)
    assert selected["plan"]["user_count"] == 2
    assert selected["plan"]["interaction_count"] == 18
    assert selected["plan"]["trimmed_event_count"] == 3
    assert selected["plan"]["excluded_users"] == [
        {"user_id": "2", "reason": "short_history", "event_count": 3}
    ]
    user1 = [row for row in selected["events"] if row["user_id"] == "1"]
    assert [row["item_id"] for row in user1] == [str(movie) for movie in range(4, 17)]
    split = selected["plan"]["splits"][0]
    user_split = next(row for row in split["users"] if row["user_id"] == "1")
    assert selected["events"][user_split["validation"]]["item_id"] == "15"
    assert selected["events"][user_split["test"]]["item_id"] == "16"
    # A newly downloaded trailer cannot change an already prepared cohort.
    (ctx.path("data", "root") / "videos" / "17.mp4").write_bytes(b"later trailer")
    assert load_validation_cohort(ctx)["manifest"] == selected["manifest"]
    report = (ctx.run_root / "reports" / "leave_one_out_cohort.md").read_text()
    assert "평가한 영화 이력의 다음 영화 예측" in report and "m3l_10m" in report


@pytest.mark.torch
def test_m3l_extraction_embedding_training_diagnosis_and_resume(context, monkeypatch):
    from validation.steps import run_diagnosis, run_recommendation

    ctx = make_m3l(context, monkeypatch)
    trained = train_inputs(ctx, monkeypatch)
    assert trained["completed"] == 6
    diagnosed = run_diagnosis(ctx, target=["base", "graph-activity"])
    assert diagnosed["status"] == "pass"
    assert read_json(ctx.diagnosis_path(["base", "graph-activity"]))["dataset"] == ctx.dataset
    report = (ctx.run_root / "reports" / "base_graph-activity" / "report.md").read_text()
    assert "m3l_10m" in report and "평가한 영화 이력의 다음 영화 예측" in report
    assert run_recommendation(ctx, target=["base", "graph-activity"])["skipped"] == 6


def _make_legacy(context):
    """Undo only the new layout in a fixture; leave every data byte untouched."""
    for parent in (
        context.artifact_root / "preparation",
        context.artifact_root / "shared_cache",
        context.artifact_root / "runs" / context.run_id,
    ):
        namespace = parent / context.dataset
        if namespace.is_dir():
            for path in list(namespace.iterdir()):
                path.rename(parent / path.name)
            namespace.rmdir()


@pytest.mark.torch
def test_migration_preserves_selection_features_and_completed_training(context, monkeypatch):
    from extraction.steps import extract_graph_scenes
    from helpers import FakeGenerator
    from preparation.input_data import prepare_input_data
    from validation.leave_one_out_recommendation import combination_complete
    from validation.steps import run_recommendation

    train_inputs(context, monkeypatch)
    before = load_validation_cohort(context)
    signature = training_signature(context, before, validation_config(context))
    directory = context.recommendations_dir / "seed_42" / "base"
    checkpoint = file_fingerprint(directory / "sasrec.pt")
    feature = file_fingerprint(context.representations_dir / "metadata" / "features.npy")
    _make_legacy(context)
    with pytest.raises(ConfigError, match="legacy MicroLens layout"):
        RunContext.load(context.run_id, dataset=context.dataset, root=context.root)
    preview = migrate(context.artifact_root)
    assert preview["moves"] and not (context.artifact_root / JOURNAL).exists()
    assert not context.run_root.exists()
    migrate(context.artifact_root, apply=True)
    migrated = RunContext.load(context.run_id, dataset=context.dataset, root=context.root)
    after = load_validation_cohort(migrated)
    assert after == before
    assert training_signature(migrated, after, validation_config(migrated)) == signature
    assert file_fingerprint(directory / "sasrec.pt") == checkpoint
    assert file_fingerprint(context.representations_dir / "metadata" / "features.npy") == feature
    assert combination_complete(directory, result_identity(migrated, after), 3)
    monkeypatch.setattr(
        "preparation.video.fixed30.extract_resized_keyframes",
        lambda *args: pytest.fail("migration discarded valid keyframes"),
    )
    assert prepare_input_data(migrated)["failure_count"] == 0
    generator = FakeGenerator()
    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    extract_graph_scenes(
        migrated, model="qwen3vl", schema="prompts/scene_graph_v9.md", arm="graph_qwen3vl"
    )
    assert not generator.tasks
    reused = run_recommendation(migrated, target=["base", "graph-activity"])
    assert reused["completed"] == 0 and reused["skipped"] == 6
    assert not (migrated.run_root / "run_contract.json").exists()
    # Repeating the migration preserves the moved artifacts.
    migrate(context.artifact_root, apply=True)


def test_migration_conflicts_resume_and_historical_runs(context, monkeypatch):
    ready_cohort(context)
    _make_legacy(context)
    historical = context.artifact_root / "runs" / "historical"
    historical.mkdir()
    old_contract = {"experiment_config_version": "v1"}
    write_json(historical / "run_contract.json", old_contract)
    (historical / "reports").mkdir()
    (historical / "reports" / "original.md").write_text("original report")
    destination = context.artifact_root / "preparation" / context.dataset
    destination.mkdir()
    with pytest.raises(FileExistsError):
        migrate(context.artifact_root, apply=True)
    assert not (context.artifact_root / JOURNAL).exists()
    destination.rmdir()
    original_rename = Path.rename
    interrupted = False

    def stop_after_rename(path, target):
        nonlocal interrupted
        result = original_rename(path, target)
        if not interrupted and path.name == "cohort":
            interrupted = True
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", stop_after_rename)
        with pytest.raises(KeyboardInterrupt):
            migrate(context.artifact_root, apply=True)
    assert any(
        row["status"] == "moving" for row in read_json(context.artifact_root / JOURNAL)["moves"]
    )
    migrate(context.artifact_root, apply=True)
    assert load_cohort(context.cohort_dir)["plan"]["item_count"] > 0
    assert read_json(historical / context.dataset / "run_contract.json") == old_contract
    assert (
        historical / context.dataset / "reports" / "original.md"
    ).read_text() == "original report"
    restored = RunContext.load("historical", dataset=context.dataset, root=context.root)
    restored.initialize()
    assert read_json(historical / context.dataset / "run_contract.json") == old_contract
