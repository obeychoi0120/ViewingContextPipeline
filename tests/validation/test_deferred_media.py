from dataclasses import replace
from pathlib import Path
from threading import Event, current_thread, main_thread
from unittest.mock import patch

from PIL import Image
import pytest

from extraction.data_preparation.media import cached_duration
from extraction.preparation import prepare_input_data
from pipeline_runtime import RunContext, read_json, read_jsonl, write_jsonl
from validation.rolling_data import prepare_full_cohort
from visual_sampling import timestamp_stem

# Reuse the small full-rolling source fixture; no real models or media are loaded.
import test_rolling

full_context = test_rolling.full_context


@pytest.fixture
def media_calls(full_context, monkeypatch):
    full_context.config["extraction"]["visual_evidence"]["image_resolution"] = [16, 8]
    calls = {"probe": [], "extract": []}

    def probe(source):
        calls["probe"].append(source.stem)
        return 31.0

    def extract(source, timestamps, output, size):
        calls["extract"].append(source.stem)
        output.mkdir(parents=True, exist_ok=True)
        for path in output.glob("*.png"):
            path.unlink()
        for timestamp in timestamps:
            Image.new("RGB", size).save(output / f"{timestamp_stem(timestamp)}.png")

    monkeypatch.setattr("extraction.data_preparation.media.probe_duration", probe)
    monkeypatch.setattr("extraction.data_preparation.fixed30.extract_resized_keyframes", extract)
    return calls, probe, extract


def test_probe_and_extraction_share_workers_and_resume_without_probe(
    full_context, media_calls, monkeypatch,
):
    calls, probe, extract = media_calls
    first_extraction = Event()

    def concurrent_probe(source):
        assert current_thread() is not main_thread()
        if source.stem == "4":
            assert first_extraction.wait(5), "extraction must start before all probes finish"
        return probe(source)

    def signal_extraction(*args):
        extract(*args)
        first_extraction.set()

    monkeypatch.setattr("extraction.data_preparation.media.probe_duration", concurrent_probe)
    monkeypatch.setattr("extraction.data_preparation.fixed30.extract_resized_keyframes", signal_extraction)
    result = prepare_input_data(full_context)
    assert result["extracted"] == 4
    assert sorted(calls["probe"]) == sorted(calls["extract"]) == ["1", "2", "3", "4"]
    report = read_json(full_context.cohort_dir / "media_preflight.json")
    assert report["duration_seconds"] == 124.0
    assert report["scene_count"] == 8 and report["keyframe_count"] == 28
    assert report["uncompressed_rgb_bytes"] == 28 * 16 * 8 * 3
    # Cohort reruns keep duration checkpoints; sampling is independent of duration.
    prepare_full_cohort(full_context)
    assert not (full_context.cohort_dir / "media_preflight.json").exists()
    calls["probe"].clear()
    calls["extract"].clear()
    assert prepare_input_data(full_context)["reused_target"] == 4
    assert calls == {"probe": [], "extract": []}
    full_context.config["extraction"]["visual_evidence"]["num_keyframes"] = 3
    assert prepare_input_data(full_context)["extracted"] == 4
    assert calls["probe"] == []
    assert read_json(full_context.cohort_dir / "media_preflight.json")["keyframe_count"] == 16


@pytest.mark.parametrize("failure_phase", ["probe", "extract"])
def test_failed_video_resumes_and_preserves_successful_duration_checkpoints(
    full_context, media_calls, monkeypatch, failure_phase,
):
    calls, probe, extract = media_calls

    def fail_probe(source):
        if source.stem == "2" and failure_phase == "probe":
            raise RuntimeError("fixture probe failure")
        return probe(source)

    def fail_extract(source, *args):
        if source.stem == "2" and failure_phase == "extract":
            raise RuntimeError("fixture extract failure")
        return extract(source, *args)

    monkeypatch.setattr("extraction.data_preparation.media.probe_duration", fail_probe)
    monkeypatch.setattr("extraction.data_preparation.fixed30.extract_resized_keyframes", fail_extract)
    with pytest.raises(RuntimeError, match="preparation is incomplete"):
        prepare_input_data(full_context)
    failures = read_jsonl(full_context.cohort_dir / "preparation_failures.jsonl")
    assert len(failures) == 1 and failures[0]["item_id"] == "2"
    assert not (full_context.cohort_dir / "media_preflight.json").exists()
    calls["probe"].clear()
    calls["extract"].clear()
    monkeypatch.setattr("extraction.data_preparation.media.probe_duration", probe)
    monkeypatch.setattr("extraction.data_preparation.fixed30.extract_resized_keyframes", extract)
    result = prepare_input_data(full_context)
    assert result["reused_target"] == 3 and result["extracted"] == 1
    assert calls["extract"] == ["2"]
    assert calls["probe"] == (["2"] if failure_phase == "probe" else [])
    assert not (full_context.cohort_dir / "preparation_failures.jsonl").exists()


@pytest.mark.parametrize("problem", ["missing", "empty", "duplicate"])
def test_cohort_still_rejects_invalid_file_inventory(full_context, problem):
    source = Path(full_context.config["data"]["videos_dir"]) / "2.mp4"
    if problem == "missing":
        source.unlink()
    elif problem == "empty":
        source.write_bytes(b"")
    else:
        source.with_name("02.mp4").write_bytes(b"duplicate")
    with pytest.raises(RuntimeError, match="unresolved assets"):
        prepare_full_cohort(full_context)
    failures = read_jsonl(full_context.cohort_dir / "preparation_failures.jsonl")
    assert len(failures) == 1 and failures[0]["item_id"] == "2"
    assert read_json(full_context.cohort_dir / "eligibility.json")["status"] == "blocked"


@pytest.mark.parametrize("new_duration", [31.0, 32.0])
def test_changed_source_does_not_reuse_duration_checkpoint(full_context, media_calls, new_duration):
    calls, _, _ = media_calls
    prepare_input_data(full_context)
    source = Path(full_context.config["data"]["videos_dir"]) / "2.mp4"
    source.write_bytes(b"changed video bytes")
    with pytest.raises(RuntimeError, match="source video changed"):
        prepare_input_data(full_context)
    prepare_full_cohort(full_context)
    row = full_context.require_ready_cohort()["inventory"][1]
    assert cached_duration(full_context.cohort_dir / "source_assets", row) is None
    calls["probe"].clear()
    calls["extract"].clear()
    # Replaced source requires fresh images even when its duration is identical.
    with patch("extraction.data_preparation.media.probe_duration", return_value=new_duration) as probe:
        result = prepare_input_data(full_context)
    assert probe.call_count == 1 and result["extracted"] == 1
    assert calls["extract"] == ["2"]
    assert cached_duration(full_context.cohort_dir / "source_assets", row) == new_duration


@pytest.mark.parametrize("legacy_duration", [False, True])
def test_reuse_donor_with_checkpoint_or_existing_inventory_duration(
    full_context, media_calls, monkeypatch, legacy_duration,
):
    calls, _, _ = media_calls
    donor = full_context
    if legacy_duration:
        for name in ("catalog.jsonl", "item_inventory.jsonl"):
            rows = read_jsonl(donor.cohort_dir / name)
            for row in rows:
                row["duration_seconds"] = 31.0
            write_jsonl(donor.cohort_dir / name, rows)
    prepare_input_data(donor)
    if legacy_duration:
        assert calls["probe"] == []
        prepare_full_cohort(donor)
        assert prepare_input_data(donor)["reused_target"] == 4
        assert calls["probe"] == []
    target = replace(donor, run_id="target", run_root=donor.run_root.parent / "target")
    prepare_full_cohort(target)
    monkeypatch.setattr(RunContext, "load", lambda run_id, root: donor)
    calls["probe"].clear()
    calls["extract"].clear()
    result = prepare_input_data(target, reuse_run_id=donor.run_id)
    assert result["reused_donor"] == 4 and result["extracted"] == 0
    assert calls == {"probe": [], "extract": []}
    assert prepare_input_data(target)["reused_target"] == 4


def test_failed_replacement_cannot_reuse_old_images_on_retry(full_context, media_calls, monkeypatch):
    calls, _, extract = media_calls
    prepare_input_data(full_context)
    source = Path(full_context.config["data"]["videos_dir"]) / "2.mp4"
    source.write_bytes(b"replacement with the same duration")
    prepare_full_cohort(full_context)

    def fail_extract(source, *args):
        if source.stem == "2":
            raise RuntimeError("fixture replacement failure")
        return extract(source, *args)

    monkeypatch.setattr("extraction.data_preparation.fixed30.extract_resized_keyframes", fail_extract)
    with pytest.raises(RuntimeError, match="preparation is incomplete"):
        prepare_input_data(full_context)
    calls["probe"].clear()
    calls["extract"].clear()
    monkeypatch.setattr("extraction.data_preparation.fixed30.extract_resized_keyframes", extract)
    result = prepare_input_data(full_context)
    assert result["reused_target"] == 3 and result["extracted"] == 1
    assert calls == {"probe": [], "extract": ["2"]}


def test_corrupt_duration_checkpoint_is_reprobed(full_context, media_calls):
    calls, _, _ = media_calls
    prepare_input_data(full_context)
    item = full_context.require_ready_cohort()["inventory"][0]
    path = full_context.cohort_dir / "source_assets" / item["content_id"] / "assets/video_duration.json"
    path.write_text("incomplete JSON", encoding="utf-8")
    calls["probe"].clear()
    calls["extract"].clear()
    assert prepare_input_data(full_context)["extracted"] == 1
    assert calls == {"probe": ["1"], "extract": ["1"]}
