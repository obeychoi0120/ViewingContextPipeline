from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from extraction.data_preparation.video_processor import extract_resized_keyframes
from extraction.preparation import prepare_input_data
from pipeline_runtime import RunContext
from validation.steps import prepare_cohort_step


def test_second_run_creates_timestamps_and_preserves_shared_frames(ready_context, monkeypatch, capsys):
    first = ready_context
    images = {p: p.read_bytes() for p in first.keyframes_dir.rglob("*.png")}
    stamps = {p: p.stat().st_mtime_ns for p in images}
    second = RunContext.load("second", root=first.root)
    prepare_cohort_step(second)
    assert not list(second.cohort_dir.rglob("timestamp_fixed*.json"))
    monkeypatch.setattr(
        "extraction.data_preparation.video_processor.subprocess.run",
        lambda *a, **k: pytest.fail("shared frames must not be extracted again"),
    )
    capsys.readouterr()
    prepare_input_data(second)
    output = capsys.readouterr()
    assert "new_frames=0" in output.out
    assert "[KEYFRAMES] extracting" not in output.out
    assert "Extract resized keyframes" not in output.err
    prepare_input_data(second, force=True)
    assert list(second.cohort_dir.rglob("timestamp_fixed*.json"))
    assert images == {p: p.read_bytes() for p in images}
    assert stamps == {p: p.stat().st_mtime_ns for p in images}


def test_missing_frames_concurrent_writers_and_invalid_existing(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "images"
    output.mkdir()
    existing = output / "0001.png"
    Image.new("RGB", (16, 8), "red").save(existing)
    original = existing.read_bytes(), existing.stat().st_mtime_ns
    calls = []

    def ffmpeg(args, **kw):
        calls.append(args[args.index("-ss") + 1])
        Image.new("RGB", (16, 8)).save(args[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("extraction.data_preparation.video_processor.subprocess.run", ffmpeg)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(extract_resized_keyframes, source, [1, 2], output, (16, 8))
            for _ in range(2)
        ]
        counts = [future.result() for future in futures]
        assert sum(row["extracted_frames"] for row in counts) == 1
        assert sum(row["reused_frames"] for row in counts) == 3
    assert calls == ["2"]
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == original
    assert {p.name for p in output.iterdir()} == {"0001.png", "0002.png"}
    (output / "0002.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="existing shared frame"):
        extract_resized_keyframes(source, [1, 2], output, (16, 8))
    assert calls == ["2"]


def test_failed_preparation_resumes_without_new_probe_for_completed_duration(
    v5_context, monkeypatch
):
    context = v5_context
    prepare_cohort_step(context)
    probes, extracted = [], []

    def probe(source):
        probes.append(source.stem)
        return 10.0

    original_failed = {"4"}

    def ffmpeg(args, **kw):
        source = Path(args[args.index("-i") + 1]).stem
        if source in original_failed:
            return SimpleNamespace(returncode=1, stdout="", stderr="fixture failure")
        extracted.append(source)
        Image.new("RGB", (16, 8)).save(args[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("extraction.data_preparation.media.probe_duration", probe)
    monkeypatch.setattr("extraction.data_preparation.video_processor.subprocess.run", ffmpeg)
    with pytest.raises(RuntimeError, match="preparation incomplete"):
        prepare_input_data(context)
    assert sorted(probes) == ["1", "2", "3", "4"]
    assert (context.cohort_dir / "preparation_failures.jsonl").is_file()
    original_failed.clear()
    before = len(extracted)
    prepare_input_data(context)
    assert sorted(probes) == ["1", "2", "3", "4"]
    assert set(extracted[before:]) == {"4"}
    assert not (context.cohort_dir / "preparation_failures.jsonl").exists()


@pytest.mark.parametrize("missing", ["timestamps", "frames", "both"])
def test_extraction_reports_exact_missing_evidence_and_preparation_command(ready_context, missing):
    from extraction.step_support import visual_rows

    context = ready_context
    cid = context.require_ready_cohort()["catalog"][0]["content_id"]
    timestamp = next((context.cohort_dir / "source_assets" / cid).rglob("timestamp_fixed*.json"))
    frames = context.keyframes_dir / cid
    if missing in {"timestamps", "both"}:
        timestamp.unlink()
    if missing in {"frames", "both"}:
        for image in frames.glob("*.png"):
            image.unlink()
    with pytest.raises(RuntimeError) as error:
        visual_rows(context)
    message = str(error.value)
    assert ("run scene timestamps" in message) == (missing in {"timestamps", "both"})
    assert ("shared keyframe images" in message) == (missing in {"frames", "both"})
    if missing in {"timestamps", "both"}:
        assert str(timestamp) in message
    if missing in {"frames", "both"}:
        assert str(frames) in message
    assert f"prepare-input-data --run-id {context.run_id}" in message
