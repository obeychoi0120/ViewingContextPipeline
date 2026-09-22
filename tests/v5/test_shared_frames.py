from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from extraction.data_preparation.video_processor import extract_resized_keyframes
from preparation.input_data import prepare_input_data
from pipeline_runtime import RunContext
from preparation.steps import prepare_cohort_step


def test_second_run_reuses_shared_timestamps_and_frames(ready_context, monkeypatch, capsys):
    first = ready_context
    images = {p: p.read_bytes() for p in first.keyframes_dir.rglob("*.png")}
    stamps = {p: p.stat().st_mtime_ns for p in images}
    second = RunContext.load("second", root=first.root)
    assert second.cohort_dir == first.cohort_dir
    assert second.require_ready_cohort() == first.require_ready_cohort()
    assert not list(second.cohort_dir.rglob("timestamp_fixed*.json"))
    monkeypatch.setattr(
        "extraction.data_preparation.video_processor.subprocess.run",
        lambda *a, **k: pytest.fail("shared frames must not be extracted again"),
    )
    capsys.readouterr()
    with monkeypatch.context() as patch:
        patch.setattr("extraction.data_preparation.media.probe_duration",
                      lambda *a: pytest.fail("shared duration must not be probed again"))
        from extraction.step_support import visual_rows
        assert visual_rows(second)
        prepare_input_data(second)
    output = capsys.readouterr()
    assert "new_frames=0" in output.out
    assert "[KEYFRAMES] extracting" not in output.out
    assert "Extract resized keyframes" not in output.err
    prepare_input_data(second, force=True)
    assert list(second.source_assets_dir.rglob("timestamp_fixed*.json"))
    assert not (second.cohort_dir / "source_assets").exists()
    for row in second.require_ready_cohort()["catalog"]:
        directory = second.source_assets_dir / row["content_id"]
        assert (directory / "video_duration.json").is_file()
        assert (directory / "timestamp_fixed_30s.json").is_file()
        assert not (directory / "assets").exists()
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


def test_extraction_builds_paths_without_probing_assets(ready_context, fake_models, monkeypatch):
    from extraction.steps import extract_description_scenes

    context = ready_context
    original_is_file, original_is_dir = Path.is_file, Path.is_dir
    original_iterdir, original_open = Path.iterdir, Path.open
    def asset(path):
        return path.is_relative_to(context.keyframes_dir) or path.is_relative_to(context.source_assets_dir)
    def is_file(path):
        assert not asset(path), f"unexpected asset is_file: {path}"
        return original_is_file(path)
    def is_dir(path):
        assert not asset(path), f"unexpected asset is_dir: {path}"
        return original_is_dir(path)
    def iterdir(path):
        assert not asset(path), f"unexpected asset scan: {path}"
        return original_iterdir(path)
    reads = []
    def open_file(path, *a, **kw):
        if asset(path):
            assert path.name.startswith("timestamp_fixed"), f"unexpected asset read: {path}"
            reads.append(path)
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(Path, "open", open_file)
    extract_description_scenes(context, model="qwen", schema="prompts/scene_description_v2.md")
    assert len(reads) == len(set(reads)) == 4
    assert fake_models


def test_preparation_rejects_changed_source_without_overwriting_shared_metadata(ready_context):
    from extraction.step_support import visual_rows

    first = ready_context
    before = {p: p.read_bytes() for p in first.source_assets_dir.rglob("*.json")}
    source = first.path("data", "videos_dir") / "1.mp4"
    source.write_bytes(b"a different source video")
    second = RunContext.load("changed_source", root=first.root)
    prepare_cohort_step(second)
    assert visual_rows(second)
    with pytest.raises(RuntimeError, match="preparation incomplete"):
        prepare_input_data(second)
    assert all(p.read_bytes() == data for p, data in before.items())
