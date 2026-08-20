from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src.video_data_collection.microlens_config import load_microlens_config
from src.video_data_collection.microlens_importer import (
    build_inventory,
    display_aspect_ratio_from_probe,
    load_item_csv,
    load_external_selection,
    run_import,
)
from src.video_data_collection.raw_pipeline import (
    build_content_paths,
    export_resized_keyframes,
    process_local_source,
    process_prepared_source,
)


def _write_config(root: Path) -> Path:
    config = {
        "schema_version": "microlens-import-config/v1",
        "dataset_id": "microlens-100k",
        "source": {
            "videos_dir": "videos",
            "titles_csv": "titles.csv",
            "tags_csv": "tags.csv",
            "pairs_tsv": "pairs.tsv",
        },
        "output_root": "output/microlens",
        "assets_root": "source_assets",
        "processing_config_path": "processing.json",
        "pilot": {
            "size": 4,
            "smoke_size": 3,
            "seed": 17,
            "minimum_per_category": 1,
        },
        "aspect_ratio": {"minimum": 4 / 3, "maximum": 2.0},
        "sampling": {
            "shot_interval": "fixed_30s",
            "interval_seconds": 10,
            "frames_per_scene": 3,
            "ocr_fps": 1,
            "resize_mode": "contain_pad",
            "padding_color": "black",
        },
        "metadata_defaults": {
            "channel": "unknown",
            "upload_date": "unknown",
            "description": "",
        },
    }
    path = root / "microlens.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _prepare_sources(root: Path, *, create_videos: bool = True) -> None:
    videos = root / "videos"
    if create_videos:
        videos.mkdir()
        for item_id in range(1, 7):
            (videos / f"{item_id}.mp4").write_bytes(f"source-{item_id}".encode())
    (root / "titles.csv").write_text(
        "item_id,title\n"
        + "".join(f"{item_id},title {item_id}\n" for item_id in range(1, 7)),
        encoding="utf-8",
    )
    (root / "tags.csv").write_text(
        "item_id,category\n"
        "1,Comedy\n2,Comedy\n3,Music\n4,Music\n5,News\n6,News\n",
        encoding="utf-8",
    )
    (root / "pairs.tsv").write_text(
        "u1\t1 2 3 4 5 6\nu2\t1 2 3 4\n",
        encoding="utf-8",
    )
    (root / "processing.json").write_text("{}", encoding="utf-8")


def test_display_aspect_ratio_uses_rotation_metadata() -> None:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "width": 720,
                "height": 1280,
                "sample_aspect_ratio": "1:1",
                "side_data_list": [{"rotation": -90}],
            }
        ]
    }
    assert display_aspect_ratio_from_probe(probe) == (720, 1280, 270, 16 / 9)


def test_load_item_csv_treats_empty_values_as_missing(tmp_path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text(
        "item_id,title\n1,title 1\n2,\n3,title 3\n",
        encoding="utf-8",
    )

    assert load_item_csv(path, "titles") == {"1": "title 1", "3": "title 3"}


def test_inventory_selection_and_sidecar_are_deterministic(tmp_path, monkeypatch) -> None:
    _prepare_sources(tmp_path)
    config = load_microlens_config(_write_config(tmp_path))
    ratios = {1: 4 / 3, 2: 16 / 9, 3: 2.0, 4: 16 / 9, 5: 1.2, 6: 2.1}

    def fake_probe(path: Path) -> dict[str, object]:
        item_id = int(path.stem)
        return {
            "width": 640,
            "height": 480,
            "rotation": 0,
            "display_aspect_ratio": ratios[item_id],
            "duration_seconds": 2.0,
            "file_size": path.stat().st_size,
            "source_mtime_ns": path.stat().st_mtime_ns,
        }

    processed: list[str] = []

    def fake_process_local_source(**kwargs):
        source = Path(kwargs["source_video_path"])
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        processed.append(kwargs["name"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        assert kwargs["frames_per_window"] == 3
        return {"content_id": kwargs["name"], "url": "unused"}

    monkeypatch.setattr(
        "src.video_data_collection.microlens_importer.probe_video", fake_probe
    )
    monkeypatch.setattr(
        "src.video_data_collection.microlens_importer.process_local_source",
        fake_process_local_source,
    )

    inventory, failures = build_inventory(config, project_root=tmp_path)
    assert [row["item_id"] for row in inventory if row["eligible"]] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert failures == []

    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (tmp_path / "videos").glob("*.mp4")
    }
    first = run_import(config, scope="smoke", project_root=tmp_path)
    selection_path = Path(first["selection"])
    selection_bytes = selection_path.read_bytes()
    second = run_import(config, scope="smoke", project_root=tmp_path)
    assert selection_path.read_bytes() == selection_bytes
    assert first["failed"] == second["failed"] == 0
    assert len(processed) == 6

    with Path(first["manifest"]).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["content_id", "url"]
        manifest_rows = list(reader)
    assert all(row["url"].startswith("microlens://100k/") for row in manifest_rows)
    sidecar = [
        json.loads(line)
        for line in Path(first["categories"]).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["category"] for row in sidecar} == {"Comedy", "Music", "News"}
    assert source_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (tmp_path / "videos").glob("*.mp4")
    }


def test_external_user_derived_selection_is_authoritative(tmp_path, monkeypatch) -> None:
    _prepare_sources(tmp_path)
    config_path = _write_config(tmp_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    selection = tmp_path / "vcp_selection.jsonl"
    smoke_selection = tmp_path / "vcp_smoke_selection.jsonl"
    rows = [
        {
            "item_id": str(item_id),
            "content_id": f"microlens_100k_{item_id:05d}",
            "source_video_path": str((tmp_path / "videos" / f"{item_id}.mp4").resolve()),
            "url": f"microlens://100k/{item_id}",
        }
        for item_id in (2, 5)
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    selection.write_text(content, encoding="utf-8")
    smoke_selection.write_text(content, encoding="utf-8")
    raw["source"]["selection_jsonl"] = str(selection)
    raw["source"]["smoke_selection_jsonl"] = str(smoke_selection)
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_microlens_config(config_path)

    monkeypatch.setattr(
        "src.video_data_collection.microlens_importer.probe_video",
        lambda path: {
            "width": 640, "height": 480, "rotation": 0,
            "display_aspect_ratio": 4 / 3, "duration_seconds": 3.0,
            "file_size": path.stat().st_size, "source_mtime_ns": path.stat().st_mtime_ns,
        },
    )
    records, failures, document = load_external_selection(
        config, scope="pilot", project_root=tmp_path
    )
    assert failures == []
    assert [row["item_id"] for row in records] == ["2", "5"]
    assert document["schema_version"] == "microlens-user-derived-selection/v1"
    assert document["selected_item_ids"] == ["2", "5"]


def test_keyframe_resize_uses_black_center_padding_without_stretch(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (400, 300), "red").save(frames / "0000.png")
    timestamps = tmp_path / "timestamps.json"
    timestamps.write_text(
        json.dumps([{"keyframe_timestamps": [0]}]), encoding="utf-8"
    )
    output = tmp_path / "resized"

    export_resized_keyframes(
        timestamps,
        frames,
        output,
        image_size=(160, 90),
        preserve_aspect_ratio=True,
    )

    with Image.open(output / "0000.png") as image:
        assert image.size == (160, 90)
        assert image.getpixel((0, 45)) == (0, 0, 0)
        assert image.getpixel((20, 45)) == (255, 0, 0)
        assert image.getpixel((80, 45)) == (255, 0, 0)
        assert image.getpixel((159, 45)) == (0, 0, 0)


def test_local_source_routes_original_video_to_direct_extraction(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "1.mp4"
    source.write_bytes(b"caller-owned-video")
    processing = tmp_path / "processing.json"
    processing.write_text(
        json.dumps(
            {
                "shot_interval": "fixed_15s",
                "asr_config": {"enabled": False},
                "ocr_config": {"enabled": False},
                "resized_keyframe_config": {"image_resolution": [160, 90]},
            }
        ),
        encoding="utf-8",
    )
    paths = build_content_paths(
        tmp_path / "source_assets",
        "microlens_100k_00001",
        output_root=tmp_path / "output",
        shot_interval="fixed_15s",
    )
    Path(paths.save_path).mkdir(parents=True)
    Path(paths.video_480p_path).write_bytes(b"stale-canonical")
    Path(paths.all_frames_dir).mkdir()
    (Path(paths.all_frames_dir) / "0000.png").write_bytes(b"stale-frame")
    captured = {}

    def fake_process_prepared_source(**kwargs):
        captured.update(kwargs)
        return {"content_id": kwargs["paths"].content_id, "url": kwargs["url"]}

    monkeypatch.setattr(
        "src.video_data_collection.raw_pipeline.process_prepared_source",
        fake_process_prepared_source,
    )

    process_local_source(
        name="microlens_100k_00001",
        source_video_path=source,
        data_root=tmp_path / "source_assets",
        output_root=tmp_path / "output",
        metadata={"title": "title 1"},
        config_path=processing,
    )

    assert captured["direct_video_path"] == source
    assert not Path(paths.video_480p_path).exists()
    assert not Path(paths.all_frames_dir).exists()
    assert source.read_bytes() == b"caller-owned-video"
    contract = json.loads(
        (Path(paths.assets_path) / "processing_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["schema_version"] == "local-video-processing-contract/v2"
    assert contract["frame_source"] == "original_video"
    assert contract["keyframe_resolution"] == [160, 90]


def test_direct_extraction_builds_fixed_timestamps_without_intermediate_frames(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "1.mp4"
    source.write_bytes(b"source-video")
    paths = build_content_paths(
        tmp_path / "source_assets",
        "microlens_100k_00001",
        url="microlens://100k/1",
        output_root=tmp_path / "output",
        shot_interval="fixed_15s",
    )
    Path(paths.assets_path).mkdir(parents=True)
    captured = {}

    monkeypatch.setattr(
        "src.video_data_collection.video_processor.get_video_duration_seconds",
        lambda _: 45.2,
    )

    def fake_extract(video_path, timestamps, output_folder, image_size):
        captured.update(
            video_path=video_path,
            timestamps=timestamps,
            image_size=image_size,
        )
        output = Path(output_folder)
        output.mkdir(parents=True)
        for timestamp in timestamps:
            Image.new("RGB", image_size, "red").save(output / f"{timestamp:04d}.png")

    monkeypatch.setattr(
        "src.video_data_collection.video_processor.extract_resized_keyframes",
        fake_extract,
    )

    result = process_prepared_source(
        paths=paths,
        config={
            "shot_interval": "fixed_15s",
            "asr_config": {"enabled": False},
            "ocr_config": {"enabled": False},
            "resized_keyframe_config": {"image_resolution": [160, 90]},
        },
        lang=None,
        url=paths.url,
        frames_per_window=1,
        direct_video_path=source,
    )

    assert result == {
        "content_id": "microlens_100k_00001",
        "url": "microlens://100k/1",
    }
    assert captured == {
        "video_path": str(source),
        "timestamps": [0, 15, 30, 45],
        "image_size": (160, 90),
    }
    assert not Path(paths.video_480p_path).exists()
    assert not Path(paths.all_frames_dir).exists()
    assert Path(paths.ref_jsonl).is_file()


@pytest.mark.parametrize("ocr_fails", [False, True])
def test_fixed_30s_local_ocr_frames_are_always_removed(
    tmp_path, monkeypatch, ocr_fails
) -> None:
    source = tmp_path / "1.mp4"
    source.write_bytes(b"source-video")
    paths = build_content_paths(
        tmp_path / "source_assets",
        "microlens_100k_00001",
        output_root=tmp_path / "output",
        shot_interval="fixed_30s",
    )
    Path(paths.assets_path).mkdir(parents=True)
    temp_ocr_frames = Path(paths.save_path) / ".ocr_frames_fixed_30s"

    monkeypatch.setattr(
        "src.video_data_collection.video_processor.get_video_duration_seconds",
        lambda _: 30.0,
    )

    def fake_extract_keyframes(_, timestamps, output_folder, image_size):
        output = Path(output_folder)
        output.mkdir(parents=True)
        for timestamp in timestamps:
            Image.new("RGB", image_size, "red").save(output / f"{timestamp:04d}.png")

    def fake_extract_frames(_, output_folder):
        output = Path(output_folder)
        output.mkdir(parents=True)
        (output / "0000.png").write_bytes(b"frame")

    def fake_process_ocr(frame_dir, output_path, **_):
        assert Path(frame_dir) == temp_ocr_frames
        assert temp_ocr_frames.is_dir()
        if ocr_fails:
            raise RuntimeError("ocr failed")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        "src.video_data_collection.video_processor.extract_resized_keyframes",
        fake_extract_keyframes,
    )
    monkeypatch.setattr(
        "src.video_data_collection.video_processor.extract_frames",
        fake_extract_frames,
    )
    monkeypatch.setattr(
        "src.video_data_collection.ocr_processor.process_ocr",
        fake_process_ocr,
    )
    monkeypatch.setattr(
        "src.video_data_collection.utils.patch_paddlex_predictor", lambda: None
    )
    monkeypatch.setattr(
        "src.video_data_collection.data_processor.intervalize_ocr",
        lambda _, output_path: Path(output_path).write_text("[]", encoding="utf-8"),
    )

    kwargs = {
        "paths": paths,
        "config": {
            "shot_interval": "fixed_30s",
            "asr_config": {"enabled": False},
            "ocr_config": {"enabled": True},
            "resized_keyframe_config": {"image_resolution": [160, 90]},
        },
        "lang": None,
        "url": paths.url,
        "direct_video_path": source,
    }
    if ocr_fails:
        with pytest.raises(RuntimeError, match="ocr failed"):
            process_prepared_source(**kwargs)
    else:
        process_prepared_source(**kwargs)

    assert not temp_ocr_frames.exists()


def _make_video(path: Path, size: str, color: str = "red") -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:d=1.2:r=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="synthetic video smoke requires ffmpeg and ffprobe",
)
def test_synthetic_video_smoke_runs_local_pipeline_without_mutating_sources(
    tmp_path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for item_id, size in {
        1: "320x240",
        2: "320x180",
        3: "400x200",
        4: "320x180",
        5: "300x250",
        6: "420x200",
    }.items():
        _make_video(videos / f"{item_id}.mp4", size)

    rotation_base = tmp_path / "rotation-base.mp4"
    _make_video(rotation_base, "180x320", color="blue")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(rotation_base),
            "-c",
            "copy",
            "-y",
            str(videos / "4.mp4"),
        ],
        check=True,
    )
    _prepare_sources(tmp_path, create_videos=False)
    processing = {
        "shot_interval": "fixed_30s",
        "asr_config": {"enabled": False},
        "ocr_config": {"enabled": False},
        "resized_keyframe_config": {"image_resolution": [160, 90]},
    }
    (tmp_path / "processing.json").write_text(
        json.dumps(processing), encoding="utf-8"
    )
    config = load_microlens_config(_write_config(tmp_path))
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in videos.glob("*.mp4")
    }

    result = run_import(config, scope="pilot", project_root=tmp_path)

    assert result["failed"] == 0
    inventory = [
        json.loads(line)
        for line in (
            tmp_path
            / "output/microlens/manifests/microlens_100k_inventory.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["item_id"] for row in inventory if row["eligible"]] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert next(row for row in inventory if row["item_id"] == "4")["rotation"] in {
        90,
        270,
    }
    canonical = (
        tmp_path
        / "output/microlens/source_assets/microlens_100k_00001"
        / "microlens_100k_00001_480p.mp4"
    )
    assert not canonical.exists()
    assert not canonical.with_name("all_frames").exists()
    keyframe = (
        tmp_path
        / "output/microlens/asset/fixed_30s/resized_keyframes"
        / "microlens_100k_00001/0001.png"
    )
    with Image.open(keyframe) as frame:
        frame = frame.convert("RGB")
        assert frame.size == (160, 90)
        assert max(frame.getpixel((5, 45))) < 20
        assert frame.getpixel((80, 45))[0] > 200
    rotated_keyframe = keyframe.parents[0].parent / "microlens_100k_00004/0001.png"
    with Image.open(rotated_keyframe) as frame:
        frame = frame.convert("RGB")
        assert frame.size == (160, 90)
        assert frame.getpixel((5, 45))[2] > 200
    contract = json.loads(
        (
            tmp_path
            / "output/microlens/source_assets/microlens_100k_00001/assets"
            / "processing_contract_fixed_30s.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["schema_version"] == "local-video-processing-contract/v3"
    assert contract["frame_source"] == "original_video"
    assert contract["keyframe_resolution"] == [160, 90]
    assert contract["sampling"]["scene_seconds"] == 30
    assert contract["sampling"]["reference_seconds"] == 10
    assert contract["keyframe_offsets_seconds"] == [5, 15, 25]
    assert contract["ocr_sampling_fps"] == 1
    assert contract["ocr_max_chars"] == 1000
    timestamp_path = (
        tmp_path
        / "output/microlens/source_assets/microlens_100k_00001/assets"
        / "timestamp_fixed_30s.json"
    )
    timestamps = json.loads(timestamp_path.read_text(encoding="utf-8"))
    assert [scene["keyframe_timestamps"] for scene in timestamps] == [[1]]
    assert (
        tmp_path
        / "output/microlens/asset/fixed_30s/ref_jsonl"
        / "microlens_100k_00001_ref.jsonl"
    ).is_file()
    assert source_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in videos.glob("*.mp4")
    }
