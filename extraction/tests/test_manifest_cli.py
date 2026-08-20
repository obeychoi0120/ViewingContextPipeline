from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.video_data_collection.cli import (
    DEFAULT_MANIFEST_PATH,
    parse_args,
    indexed_content_id,
    inspect_output,
    manifest_path,
    manifest_rows_from_txt,
    metadata_json_path_for_row,
    read_txt_video_list,
    resolve_assets_root,
    resolve_output_root,
    run_inspect_output,
    run_create_manifest,
    run_get_metadata,
)
from src.video_data_collection.raw_pipeline import (
    build_fixed_interval_windows,
    build_content_paths,
    collect_video_metadata,
    export_resized_keyframes,
    is_nonempty_file,
    manifest_row,
    metadata_from_row,
    multimodal_from_config,
    shot_interval_from_config,
    write_manifest,
)


class ManifestCliTests(unittest.TestCase):
    def test_multimodal_config_is_required_and_boolean(self):
        self.assertFalse(multimodal_from_config({"multimodal": False}))
        self.assertTrue(multimodal_from_config({"multimodal": True}))
        with self.assertRaisesRegex(ValueError, "multimodal must be explicitly set"):
            multimodal_from_config({})
        with self.assertRaisesRegex(ValueError, "multimodal must be a boolean"):
            multimodal_from_config({"multimodal": "true"})

    def test_collected_metadata_preserves_multiline_description(self):
        description = "첫 번째 문단입니다.\n\n두 번째 문단입니다."
        with mock.patch("yt_dlp.YoutubeDL") as youtube_dl:
            youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
                "title": "title",
                "description": description,
            }

            metadata = collect_video_metadata("https://www.youtube.com/watch?v=demo", "fallback")

        self.assertEqual(metadata["description"], description)

    def test_collected_metadata_uses_empty_title_when_unavailable(self):
        with mock.patch("yt_dlp.YoutubeDL") as youtube_dl:
            youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {}

            metadata = collect_video_metadata("https://www.youtube.com/watch?v=demo", "content_id")

        self.assertEqual(metadata["title"], "")

    def test_manifest_metadata_normalizes_missing_description(self):
        metadata = metadata_from_row({}, "https://www.youtube.com/watch?v=demo", "fallback")

        self.assertEqual(metadata["title"], "")
        self.assertEqual(metadata["description"], "")

    def test_reference_asr_cache_requires_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            asr_ref = Path(tmp) / "ASR_Ref.json"
            self.assertFalse(is_nonempty_file(asr_ref))
            asr_ref.write_text("", encoding="utf-8")
            self.assertFalse(is_nonempty_file(asr_ref))
            asr_ref.write_text("[]", encoding="utf-8")
            self.assertTrue(is_nonempty_file(asr_ref))

    def test_txt_content_id_combines_list_id_and_youtube_video_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "VideoList_manual.txt"
            data_root = root / "data"
            txt_path.write_text(
                "[News]\n"
                "News_Manual_001 https://www.youtube.com/watch?v=VV3qIkq5ofY\n",
                encoding="utf-8",
            )

            rows = manifest_rows_from_txt(txt_path, data_root, config={}, output_root=Path("output"))

            self.assertEqual(
                rows,
                [{
                    "content_id": "News_Manual_001_VV3qIkq5ofY",
                    "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                }],
            )

    def test_output_root_does_not_add_paths_to_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "VideoList_manual.txt"
            txt_path.write_text(
                "[News]\n"
                "News_Manual_001 https://www.youtube.com/watch?v=VV3qIkq5ofY\n",
                encoding="utf-8",
            )

            rows = manifest_rows_from_txt(
                txt_path,
                root / "assets",
                config={},
                output_root=root / "pipeline_output",
            )

            self.assertEqual(
                set(rows[0]),
                {"content_id", "url"},
            )

    def test_manifest_has_only_identity_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt_path = Path(tmp) / "VideoList_manual.txt"
            txt_path.write_text(
                "[News]\n"
                "News_Manual_001 https://www.youtube.com/watch?v=VV3qIkq5ofY\n",
                encoding="utf-8",
            )

            rows = manifest_rows_from_txt(
                txt_path,
                "/home_nvme/shared/data/YoutubeVideoDataset/00_Junsu",
                config={},
                output_root="output",
            )
            row = rows[0]
            self.assertEqual(
                row,
                {
                    "content_id": "News_Manual_001_VV3qIkq5ofY",
                    "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                },
            )

    def test_resolve_roots_use_new_environment_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(
                os.environ,
                {
                    "LINUX_ASSETS_SAVE_PATH": str(root / "linux_assets"),
                    "OUTPUT_SAVE_PATH": str(root / "pipeline_output"),
                },
                clear=False,
            ):
                self.assertEqual(resolve_assets_root(), str(root / "linux_assets"))
                output_root = resolve_output_root()

                self.assertEqual(output_root, root / "pipeline_output" / "custom")
            self.assertEqual(manifest_path(), DEFAULT_MANIFEST_PATH)

    def test_roots_require_new_environment_names(self):
        with mock.patch.dict(os.environ, {"LINUX_ASSETS_SAVE_PATH": "", "OUTPUT_SAVE_PATH": ""}, clear=False):
            with self.assertRaises(ValueError):
                resolve_assets_root()
            with self.assertRaises(ValueError):
                resolve_output_root()

    def test_create_manifest_uses_environment_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "VideoList_manual.txt"
            txt_path.write_text(
                "[News]\n"
                "News_Manual_001 https://www.youtube.com/watch?v=VV3qIkq5ofY\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                search=False,
                list_file_path=str(txt_path),
                output="manifest.csv",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "LINUX_ASSETS_SAVE_PATH": str(root / "linux_assets"),
                    "OUTPUT_SAVE_PATH": str(root / "pipeline_output"),
                },
                clear=False,
            ):
                rows, manifest_output = run_create_manifest(args)

            self.assertEqual(
                manifest_output,
                root / "pipeline_output" / "custom" / "manifest.csv",
            )
            self.assertTrue(manifest_output.exists())
            self.assertEqual(set(rows[0]), {"content_id", "url"})

    def test_create_manifest_output_name_writes_under_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt_path = root / "VideoList_manual.txt"
            txt_path.write_text(
                "[News]\n"
                "News_Manual_001 https://www.youtube.com/watch?v=VV3qIkq5ofY\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                search=False,
                list_file_path=str(txt_path),
                output="manifest_manual.csv",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "LINUX_ASSETS_SAVE_PATH": str(root / "linux_assets"),
                    "OUTPUT_SAVE_PATH": str(root / "pipeline_output"),
                },
                clear=False,
            ):
                rows, manifest_output = run_create_manifest(args)

            self.assertEqual(
                manifest_output,
                root / "pipeline_output" / "custom" / "manifest_manual.csv",
            )
            self.assertTrue(manifest_output.exists())
            self.assertEqual(len(rows), 1)

    def test_txt_rejects_list_id_that_does_not_match_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt_path = Path(tmp) / "VideoList_manual.txt"
            txt_path.write_text(
                "[News]\n"
                "Tech_Manual_001 https://www.youtube.com/watch?v=TJ7YMKDR4D0\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                read_txt_video_list(txt_path)

    def test_txt_accepts_manual_and_auto_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt_path = Path(tmp) / "video_list.txt"
            txt_path.write_text(
                "[News]\n"
                "News_Manual_001 https://www.youtube.com/watch?v=manual001\n"
                "News_Auto_001 https://www.youtube.com/watch?v=auto001\n",
                encoding="utf-8",
            )

            self.assertEqual(
                read_txt_video_list(txt_path),
                [
                    ("News_Manual_001", "https://www.youtube.com/watch?v=manual001"),
                    ("News_Auto_001", "https://www.youtube.com/watch?v=auto001"),
                ],
            )

    def test_txt_rejects_legacy_list_id_formats(self):
        legacy_lines = [
            "News001 https://www.youtube.com/watch?v=legacy001",
            "News_M_001 https://www.youtube.com/watch?v=legacy001",
            "News_S_001 https://www.youtube.com/watch?v=legacy001",
        ]
        for line in legacy_lines:
            with self.subTest(line=line):
                with tempfile.TemporaryDirectory() as tmp:
                    txt_path = Path(tmp) / "video_list.txt"
                    txt_path.write_text(f"[News]\n{line}\n", encoding="utf-8")

                    with self.assertRaises(ValueError):
                        read_txt_video_list(txt_path)

    def test_process_batch_preserves_explicit_content_id(self):
        row = {
            "content_id": "News_Manual_001_VV3qIkq5ofY",
            "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
        }

        self.assertEqual(indexed_content_id(row, 1), "News_Manual_001_VV3qIkq5ofY")

    def test_process_batch_and_get_metadata_default_to_canonical_manifest(self):
        for command in ("process-batch", "get-metadata"):
            with self.subTest(command=command), mock.patch("sys.argv", ["cli", command]):
                args = parse_args()

            self.assertEqual(Path(args.manifest), DEFAULT_MANIFEST_PATH)

    def test_inspect_output_reports_missing_mismatches_and_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "asset" / "metadata"
            ref_dir = root / "asset" / "fixed_15s" / "ref_jsonl"
            resized_dir = root / "asset" / "fixed_15s" / "resized_keyframes"
            metadata_dir.mkdir(parents=True)
            ref_dir.mkdir(parents=True)
            resized_dir.mkdir(parents=True)
            for content_id in ("complete", "missing_frames", "missing_ref", "mismatch", "invalid_ref"):
                (metadata_dir / f"{content_id}.json").write_text("{}", encoding="utf-8")

            self._write_ref(ref_dir / "complete_ref.jsonl", [0, 3.2, 3])
            complete_frames = resized_dir / "complete"
            complete_frames.mkdir()
            for name in ("0000.png", "0003.png"):
                (complete_frames / name).write_bytes(b"png")

            self._write_ref(ref_dir / "missing_frames_ref.jsonl", [0])
            (resized_dir / "missing_frames").mkdir()
            missing_ref_frames = resized_dir / "missing_ref"
            missing_ref_frames.mkdir()
            (missing_ref_frames / "0000.png").write_bytes(b"png")

            self._write_ref(ref_dir / "mismatch_ref.jsonl", [0, 4])
            mismatch_frames = resized_dir / "mismatch"
            mismatch_frames.mkdir()
            for name in ("0000.png", "0005.png"):
                (mismatch_frames / name).write_bytes(b"png")

            (ref_dir / "invalid_ref_ref.jsonl").write_text("{invalid}\n", encoding="utf-8")
            invalid_frames = resized_dir / "invalid_ref"
            invalid_frames.mkdir()
            (invalid_frames / "0000.png").write_bytes(b"png")

            (resized_dir / "orphan_frames").mkdir()
            (ref_dir / "orphan_ref_ref.jsonl").write_text("{}\n", encoding="utf-8")

            report = inspect_output(root, shot_interval="fixed_15s")

        self.assertEqual(
            {item["content_id"] for item in report["missing"]},
            {"missing_frames", "missing_ref"},
        )
        mismatch = next(item for item in report["mismatches"] if item["content_id"] == "mismatch")
        self.assertEqual(mismatch["missing_png"], ["0004.png"])
        self.assertEqual(mismatch["extra_png"], ["0005.png"])
        self.assertTrue(next(item for item in report["mismatches"] if item["content_id"] == "invalid_ref")["error"])
        self.assertEqual(report["cleanup_candidates"], ["missing_frames"])
        self.assertEqual(report["orphan_resized"], ["orphan_frames"])
        self.assertEqual(report["orphan_ref"], ["orphan_ref"])

    def test_inspect_output_cleanup_requires_confirmation(self):
        for answer, should_delete in (("n", False), ("invalid", False), ("y", True)):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "asset" / "metadata").mkdir(parents=True)
                (root / "asset" / "fixed_30s" / "ref_jsonl").mkdir(parents=True)
                metadata_path = root / "asset" / "metadata" / "missing.json"
                ref_path = root / "asset" / "fixed_30s" / "ref_jsonl" / "missing_ref.jsonl"
                metadata_path.write_text("{}", encoding="utf-8")
                ref_path.write_text("{}\n", encoding="utf-8")
                answers = [answer, "n"] if answer == "invalid" else [answer]
                with mock.patch("builtins.input", side_effect=answers), mock.patch("builtins.print"):
                    report = run_inspect_output(root)

                self.assertEqual(metadata_path.exists(), not should_delete)
                self.assertEqual(ref_path.exists(), not should_delete)
                self.assertEqual(report["deleted"], ["missing"] if should_delete else [])

    def test_inspect_output_does_not_prompt_without_cleanup_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asset" / "metadata").mkdir(parents=True)
            (root / "asset" / "fixed_30s" / "ref_jsonl").mkdir(parents=True)
            frames = root / "asset" / "fixed_30s" / "resized_keyframes" / "complete"
            frames.mkdir(parents=True)
            (root / "asset" / "metadata" / "complete.json").write_text("{}", encoding="utf-8")
            self._write_ref(root / "asset" / "fixed_30s" / "ref_jsonl" / "complete_ref.jsonl", [0])
            (frames / "0000.png").write_bytes(b"png")

            with mock.patch("builtins.input") as user_input, mock.patch("builtins.print"):
                run_inspect_output(root)

            user_input.assert_not_called()

    @staticmethod
    def _write_ref(path: Path, timestamps: list[float]) -> None:
        records = [
            {"_type": "video_metadata", "title": "ignored"},
            {"scene_idx": 0, "timeline": [{"timestamp": timestamp} for timestamp in timestamps]},
        ]
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    def test_get_metadata_writes_skips_and_forces_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale_metadata_json = root / "pipeline_output" / "custom" / "asset" / "metadata" / "VV3qIkq5ofY.json"
            metadata_json = root / "pipeline_output" / "custom" / "asset" / "metadata" / "News_Manual_001_VV3qIkq5ofY.json"
            output_root = root / "pipeline_output"
            manifest = output_root / "manifest.csv"
            manifest.parent.mkdir(parents=True)
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["content_id", "url"])
                writer.writeheader()
                writer.writerow(
                    {
                        "content_id": "News_Manual_001_VV3qIkq5ofY",
                        "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                    }
                )

            args = argparse.Namespace(
                force=False,
                manifest=str(manifest),
            )
            with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": str(output_root)}, clear=False), mock.patch(
                "src.video_data_collection.cli.collect_url_metadata",
                return_value={
                    "title": "first",
                    "channel": "news",
                    "description": "첫 문단\n\n둘째 문단",
                    "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                },
            ):
                result = run_get_metadata(args)

            self.assertEqual(result, {"written": 1, "skipped": 0})
            written_metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(written_metadata["title"], "first")
            self.assertEqual(written_metadata["description"], "첫 문단\n\n둘째 문단")
            self.assertFalse(stale_metadata_json.exists())

            with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": str(output_root)}, clear=False), mock.patch(
                "src.video_data_collection.cli.collect_url_metadata",
                return_value={"title": "second", "channel": "news", "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY"},
            ):
                result = run_get_metadata(args)

            self.assertEqual(result, {"written": 0, "skipped": 1})
            self.assertEqual(json.loads(metadata_json.read_text(encoding="utf-8"))["title"], "first")

            args.force = True
            with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": str(output_root)}, clear=False), mock.patch(
                "src.video_data_collection.cli.collect_url_metadata",
                return_value={"title": "second", "channel": "news", "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY"},
            ):
                result = run_get_metadata(args)

            self.assertEqual(result, {"written": 1, "skipped": 0})
            self.assertEqual(json.loads(metadata_json.read_text(encoding="utf-8"))["title"], "second")

    def test_get_metadata_default_path_uses_output_metadata(self):
        path = metadata_json_path_for_row(
            row={
                "content_id": "News_Manual_001_VV3qIkq5ofY",
                "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                "metadata_json": "output/asset/metadata/VV3qIkq5ofY.json",
            },
            content_id="News_Manual_001_VV3qIkq5ofY",
            url="https://www.youtube.com/watch?v=VV3qIkq5ofY",
            config={},
            output_root=Path("output"),
        )

        self.assertEqual(path, Path("output/asset/metadata/News_Manual_001_VV3qIkq5ofY.json"))

    def test_get_metadata_default_path_does_not_require_assets_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "pipeline_output"
            with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": str(output_root)}, clear=False):
                path = metadata_json_path_for_row(
                    row={
                        "content_id": "News_Manual_001_VV3qIkq5ofY",
                        "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                    },
                    content_id="News_Manual_001_VV3qIkq5ofY",
                    url="https://www.youtube.com/watch?v=VV3qIkq5ofY",
                    config={},
                )

            self.assertEqual(
                path,
                output_root
                / "custom"
                / "asset"
                / "metadata"
                / "News_Manual_001_VV3qIkq5ofY.json",
            )

    def test_manifest_row_has_only_content_id_and_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_content_paths(
                tmp,
                "News_Manual_001_VV3qIkq5ofY",
                url="https://www.youtube.com/watch?v=VV3qIkq5ofY",
                output_root=Path("output"),
            )

            row = manifest_row(paths)

            self.assertEqual(
                row,
                {
                    "content_id": "News_Manual_001_VV3qIkq5ofY",
                    "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                },
            )

    def test_shot_interval_selects_mode_specific_keyframe_and_timestamp_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode, directory, timestamp_name in (
                ("shot_wise", "shot_wise/resized_keyframes", "timestamp_filtered.json"),
                ("fixed_15s", "fixed_15s/resized_keyframes", "timestamp_fixed_15s.json"),
                ("fixed_30s", "fixed_30s/resized_keyframes", "timestamp_fixed_30s.json"),
            ):
                with self.subTest(mode=mode):
                    paths = build_content_paths(
                        tmp,
                        "content",
                        output_root=Path("output"),
                        shot_interval=mode,
                    )

                    self.assertEqual(
                        Path(paths.resized_keyframes_dir),
                        Path("output") / "asset" / directory / "content",
                    )
                    self.assertEqual(
                        Path(paths.ref_jsonl),
                        Path("output") / "asset" / mode / "ref_jsonl" / "content_ref.jsonl",
                    )
                    self.assertEqual(Path(paths.filtered_timestamp_json).name, timestamp_name)

    def test_fixed_30s_is_default_and_invalid_shot_interval_is_rejected(self):
        self.assertEqual(shot_interval_from_config({}), "fixed_30s")
        with self.assertRaisesRegex(ValueError, "shot_interval must be one of"):
            build_content_paths("data", "content", shot_interval="fixed_10s")

    def test_fixed_interval_windows_use_four_15_second_frames_and_partial_tail(self):
        self.assertEqual(
            build_fixed_interval_windows(138),
            [
                {
                    "scene_start": 0,
                    "scene_end": 60,
                    "duration": 60,
                    "shot_change_timestamps": [0, 15, 30, 45],
                    "keyframe_timestamps": [0, 15, 30, 45],
                },
                {
                    "scene_start": 60,
                    "scene_end": 120,
                    "duration": 60,
                    "shot_change_timestamps": [60, 75, 90, 105],
                    "keyframe_timestamps": [60, 75, 90, 105],
                },
                {
                    "scene_start": 120,
                    "scene_end": 138,
                    "duration": 18,
                    "shot_change_timestamps": [120, 135],
                    "keyframe_timestamps": [120, 135],
                },
            ],
        )

    def test_fixed_interval_windows_support_one_frame_per_microlens_scene(self):
        self.assertEqual(
            build_fixed_interval_windows(46, frames_per_window=1),
            [
                {
                    "scene_start": timestamp,
                    "scene_end": min(timestamp + 15, 46),
                    "duration": min(timestamp + 15, 46) - timestamp,
                    "shot_change_timestamps": [timestamp],
                    "keyframe_timestamps": [timestamp],
                }
                for timestamp in (0, 15, 30, 45)
            ],
        )

    def test_fixed_30s_windows_use_three_midpoint_frames_without_overlap(self):
        self.assertEqual(
            build_fixed_interval_windows(
                90,
                frames_per_window=3,
                shot_interval="fixed_30s",
            ),
            [
                {
                    "scene_start": start,
                    "scene_end": start + 30,
                    "duration": 30,
                    "shot_change_timestamps": [start, start + 10, start + 20],
                    "keyframe_timestamps": [start + 5, start + 15, start + 25],
                }
                for start in (0, 30, 60)
            ],
        )

    def test_fixed_30s_windows_clip_partial_tail_midpoints(self):
        self.assertEqual(
            build_fixed_interval_windows(
                65,
                frames_per_window=3,
                shot_interval="fixed_30s",
            )[-1],
            {
                "scene_start": 60,
                "scene_end": 65,
                "duration": 5,
                "shot_change_timestamps": [60],
                "keyframe_timestamps": [62],
            },
        )
        self.assertEqual(
            build_fixed_interval_windows(
                3,
                frames_per_window=3,
                shot_interval="fixed_30s",
            )[0]["keyframe_timestamps"],
            [1],
        )

    def test_write_manifest_removes_legacy_path_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "manifest.csv"
            write_manifest(
                [{
                    "content_id": "content",
                    "url": "https://example.com/video",
                    "metadata_json": "output/asset/metadata/content.json",
                    "ref_jsonl": "output/ref_jsonl/content_ref.jsonl",
                }],
                output_path,
            )

            with output_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, ["content_id", "url"])
            self.assertEqual(rows, [{"content_id": "content", "url": "https://example.com/video"}])

    def test_export_resized_keyframes_writes_timestamp_names_and_cleans_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timestamp_file = root / "timestamp_filtered.json"
            source_dir = root / "all_frames"
            output_dir = root / "resized_keyframes"
            source_dir.mkdir()
            output_dir.mkdir()
            timestamp_file.write_text(
                json.dumps([{"keyframe_timestamps": [0, 3.2]}, {"keyframe_timestamps": [22, 3]}]),
                encoding="utf-8",
            )
            for timestamp in (0, 3, 22):
                Image.new("RGB", (854, 480), color=(10, 20, 30)).save(source_dir / f"{timestamp:04d}.png")
            Image.new("RGB", (10, 10)).save(output_dir / "0099.png")
            Image.new("RGB", (10, 10)).save(output_dir / "Scene001_Shot01_0000.png")

            export_resized_keyframes(timestamp_file, source_dir, output_dir, (672, 384))

            self.assertEqual(sorted(path.name for path in output_dir.glob("*.png")), ["0000.png", "0003.png", "0022.png"])
            for timestamp in (0, 3, 22):
                with Image.open(output_dir / f"{timestamp:04d}.png") as image:
                    self.assertEqual(image.size, (672, 384))

    def test_export_resized_keyframes_fails_before_writing_when_all_frame_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timestamp_file = root / "timestamp_filtered.json"
            source_dir = root / "all_frames"
            output_dir = root / "resized_keyframes"
            source_dir.mkdir()
            timestamp_file.write_text(
                json.dumps([{"keyframe_timestamps": [0, 3]}]),
                encoding="utf-8",
            )
            Image.new("RGB", (854, 480)).save(source_dir / "0000.png")

            with self.assertRaisesRegex(FileNotFoundError, "0003.png"):
                export_resized_keyframes(timestamp_file, source_dir, output_dir, (672, 384))

            self.assertFalse(output_dir.exists())

if __name__ == "__main__":
    unittest.main()
