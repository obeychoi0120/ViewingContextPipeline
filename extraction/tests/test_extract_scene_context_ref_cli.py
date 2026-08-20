from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.common.manifest import CANONICAL_MANIFEST_PATH
from src.scene_context_extraction.ref.cli import (
    load_context_rows,
    load_settings_config,
    parse_args,
    print_startup_config,
)
from src.scene_context_extraction.ref.extractor import SceneContextRefConfig


class ExtractSceneContextRefCliTests(unittest.TestCase):
    def test_parse_args_defaults_to_canonical_manifest(self) -> None:
        with mock.patch.object(sys, "argv", ["cli"]):
            args = parse_args()

        self.assertEqual(args.manifest, CANONICAL_MANIFEST_PATH)
        self.assertEqual(
            args.settings,
            Path(__file__).resolve().parents[1]
            / "config"
            / "scene_context_extraction_ref.json",
        )
        self.assertFalse(args.force)

    def test_force_is_explicit_opt_in(self) -> None:
        with mock.patch.object(sys, "argv", ["cli", "--force"]):
            args = parse_args()

        self.assertTrue(args.force)

    def test_print_startup_config_shows_local_ref_context_counts(self) -> None:
        args = argparse.Namespace(
            settings="config/scene_context_extraction_ref.json",
            manifest=CANONICAL_MANIFEST_PATH,
            training_output=None,
            force=False,
            max_scenes=10,
            sleep_sec=0.5,
        )
        config = SceneContextRefConfig(
            gcp_project_id="project",
            output_dir="output",
            location="global",
            model="gemini-test",
            thinking_level="medium",
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            print_startup_config(
                args=args,
                config=config,
                ref_jsonl_file_count=2,
                content_count=2,
                scene_count=3,
                scene_context_output_dir=Path("output/viewing_context/img_only/fixed_15s/scene_context_ref"),
                video_context_output_dir=Path("output/viewing_context/img_only/fixed_15s/video_context_graph_ref"),
            )

        text = stdout.getvalue()
        self.assertIn(str((Path("output") / "asset" / "fixed_15s" / "ref_jsonl").resolve()), text)
        self.assertNotIn("runtime_manifest", text)
        self.assertIn("ref_jsonl_files: 2", text)
        self.assertIn("contents: 2", text)
        self.assertIn("scenes: 3", text)
        self.assertIn("gcp_project_id: project", text)
        self.assertIn("video_context_graph_ref_dir: output", text)
        self.assertIn("resume: True", text)
        self.assertIn("force: False", text)
        self.assertIn("max_scenes: 10", text)

    def test_load_settings_config_uses_output_save_path_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "pipeline_output"
            args = argparse.Namespace(
                settings=str(Path(tmp) / "settings.json"),
                gcp_project_id="project",
                output_dir=None,
                location=None,
                model=None,
                thinking_level=None,
            )
            Path(args.settings).write_text('{"multimodal": false}', encoding="utf-8")
            with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": str(output_root)}, clear=False):
                config = load_settings_config(args)

        self.assertEqual(config.output_dir, str(output_root / "custom"))

    def test_load_settings_config_uses_gemini_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.json"
            settings.write_text(
                json.dumps(
                    {
                        "multimodal": False,
                        "shot_interval": "shot_wise",
                        "gemini_location": "asia-northeast3",
                        "gemini_model": "gemini-test",
                        "gemini_thinking_level": "high",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                settings=str(settings),
                gcp_project_id="project",
                output_dir=None,
                location=None,
                model=None,
                thinking_level=None,
            )
            with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": "output"}, clear=False):
                config = load_settings_config(args)

        self.assertEqual(config.location, "asia-northeast3")
        self.assertEqual(config.model, "gemini-test")
        self.assertEqual(config.thinking_level, "high")
        self.assertEqual(config.shot_interval, "shot_wise")

    def test_load_settings_config_uses_cloud_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.json"
            settings.write_text(
                '{"multimodal": false, "location": "global"}',
                encoding="utf-8",
            )
            args = argparse.Namespace(
                settings=str(settings),
                gcp_project_id=None,
                output_dir=None,
                location=None,
                model=None,
                thinking_level=None,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "GCP_PROJECT_ID": "env-project",
                    "OUTPUT_SAVE_PATH": "output",
                },
                clear=False,
            ):
                config = load_settings_config(args)

        self.assertEqual(config.gcp_project_id, "env-project")
        self.assertFalse(hasattr(config, "gs_bucket_name"))

    def test_load_settings_config_rejects_legacy_cloud_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.json"
            settings.write_text('{"gcp_project_id": "legacy"}', encoding="utf-8")
            args = argparse.Namespace(
                settings=str(settings),
                gcp_project_id=None,
                output_dir="output",
                location=None,
                model=None,
                thinking_level=None,
            )

            with self.assertRaisesRegex(RuntimeError, "config/.env"):
                load_settings_config(args)

    def test_load_settings_config_rejects_json_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.json"
            settings.write_text('{"output_dir": "legacy-output"}', encoding="utf-8")
            args = argparse.Namespace(
                settings=str(settings),
                gcp_project_id="project",
                output_dir=None,
                location=None,
                model=None,
                thinking_level=None,
            )

            with self.assertRaisesRegex(RuntimeError, "OUTPUT_SAVE_PATH"):
                load_settings_config(args)

    def test_load_context_rows_uses_local_ref_jsonl_directory(self) -> None:
        record = {
            "scene_idx": 0,
            "timeline": [
                {"shot_idx": 0, "timestamp": 0, "raw_asr": "", "raw_ocr": ""},
                {"shot_idx": 1, "timestamp": 3, "raw_asr": "", "raw_ocr": ""},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp) / "asset" / "fixed_15s" / "ref_jsonl"
            ref_dir.mkdir(parents=True)
            (ref_dir / "video-1_ref.jsonl").write_text(json.dumps(record), encoding="utf-8")
            (ref_dir / "stale_ref.jsonl").write_text(json.dumps(record), encoding="utf-8")
            manifest_path = Path(tmp) / "manifest.csv"
            manifest_path.write_text(
                "content_id,url\nvideo-1,https://example.com/video-1\n",
                encoding="utf-8",
            )
            config = SceneContextRefConfig(
                gcp_project_id="project",
                output_dir=tmp,
            )

            rows, file_count = load_context_rows(config, manifest_path)
            manifest_exists = (Path(tmp) / "scene_context_ref_runtime_manifest.jsonl").exists()

        self.assertEqual(file_count, 1)
        self.assertFalse(manifest_exists)
        self.assertEqual(
            rows[0],
            {"content_id": "video-1", "scene_idx": 0, "keyframes": [0, 3]},
        )

    def test_load_context_rows_uses_configured_shot_wise_directory(self) -> None:
        record = {"scene_idx": 0, "timeline": [{"timestamp": 7}]}
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp) / "asset" / "shot_wise" / "ref_jsonl"
            ref_dir.mkdir(parents=True)
            (ref_dir / "video-1_ref.jsonl").write_text(
                json.dumps(record),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "manifest.csv"
            manifest_path.write_text(
                "content_id,url\nvideo-1,https://example.com/video-1\n",
                encoding="utf-8",
            )
            config = SceneContextRefConfig(
                gcp_project_id="project",
                output_dir=tmp,
                shot_interval="shot_wise",
            )

            rows, file_count = load_context_rows(config, manifest_path)

        self.assertEqual(file_count, 1)
        self.assertEqual(rows[0]["keyframes"], [7])


if __name__ == "__main__":
    unittest.main()
