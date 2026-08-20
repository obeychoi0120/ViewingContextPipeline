from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.common.manifest import CANONICAL_MANIFEST_PATH
from src.scene_context_extraction.ondevice.cli import (
    SCENE_CONTEXT_ONDEVICE_CONFIG_PATH,
    parse_args,
    scene_progress,
    update_generation_progress,
    update_scene_progress,
)
from src.scene_context_extraction.ondevice.pipeline import (
    SceneContextJob,
    infer_scene_context_jsonl,
    manifest_row_to_job,
    ondevice_failure_path,
    ondevice_context_path,
    resolve_frames_dir,
)


class OnDeviceSceneContextPathTests(unittest.TestCase):
    def test_cli_defaults_to_canonical_manifest(self) -> None:
        with mock.patch(
            "sys.argv",
            ["cli"],
        ):
            args = parse_args()

        self.assertEqual(Path(args.manifest), CANONICAL_MANIFEST_PATH)
        self.assertEqual(args.settings, SCENE_CONTEXT_ONDEVICE_CONFIG_PATH)
        self.assertFalse(args.force)

    def test_cli_accepts_force(self) -> None:
        with mock.patch(
            "sys.argv",
            ["cli", "--force"],
        ):
            args = parse_args()

        self.assertTrue(args.force)

    def test_progress_uses_total_scene_count_without_failure_or_warning_postfix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_ref = root / "first.jsonl"
            second_ref = root / "second.jsonl"
            first_ref.write_text("{}\n{}\n", encoding="utf-8")
            second_ref.write_text("{}\n", encoding="utf-8")

            def job(content_id: str, ref_jsonl: Path) -> SceneContextJob:
                return SceneContextJob(
                    content_id=content_id,
                    ref_jsonl=str(ref_jsonl),
                    scene_context_jsonl=str(root / f"{content_id}_output.jsonl"),
                    frames_dir=str(root),
                    timestamp_json=str(root / "timestamps.json"),
                )

            first_job = job("first", first_ref)
            second_job = job("second", second_ref)
            Path(first_job.scene_context_jsonl).write_text(
                '{"scene_idx":0,"keyframes":[],"vlm_visual_graph":{"scene_type":"unknown"}}\n',
                encoding="utf-8",
            )

            pbar = mock.Mock()
            with mock.patch(
                "src.scene_context_extraction.ondevice.cli.tqdm",
                return_value=pbar,
            ) as tqdm_mock:
                result = scene_progress(
                    [first_job, second_job]
                )
                scene_progress([first_job, second_job], force=True)

            self.assertIs(result.pbar, pbar)
            self.assertEqual(result.resumed_content_ids, frozenset({"first"}))
            self.assertEqual(result.pending_scenes, 2)
            self.assertEqual(
                tqdm_mock.call_args_list,
                [
                    mock.call(
                        total=3,
                        initial=1,
                        desc="On-device scene contexts",
                        unit="scene",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
                    ),
                    mock.call(
                        total=3,
                        initial=0,
                        desc="On-device scene contexts",
                        unit="scene",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
                    ),
                ],
            )
            with mock.patch(
                "src.scene_context_extraction.ondevice.cli.perf_counter",
                return_value=result.started_at + 5.0,
            ):
                update_generation_progress(result, 25, 2.0)
                update_scene_progress(result, 1)
            pbar.set_postfix_str.assert_called_once_with(
                "ETA 00:05 | 5.00s/scene | TPS 12.50",
                refresh=True,
            )
            pbar.update.assert_called_once_with(1)

    def test_output_uses_mistral_postfix(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OUTPUT_SAVE_PATH": "output", "LINUX_ASSETS_SAVE_PATH": "data"},
            clear=False,
        ):
            path = infer_scene_context_jsonl("001_VV3qIkq5ofY", "ministral3")
            job = manifest_row_to_job(
                {"content_id": "001_VV3qIkq5ofY"},
                "ministral3",
            )
            failure_path = ondevice_failure_path(job)
            context_path = ondevice_context_path(job)

        self.assertEqual(
            Path(path),
            Path("output/custom/viewing_context/img_only/fixed_15s/scene_context_graph_mistral/001_VV3qIkq5ofY_scene_context.jsonl"),
        )
        self.assertEqual(
            failure_path,
            Path("output/custom/failures/viewing_context/img_only/fixed_15s/scene_context_graph_mistral/001_VV3qIkq5ofY_failures.jsonl"),
        )
        self.assertEqual(
            context_path,
            Path("output/custom/viewing_context/img_only/fixed_15s/video_context_graph_mistral/001_VV3qIkq5ofY_context_graph_ond.json"),
        )

    def test_shot_wise_config_scopes_all_ondevice_paths(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OUTPUT_SAVE_PATH": "output", "LINUX_ASSETS_SAVE_PATH": "data"},
            clear=False,
        ):
            job = manifest_row_to_job(
                {"content_id": "demo"},
                "qwen3_vl",
                "shot_wise",
            )
            failure_path = ondevice_failure_path(job)
            context_path = ondevice_context_path(job)

        self.assertEqual(Path(job.ref_jsonl), Path("output/custom/asset/shot_wise/ref_jsonl/demo_ref.jsonl"))
        self.assertEqual(Path(job.frames_dir), Path("output/custom/asset/shot_wise/resized_keyframes/demo"))
        self.assertEqual(
            Path(job.timestamp_json),
            Path("data/demo/assets/timestamp_filtered.json"),
        )
        self.assertEqual(
            Path(job.scene_context_jsonl),
            Path("output/custom/viewing_context/img_only/shot_wise/scene_context_graph_qwen/demo_scene_context.jsonl"),
        )
        self.assertEqual(
            failure_path,
            Path("output/custom/failures/viewing_context/img_only/shot_wise/scene_context_graph_qwen/demo_failures.jsonl"),
        )
        self.assertEqual(
            context_path,
            Path("output/custom/viewing_context/img_only/shot_wise/video_context_graph_qwen/demo_context_graph_ond.json"),
        )

    def test_manifest_graph_path_override_is_ignored(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OUTPUT_SAVE_PATH": "output", "LINUX_ASSETS_SAVE_PATH": "data"},
            clear=False,
        ):
            job = manifest_row_to_job(
                {
                    "content_id": "001_VV3qIkq5ofY",
                    "graph_jsonl": "custom/graph.jsonl",
                    "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                }
            )

        self.assertEqual(
            Path(job.scene_context_jsonl),
            Path("output/custom/viewing_context/img_only/fixed_15s/scene_context_graph_qwen/001_VV3qIkq5ofY_scene_context.jsonl"),
        )

    def test_manifest_ref_jsonl_override_is_ignored(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OUTPUT_SAVE_PATH": "output", "LINUX_ASSETS_SAVE_PATH": "data"},
            clear=False,
        ):
            job = manifest_row_to_job(
                {
                    "content_id": "001_VV3qIkq5ofY",
                    "ref_jsonl": "custom/legacy_ref.jsonl",
                    "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                }
            )

        self.assertEqual(
            Path(job.ref_jsonl),
            Path("output/custom/asset/fixed_15s/ref_jsonl/001_VV3qIkq5ofY_ref.jsonl"),
        )

    def test_output_save_path_controls_context_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "pipeline_output"
            with mock.patch.dict(
                os.environ,
                {"OUTPUT_SAVE_PATH": str(output_root), "LINUX_ASSETS_SAVE_PATH": str(Path(tmp) / "assets")},
                clear=False,
            ):
                scene_context_path = infer_scene_context_jsonl("001_VV3qIkq5ofY")
                job = manifest_row_to_job(
                    {
                        "content_id": "001_VV3qIkq5ofY",
                        "url": "https://www.youtube.com/watch?v=VV3qIkq5ofY",
                    }
                )

            self.assertEqual(
                Path(scene_context_path),
                output_root / "custom" / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_qwen" / "001_VV3qIkq5ofY_scene_context.jsonl",
            )
            self.assertEqual(
                Path(job.ref_jsonl),
                output_root / "custom" / "asset" / "fixed_15s" / "ref_jsonl" / "001_VV3qIkq5ofY_ref.jsonl",
            )
            self.assertEqual(Path(job.metadata_json), output_root / "custom" / "asset" / "metadata" / "001_VV3qIkq5ofY.json")

    def test_frames_dir_falls_back_to_output_resized_content_id_dir(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with mock.patch.dict(os.environ, {"OUTPUT_SAVE_PATH": "output"}, clear=False):
                    path = resolve_frames_dir("001_VV3qIkq5ofY")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(Path(path), Path("output/custom/asset/fixed_15s/resized_keyframes/001_VV3qIkq5ofY"))


if __name__ == "__main__":
    unittest.main()
