from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.scene_context_extraction.ondevice.pipeline import (
    SceneContextJob,
    ondevice_failure_path,
    ondevice_context_path,
    output_covers_scenes,
    run_scene_context_job,
    run_scene_context_job_parallel,
)
from src.scene_context_extraction.graph_adapter.payload import select_scene_image_paths
from src.scene_context_extraction.graph_adapter.writer import build_graph_record
from src.scene_context_extraction.graph_core.fingerprint import (
    build_input_fingerprint,
    write_fingerprint,
)


class OnDeviceGraphContractTests(unittest.TestCase):
    def test_graph_record_has_exact_three_field_schema(self) -> None:
        scene = {
            "scene_idx": 4,
            "timeline": [
                {"time": [0, 3], "raw_asr": "first", "raw_ocr": "title, logo"},
                {"time": [3, 6], "raw_asr": "second", "raw_ocr": "logo, subtitle"},
            ],
        }

        record = build_graph_record(
            scene,
            0,
            [0, 3],
            {"scene_type": "unknown"},
        )

        self.assertEqual(
            list(record),
            [
                "scene_idx",
                "keyframes",
                "vlm_visual_graph",
            ],
        )
        self.assertEqual(record["scene_idx"], 4)
        self.assertEqual(record["keyframes"], [0, 3])
        self.assertEqual(
            record["vlm_visual_graph"],
            {"scene_type": "unknown"},
        )

    def test_timestamp_only_keyframes_are_selected_without_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            Image.new("RGB", (672, 384)).save(frames_dir / "0003.png")
            Image.new("RGB", (672, 384)).save(frames_dir / "Scene001_Shot01_0000.png")

            paths = select_scene_image_paths(
                frames_dir,
                item={"keyframe_timestamps": [0, 3]},
                scene_timestamps=[],
                fallback_idx=0,
            )

        self.assertEqual([Path(path).name for path in paths], ["0003.png"])

    def test_old_duration_schema_is_not_considered_complete(self) -> None:
        scenes = [{"scene_idx": 0}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "graph.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "duration": [0, 3],
                        "vlm_visual_graph": {},
                        "vlm_visual_graph_warnings": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertFalse(output_covers_scenes(output, scenes))

    def test_legacy_five_field_schema_is_normalized_without_reextraction(self) -> None:
        scenes = [{"scene_idx": 0}, {"scene_idx": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "graph.jsonl"
            records = [
                {
                    "scene_idx": scene_idx,
                    "keyframes": [scene_idx],
                    "raw_data": {"asr_text": "", "ocr_text": ""},
                    "vlm_visual_graph": {"scene_type": "unknown"},
                    "vlm_visual_graph_warnings": [],
                }
                for scene_idx in range(2)
            ]
            output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            self.assertTrue(output_covers_scenes(output, scenes))
            normalized = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(
            all(
                set(record) == {"scene_idx", "keyframes", "vlm_visual_graph"}
                for record in normalized
            )
        )

    def test_single_and_parallel_execution_write_the_same_schema(self) -> None:
        class InProcessWorkerPool:
            gpus = ["0", "1"]

            def run_tasks(self, tasks, on_task_complete=None):
                from src.scene_context_extraction.ondevice.extractor import extract_visual_graphs

                results = []
                for task in tasks:
                    with Path(task["input_json_path"]).open("r", encoding="utf-8") as f:
                        scenes = [json.loads(line) for line in f if line.strip()]
                    extract_visual_graphs(
                        model=None,
                        processor=None,
                        frame_save_folder=task["frame_save_folder"],
                        scenes=scenes,
                        timestamp_json_path=task["timestamp_json_path"],
                        final_output_path=task["final_output_path"],
                        content_id=task["content_id"],
                        failure_output_path=task["failure_output_path"],
                    )
                    result = {
                        **task,
                        "ok": True,
                        "output_jsonl": task["final_output_path"],
                        "failure_jsonl": task["failure_output_path"],
                    }
                    results.append(result)
                    if on_task_complete is not None:
                        on_task_complete(result)
                return results

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (672, 384)).save(frames / "0000.png")
            Image.new("RGB", (672, 384)).save(frames / "0003.png")
            raw_jsonl = root / "raw.jsonl"
            scenes = [
                {"scene_idx": 0, "timeline": [{"raw_asr": "a", "raw_ocr": "x"}]},
                {"scene_idx": 1, "timeline": [{"raw_asr": "b", "raw_ocr": "y"}]},
            ]
            raw_jsonl.write_text("".join(json.dumps(scene) + "\n" for scene in scenes), encoding="utf-8")
            timestamps = root / "timestamp_filtered.json"
            timestamps.write_text(
                json.dumps([{"keyframe_timestamps": [0]}, {"keyframe_timestamps": [3]}]),
                encoding="utf-8",
            )
            single_output = root / "single.jsonl"
            parallel_output = root / "parallel.jsonl"
            common = {
                "content_id": "demo",
                "ref_jsonl": str(raw_jsonl),
                "frames_dir": str(frames),
                "timestamp_json": str(timestamps),
            }
            single_progress = []
            parallel_progress = []

            with mock.patch(
                "src.scene_context_extraction.ondevice.extractor.extract_scene_graph",
                return_value=({"scene_type": "unknown"}, ["warning stays in logs only"]),
            ):
                run_scene_context_job(
                    SceneContextJob(scene_context_jsonl=str(single_output), **common),
                    None,
                    None,
                    progress_callback=single_progress.append,
                )
                single_profile = json.loads(
                    ondevice_context_path(
                        SceneContextJob(scene_context_jsonl=str(single_output), **common)
                    ).read_text(encoding="utf-8")
                )
                run_scene_context_job_parallel(
                    SceneContextJob(scene_context_jsonl=str(parallel_output), **common),
                    InProcessWorkerPool(),
                    progress_callback=parallel_progress.append,
                )

            single_records = [json.loads(line) for line in single_output.read_text(encoding="utf-8").splitlines()]
            parallel_records = [json.loads(line) for line in parallel_output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(single_records, parallel_records)
        self.assertEqual(single_progress, [1, 1])
        self.assertEqual(parallel_progress, [1, 1])
        self.assertEqual(single_profile["content_id"], "demo")
        self.assertEqual(single_profile["source_scene_context_path"], str(single_output))
        self.assertIn("content_axes_4d", single_profile["context"])
        self.assertTrue(all(set(record) == {
            "scene_idx",
            "keyframes",
            "vlm_visual_graph",
        } for record in single_records))

    def test_complete_scene_context_rebuilds_missing_profile_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (672, 384)).save(frames / "0000.png")
            ref_jsonl = root / "ref.jsonl"
            ref_jsonl.write_text(json.dumps({"scene_idx": 0}) + "\n", encoding="utf-8")
            timestamps = root / "timestamp.json"
            timestamps.write_text(
                json.dumps([{"keyframe_timestamps": [0]}]),
                encoding="utf-8",
            )
            output = root / "scene_context.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "keyframes": [0],
                        "vlm_visual_graph": {"scene_type": "unknown"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            job = SceneContextJob(
                content_id="demo",
                ref_jsonl=str(ref_jsonl),
                scene_context_jsonl=str(output),
                frames_dir=str(frames),
                timestamp_json=str(timestamps),
            )
            write_fingerprint(
                output,
                build_input_fingerprint(
                    content_id="demo",
                    scenes=[{"scene_idx": 0}],
                    frames_dir=frames,
                    multimodal=False,
                    backend="ondevice",
                    model_config={},
                ),
            )

            run_scene_context_job(job, model=None, processor=None)

            profile_path = ondevice_context_path(job)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))

        self.assertEqual(profile["content_id"], "demo")

    def test_failed_scene_is_written_only_to_terminal_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (672, 384)).save(frames / "0000.png")
            ref_jsonl = root / "ref.jsonl"
            ref_jsonl.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "timeline": [{"timestamp": 0}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            timestamps = root / "timestamp.json"
            timestamps.write_text(
                json.dumps([{"keyframe_timestamps": [0]}]),
                encoding="utf-8",
            )
            output = root / "scene_context.jsonl"
            job = SceneContextJob(
                content_id="demo",
                ref_jsonl=str(ref_jsonl),
                scene_context_jsonl=str(output),
                frames_dir=str(frames),
                timestamp_json=str(timestamps),
            )

            with mock.patch(
                "src.scene_context_extraction.ondevice.extractor.extract_scene_graph",
                return_value=(None, ["deterministic failure"]),
            ):
                first = run_scene_context_job(
                    job,
                    model=None,
                    processor=None,
                )
                second = run_scene_context_job(
                    job,
                    model=None,
                    processor=None,
                )

            failures = [
                json.loads(line)
                for line in ondevice_failure_path(job)
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(first["failed"], 1)
        self.assertEqual(second["failed"], 1)
        self.assertFalse(output.exists())
        self.assertFalse(ondevice_context_path(job).exists())
        self.assertEqual(
            failures,
            [
                {
                    "content_id": "demo",
                    "scene_idx": 0,
                    "keyframes": [0],
                    "error": "deterministic failure",
                }
            ],
        )

    def test_partial_profile_uses_successes_and_resume_skips_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (672, 384)).save(frames / "0000.png")
            Image.new("RGB", (672, 384)).save(frames / "0003.png")
            scenes = [
                {
                    "scene_idx": 0,
                    "timeline": [{"timestamp": 0}],
                },
                {
                    "scene_idx": 1,
                    "timeline": [{"timestamp": 3}],
                },
            ]
            ref_jsonl = root / "ref.jsonl"
            ref_jsonl.write_text(
                "".join(json.dumps(scene) + "\n" for scene in scenes),
                encoding="utf-8",
            )
            timestamps = root / "timestamp.json"
            timestamps.write_text(
                json.dumps(
                    [
                        {"keyframe_timestamps": [0]},
                        {"keyframe_timestamps": [3]},
                    ]
                ),
                encoding="utf-8",
            )
            job = SceneContextJob(
                content_id="demo",
                ref_jsonl=str(ref_jsonl),
                scene_context_jsonl=str(root / "scene_context.jsonl"),
                frames_dir=str(frames),
                timestamp_json=str(timestamps),
            )

            with mock.patch(
                "src.scene_context_extraction.ondevice.extractor.extract_scene_graph",
                side_effect=[
                    ({"scene_type": "unknown"}, []),
                    (None, ["terminal"]),
                ],
            ):
                first = run_scene_context_job(job, None, None)
            with mock.patch(
                "src.scene_context_extraction.ondevice.extractor.extract_scene_graph"
            ) as inference:
                resumed_progress = []
                resumed = run_scene_context_job(
                    job,
                    None,
                    None,
                    progress_callback=resumed_progress.append,
                )

            canonical = [
                json.loads(line)
                for line in Path(job.scene_context_jsonl)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            profile = json.loads(
                ondevice_context_path(job).read_text(encoding="utf-8")
            )

        self.assertEqual(first["failed"], 1)
        self.assertEqual(resumed["failed"], 1)
        self.assertEqual(resumed_progress, [])
        inference.assert_not_called()
        self.assertEqual(
            [record["scene_idx"] for record in canonical],
            [0],
        )
        self.assertIn(
            "scene_ids=1",
            profile["aggregation_warnings"][-1],
        )

    def test_parallel_execution_merges_success_and_failure_sidecars(
        self,
    ) -> None:
        class InProcessWorkerPool:
            gpus = ["0", "1"]

            def run_tasks(self, tasks):
                from src.scene_context_extraction.ondevice.extractor import (
                    extract_visual_graphs,
                )

                results = []
                for task in tasks:
                    scenes = [
                        json.loads(line)
                        for line in Path(task["input_json_path"])
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    ]
                    summary = extract_visual_graphs(
                        model=None,
                        processor=None,
                        frame_save_folder=task["frame_save_folder"],
                        scenes=scenes,
                        timestamp_json_path=task["timestamp_json_path"],
                        final_output_path=task["final_output_path"],
                        content_id=task["content_id"],
                        failure_output_path=task["failure_output_path"],
                    )
                    results.append(
                        {
                            **task,
                            "ok": True,
                            "output_jsonl": task["final_output_path"],
                            "failure_jsonl": task["failure_output_path"],
                            "summary": summary,
                        }
                    )
                return results

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (672, 384)).save(frames / "0000.png")
            Image.new("RGB", (672, 384)).save(frames / "0003.png")
            scenes = [
                {"scene_idx": 0, "timeline": [{"timestamp": 0}]},
                {"scene_idx": 1, "timeline": [{"timestamp": 3}]},
            ]
            ref_jsonl = root / "ref.jsonl"
            ref_jsonl.write_text(
                "".join(json.dumps(scene) + "\n" for scene in scenes),
                encoding="utf-8",
            )
            timestamps = root / "timestamp.json"
            timestamps.write_text(
                json.dumps(
                    [
                        {"keyframe_timestamps": [0]},
                        {"keyframe_timestamps": [3]},
                    ]
                ),
                encoding="utf-8",
            )
            job = SceneContextJob(
                content_id="demo",
                ref_jsonl=str(ref_jsonl),
                scene_context_jsonl=str(root / "parallel.jsonl"),
                frames_dir=str(frames),
                timestamp_json=str(timestamps),
            )

            def fake_extract(**kwargs):
                if kwargs["image_paths"][0].endswith("0000.png"):
                    return {"scene_type": "unknown"}, []
                return None, ["parallel terminal"]

            with mock.patch(
                "src.scene_context_extraction.ondevice.extractor.extract_scene_graph",
                side_effect=fake_extract,
            ):
                summary = run_scene_context_job_parallel(
                    job,
                    InProcessWorkerPool(),
                )
            canonical = [
                json.loads(line)
                for line in Path(job.scene_context_jsonl)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            failures = [
                json.loads(line)
                for line in ondevice_failure_path(job)
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(
            [record["scene_idx"] for record in canonical],
            [0],
        )
        self.assertEqual(
            [record["scene_idx"] for record in failures],
            [1],
        )


if __name__ == "__main__":
    unittest.main()
