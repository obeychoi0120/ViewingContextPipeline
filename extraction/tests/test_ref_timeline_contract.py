from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scene_context_extraction.graph_adapter.payload import get_keyframe_timestamps
from src.video_data_collection.data_processor import merge_scene_data


class RefTimelineContractTests(unittest.TestCase):
    def test_ref_timeline_uses_filtered_keyframe_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timestamps = root / "timestamp_filtered.json"
            output = root / "demo_ref.jsonl"
            timestamps.write_text(
                json.dumps(
                    [
                        {
                            "scene_start": 0.0,
                            "scene_end": 49.08,
                            "shot_change_timestamps": [0.0, 24.12, 34.53, 43.31],
                            "keyframe_timestamps": [0, 24, 35, 43],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            merge_scene_data(timestamps, "", "", output)
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(set(record), {"scene_idx", "timeline"})
        self.assertEqual([shot["shot_idx"] for shot in record["timeline"]], [0, 1, 2, 3])
        self.assertEqual([shot["timestamp"] for shot in record["timeline"]], [0, 24, 35, 43])
        self.assertTrue(
            all(set(shot) == {"shot_idx", "timestamp", "raw_asr", "raw_ocr"} for shot in record["timeline"])
        )

    def test_ref_generation_rejects_mismatched_shots_and_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timestamps = root / "timestamp_filtered.json"
            timestamps.write_text(
                json.dumps(
                    [
                        {
                            "scene_start": 0.0,
                            "scene_end": 10.0,
                            "shot_change_timestamps": [0.0, 5.0],
                            "keyframe_timestamps": [0],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "shot intervals"):
                merge_scene_data(timestamps, "", "", root / "demo_ref.jsonl")

    def test_fixed_30s_references_use_half_open_ocr_seconds_and_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timestamps = root / "timestamp_fixed_30s.json"
            ocr = root / "ocr_cleaned.json"
            output = root / "demo_ref.jsonl"
            timestamps.write_text(
                json.dumps(
                    [
                        {
                            "scene_start": 0,
                            "scene_end": 30,
                            "shot_change_timestamps": [0, 10, 20],
                            "keyframe_timestamps": [5, 15, 25],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            ocr.write_text(
                json.dumps(
                    [
                        {"start_time": 1, "end_time": 1, "texts": ["Repeated"]},
                        {"start_time": 2, "end_time": 2, "texts": ["Repeated!"]},
                        {"start_time": 9, "end_time": 9, "texts": ["At nine"]},
                        {"start_time": 10, "end_time": 10, "texts": ["At ten"]},
                    ]
                ),
                encoding="utf-8",
            )

            merge_scene_data(timestamps, "", ocr, output)
            timeline = json.loads(output.read_text(encoding="utf-8"))["timeline"]

        self.assertEqual(timeline[0]["timestamp"], 5)
        self.assertIn("At nine", timeline[0]["raw_ocr"])
        self.assertNotIn("At ten", timeline[0]["raw_ocr"])
        self.assertEqual(timeline[0]["raw_ocr"].count("Repeated"), 1)
        self.assertEqual(timeline[1]["timestamp"], 15)
        self.assertIn("At ten", timeline[1]["raw_ocr"])

    def test_ondevice_uses_ref_timeline_timestamps_first(self) -> None:
        timestamps = get_keyframe_timestamps(
            item={"timeline": [{"timestamp": 24}, {"timestamp": 35}]},
            scene_timestamps=[{"keyframe_timestamps": [0, 1]}],
            idx=0,
        )

        self.assertEqual(timestamps, [24, 35])


if __name__ == "__main__":
    unittest.main()
