from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from src.scene_context_extraction.ref.extractor import (
    SceneContextRefConfig,
    extract_scene_contexts_ref,
    image_part,
    load_scene_context_rows,
    output_path_for_content,
    context_path_for_content,
    reference_failure_path,
    scene_context_ref_output_dir,
    scene_context_image_paths,
    scene_from_context_row,
    video_context_graph_ref_output_dir,
)
from src.scene_context_extraction.graph_core.prompt import SCENE_EXTRACTION_PROMPT, USER_MESSAGE
from src.scene_context_extraction.graph_core.validator import validate_observation
from src.scene_context_extraction.graph_core.fingerprint import (
    build_input_fingerprint,
    write_fingerprint,
)


def write_ref_fingerprint(
    output: Path,
    rows: list[dict[str, Any]],
    config: SceneContextRefConfig,
) -> None:
    write_fingerprint(
        output,
        build_input_fingerprint(
            content_id="demo",
            scenes=rows,
            frames_dir=(
                Path(config.output_dir)
                / "asset"
                / config.shot_interval
                / "resized_keyframes"
                / "demo"
            ),
            multimodal=config.multimodal,
            backend="ref",
            model_config={
                "model": config.model,
                "location": config.location,
                "thinking_level": config.thinking_level,
            },
        ),
    )


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, model: str, contents: list[Any], config: Any) -> FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return FakeResponse(json.dumps(self.response))


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.models = FakeModels(response)


class ConcurrencyTrackingModels:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.first_wave = threading.Barrier(8)
        self.calls_by_content: dict[str, int] = {}
        self.active_by_content: dict[str, int] = {}
        self.max_active = 0
        self.contents_overlapped = False

    def generate_content(self, model: str, contents: list[Any], config: Any) -> FakeResponse:
        content_id = Path(contents[0].inline_data.data.decode("utf-8")).parent.name
        with self.lock:
            call_number = self.calls_by_content.get(content_id, 0) + 1
            self.calls_by_content[content_id] = call_number
            self.active_by_content[content_id] = self.active_by_content.get(content_id, 0) + 1
            self.max_active = max(self.max_active, sum(self.active_by_content.values()))
            if len(self.active_by_content) > 1:
                self.contents_overlapped = True

        try:
            if content_id == "a" and call_number <= 8:
                self.first_wave.wait(timeout=5)
            return FakeResponse("{}")
        finally:
            with self.lock:
                self.active_by_content[content_id] -= 1
                if self.active_by_content[content_id] == 0:
                    del self.active_by_content[content_id]


class ConcurrencyTrackingClient:
    def __init__(self) -> None:
        self.models = ConcurrencyTrackingModels()


class FailingModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, model: str, contents: list[Any], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        raise RuntimeError("boom")


class FailingClient:
    def __init__(self) -> None:
        self.models = FailingModels()


def scene_context_rows(
    content_id: str = "demo",
    scenes: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_idx, timestamps in enumerate(scenes or [[0]]):
        rows.append(
            {
                "content_id": content_id,
                "scene_idx": scene_idx,
                "keyframes": timestamps,
            }
        )
    return rows


def write_keyframes(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    shot_interval: str = "fixed_15s",
) -> None:
    paths = {
        Path(output_dir)
        / "asset"
        / shot_interval
        / "resized_keyframes"
        / row["content_id"]
        / f"{timestamp:04d}.png"
        for row in rows
        for timestamp in row["keyframes"]
    }
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(path).encode("utf-8"))


def ref_config(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    **overrides: Any,
) -> SceneContextRefConfig:
    shot_interval = str(overrides.get("shot_interval", "fixed_15s"))
    write_keyframes(output_dir, rows, shot_interval=shot_interval)
    return SceneContextRefConfig(
        gcp_project_id="project",
        output_dir=str(output_dir),
        **overrides,
    )


class SceneContextRefExtractorTests(unittest.TestCase):
    def test_load_scene_context_rows_lists_direct_files_and_sorts_contents_and_scenes(self) -> None:
        scene_1 = {
            "scene_idx": 1,
            "timeline": [
                {"shot_idx": 0, "timestamp": 22, "raw_asr": "second", "raw_ocr": "title"},
                {"shot_idx": 1, "timestamp": 30, "raw_asr": "scene", "raw_ocr": "title, logo"},
            ],
        }
        scene_0 = {
            "scene_idx": 0,
            "timeline": [
                {"shot_idx": 0, "timestamp": 0, "raw_asr": "first", "raw_ocr": "title"},
                {"shot_idx": 1, "timestamp": 3, "raw_asr": "", "raw_ocr": ""},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            (ref_dir / "z_ref.jsonl").write_text(json.dumps(scene_0), encoding="utf-8")
            (ref_dir / "nested").mkdir()
            (ref_dir / "nested" / "ignored_ref.jsonl").write_text(json.dumps(scene_0), encoding="utf-8")
            (ref_dir / "a_ref.jsonl").write_text(
                "\n".join(
                    map(
                        json.dumps,
                        [{"_type": "video_metadata", "title": "demo"}, scene_1, scene_0],
                    )
                ),
                encoding="utf-8",
            )
            (ref_dir / "readme.txt").write_text("ignored", encoding="utf-8")

            rows, file_count = load_scene_context_rows(ref_dir)

        self.assertEqual(file_count, 2)
        self.assertEqual([(row["content_id"], row["scene_idx"]) for row in rows], [("a", 0), ("a", 1), ("z", 0)])
        self.assertEqual(rows[0], {"content_id": "a", "scene_idx": 0, "keyframes": [0, 3]})

    def test_load_scene_context_rows_expands_sample_16_scenes_and_55_keyframes(self) -> None:
        scenes = [
            [0, 34, 59], [66, 79, 89, 105], [116, 122, 132, 146], [151, 158, 164, 176],
            [183, 198, 204, 221], [230, 254, 261, 287], [293, 298, 313, 328],
            [340, 370, 384, 393], [405, 416, 421, 427], [435, 458, 474, 495], [501],
            [570, 600, 609], [631, 637, 649, 664], [674, 683, 689, 717], [724, 744], [804, 807],
        ]
        text = "\n".join(
            json.dumps(
                {
                    "scene_idx": scene_idx,
                    "timeline": [
                        {"shot_idx": shot_idx, "timestamp": timestamp, "raw_asr": "", "raw_ocr": ""}
                        for shot_idx, timestamp in enumerate(keyframes)
                    ],
                }
            )
            for scene_idx, keyframes in enumerate(scenes)
        )
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "sample_ref.jsonl"
            ref_path.write_text(text, encoding="utf-8")
            rows, _ = load_scene_context_rows(Path(tmp))

        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(len(row["keyframes"]) for row in rows), 55)
        self.assertEqual(rows[-1]["keyframes"][-1], 807)

    def test_load_scene_context_rows_does_not_create_runtime_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp) / "ref_jsonl"
            ref_dir.mkdir()
            (ref_dir / "demo_ref.jsonl").write_text(
                json.dumps({"scene_idx": 0, "timeline": [{"timestamp": 7}]}),
                encoding="utf-8",
            )
            rows, file_count = load_scene_context_rows(ref_dir)

            self.assertEqual(rows, [{"content_id": "demo", "scene_idx": 0, "keyframes": [7]}])
            self.assertEqual(file_count, 1)
            self.assertEqual([path.name for path in ref_dir.iterdir()], ["demo_ref.jsonl"])

    def test_load_scene_context_rows_rejects_invalid_input_before_extraction(self) -> None:
        timeline_item = {"shot_idx": 0, "timestamp": 0, "raw_asr": "", "raw_ocr": ""}
        valid = {"scene_idx": 0, "timeline": [timeline_item]}
        cases = {
            "invalid JSON": "{",
            "scene_idx": json.dumps({**valid, "scene_idx": -1}),
            "timeline": json.dumps({**valid, "timeline": []}),
            "timestamp": json.dumps({**valid, "timeline": [{**timeline_item, "timestamp": 1.2}]}),
            "strictly increasing": json.dumps(
                {**valid, "timeline": [timeline_item, {**timeline_item, "shot_idx": 1}]}
            ),
        }
        for message, text in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    with tempfile.TemporaryDirectory() as tmp:
                        (Path(tmp) / "demo_ref.jsonl").write_text(text, encoding="utf-8")
                        load_scene_context_rows(tmp)

    def test_load_scene_context_rows_ignores_asr_and_ocr_shape(self) -> None:
        record = {
            "scene_idx": 0,
            "timeline": [{"timestamp": 0, "raw_asr": None, "raw_ocr": {"bad": "shape"}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "demo_ref.jsonl").write_text(json.dumps(record), encoding="utf-8")

            rows, _ = load_scene_context_rows(tmp)

        self.assertEqual(rows[0], {"content_id": "demo", "scene_idx": 0, "keyframes": [0]})

    def test_load_scene_context_rows_rejects_duplicate_scene_idx_and_empty_prefix(self) -> None:
        record = json.dumps(
            {
                "scene_idx": 0,
                "timeline": [{"shot_idx": 0, "timestamp": 0, "raw_asr": "", "raw_ocr": ""}],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "demo_ref.jsonl").write_text(f"{record}\n{record}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate scene_idx"):
                load_scene_context_rows(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no .*_ref.jsonl"):
                load_scene_context_rows(tmp)

    def test_context_scene_builds_local_image_paths(self) -> None:
        row = scene_context_rows(content_id="video-1", scenes=[[20]])[0]

        self.assertEqual(
            scene_from_context_row(row),
            {
                "scene_idx": 0,
                "keyframes": [20],
            },
        )
        self.assertEqual(
            scene_context_image_paths(row, "output"),
            [str(Path("output/asset/fixed_15s/resized_keyframes/video-1/0020.png"))],
        )
        self.assertEqual(
            scene_context_image_paths(row, "output", "shot_wise"),
            [str(Path("output/asset/shot_wise/resized_keyframes/video-1/0020.png"))],
        )

    def test_image_part_embeds_local_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            path.write_bytes(b"png-data")
            part = image_part(path)

        self.assertEqual(part.inline_data.data, b"png-data")
        self.assertEqual(part.inline_data.mime_type, "image/png")
        with self.assertRaises(FileNotFoundError):
            image_part("local/path/missing.png")

    def test_extract_scene_contexts_ref_writes_four_field_visual_graph_output(self) -> None:
        response = {
            "scene_type": "people_social",
            "visual_style_cues": {
                "media_form": "live_action",
                "fantasy_element": "none",
                "shot_scale": "medium",
                "graphic_density": "low",
                "composition_density": "balanced",
            },
            "people_density": "one",
            "face_prominence": "mid",
            "mood_bin": "neutral",
            "affect_cues": ["neutral"],
            "scene_function": "information_report",
            "setting": "studio",
            "entities": [
                {
                    "local_id": "e1",
                    "category": "person",
                    "name": "anchor",
                    "role": "main_subject",
                    "relations": {"DOING": "reporting"},
                }
            ],
            "content_axes_4d": {"subject_sociality": 1.0},
            "style": "editorial_portrait",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            client = FakeClient(response)
            rows = scene_context_rows()

            video_context_output_dir = Path(tmp) / "video_context_graph_ref"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                summary = extract_scene_contexts_ref(
                    scene_context_rows=rows,
                    output_path=output,
                    client=client,
                    config=ref_config(tmp, rows, model="gemini-test"),
                    extraction_config={"fake": True},
                    sleep_sec=0,
                    video_context_output_dir=video_context_output_dir,
                    show_progress=False,
                )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            profile_document = json.loads(
                (video_context_output_dir / "demo_context_graph_ref.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["video_contexts_written"], 1)
        self.assertNotIn("scene=0", stdout.getvalue())
        self.assertNotIn("[Done]", stdout.getvalue())
        self.assertNotIn('"content_axes_4d"', stdout.getvalue())
        self.assertNotIn('"top_motifs"', stdout.getvalue())
        self.assertEqual(records[0]["scene_idx"], 0)
        self.assertEqual(
            set(records[0]),
            {"scene_idx", "keyframes", "vlm_visual_graph", "vlm_visual_graph_warnings"},
        )
        self.assertEqual(records[0]["keyframes"], [0])
        self.assertNotIn("raw_data", records[0])
        self.assertNotIn("content_axes_4d", records[0]["vlm_visual_graph"])
        self.assertEqual(profile_document["content_id"], "demo")
        self.assertEqual(profile_document["source_scene_context_path"], str(output))
        self.assertIn("content_axes_4d", profile_document["context"])
        self.assertIn("top_motifs", profile_document["context"])
        self.assertEqual(client.models.calls[0]["model"], "gemini-test")
        self.assertEqual(client.models.calls[0]["contents"][0].inline_data.mime_type, "image/png")
        self.assertEqual(client.models.calls[0]["contents"][-1], USER_MESSAGE)
        self.assertNotIn("asr-0", str(client.models.calls[0]["contents"]))

    def test_extract_scene_contexts_ref_can_write_training_messages_separately(self) -> None:
        response = {
            "scene_type": "people_social",
            "visual_style_cues": {
                "media_form": "live_action",
                "fantasy_element": "none",
                "shot_scale": "medium",
                "graphic_density": "low",
                "composition_density": "balanced",
            },
            "people_density": "one",
            "face_prominence": "mid",
            "mood_bin": "neutral",
            "affect_cues": ["neutral"],
            "scene_function": "information_report",
            "setting": "studio",
            "entities": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo_scene_context_ref.jsonl"
            training_output = Path(tmp) / "training.jsonl"

            rows = scene_context_rows()
            extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=FakeClient(response),
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                training_output_path=training_output,
                show_progress=False,
            )

            visual_record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            training_record = json.loads(training_output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(training_record["content_id"], "demo")
        self.assertEqual(training_record["reference_model"], "gemini-3.5-flash")
        self.assertEqual(training_record["reference_error"], "")
        self.assertEqual(
            set(training_record),
            {
                "content_id",
                "scene_idx",
                "image_paths",
                "target_graph",
                "messages",
                "reference_model",
                "reference_error",
            },
        )
        self.assertEqual(json.loads(training_record["messages"][2]["content"]), visual_record["vlm_visual_graph"])
        self.assertEqual(
            training_record["image_paths"],
            [str(Path(tmp) / "asset/fixed_15s/resized_keyframes/demo/0000.png")],
        )

    def test_output_dir_writes_content_id_visual_graph_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"

            path = output_path_for_content("ignored.jsonl", output_dir, "demo")

        self.assertEqual(path, output_dir / "demo_scene_context_ref.jsonl")
        self.assertEqual(
            scene_context_ref_output_dir(Path("root")),
            Path("root") / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_ref",
        )
        self.assertEqual(
            video_context_graph_ref_output_dir(Path("root")),
            Path("root") / "viewing_context" / "img_only" / "fixed_15s" / "video_context_graph_ref",
        )
        self.assertEqual(
            scene_context_ref_output_dir(Path("root"), "shot_wise"),
            Path("root") / "viewing_context" / "img_only" / "shot_wise" / "scene_context_graph_ref",
        )
        self.assertEqual(
            video_context_graph_ref_output_dir(Path("root"), "shot_wise"),
            Path("root") / "viewing_context" / "img_only" / "shot_wise" / "video_context_graph_ref",
        )
        self.assertEqual(
            context_path_for_content(Path("root/video_context_graph_ref"), "demo"),
            Path("root/video_context_graph_ref/demo_context_graph_ref.json"),
        )

    def test_resume_does_not_skip_incomplete_legacy_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            output.write_text(
                json.dumps({"scene_idx": 0, "vlm_visual_graph": {"scene_type": "unknown"}}) + "\n",
                encoding="utf-8",
            )
            client = FakeClient({})

            rows = scene_context_rows()
            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                resume=True,
                video_context_output_dir=Path(tmp) / "video_context_graph_ref",
                show_progress=False,
            )

        self.assertEqual(len(client.models.calls), 1)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["video_contexts_written"], 1)

    def test_resume_skips_scene_only_when_source_fields_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            rows = scene_context_rows()
            output.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "keyframes": [0],
                        "raw_data": {"asr_text": "asr-0", "ocr_text": "ocr-0"},
                        "vlm_visual_graph": {"scene_type": "unknown"},
                        "vlm_visual_graph_warnings": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            client = FakeClient({})
            config = ref_config(tmp, rows)
            write_ref_fingerprint(output, rows, config)

            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=config,
                extraction_config={},
                sleep_sec=0,
                resume=True,
                video_context_output_dir=Path(tmp) / "video_context_graph_ref",
                show_progress=False,
            )
            normalized_record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(len(client.models.calls), 0)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(
            set(normalized_record),
            {"scene_idx", "keyframes", "vlm_visual_graph", "vlm_visual_graph_warnings"},
        )
        self.assertNotIn("raw_data", normalized_record)

    def test_force_mode_reextracts_complete_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            rows = scene_context_rows()
            output.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "keyframes": [0],
                        "vlm_visual_graph": {"scene_type": "unknown"},
                        "vlm_visual_graph_warnings": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            client = FakeClient({})

            extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                resume=False,
                video_context_output_dir=Path(tmp) / "video_context_graph_ref",
                show_progress=False,
            )

        self.assertEqual(len(client.models.calls), 1)

    def test_resume_does_not_skip_scene_when_keyframes_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "keyframes": [99],
                        "raw_data": {"asr_text": "asr-0", "ocr_text": "ocr-0"},
                        "vlm_visual_graph": {"scene_type": "unknown"},
                        "vlm_visual_graph_warnings": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = scene_context_rows()
            client = FakeClient({})

            extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                resume=True,
                show_progress=False,
            )

        self.assertEqual(len(client.models.calls), 1)

    def test_resume_does_not_skip_scene_with_invalid_graph_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "scene_idx": 0,
                        "keyframes": [0],
                        "vlm_visual_graph": [],
                        "vlm_visual_graph_warnings": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = scene_context_rows()
            client = FakeClient({})

            extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                resume=True,
                show_progress=False,
            )

        self.assertEqual(len(client.models.calls), 1)

    def test_failed_scene_is_excluded_from_reference_profile(self) -> None:
        class PartiallyFailingModels:
            def __init__(self) -> None:
                self.calls = []

            def generate_content(self, model, contents, config):
                self.calls.append(
                    {"model": model, "contents": contents, "config": config}
                )
                path = contents[0].inline_data.data.decode("utf-8")
                if path.endswith("0000.png"):
                    raise RuntimeError("boom")
                return FakeResponse("{}")

        class PartiallyFailingClient:
            def __init__(self) -> None:
                self.models = PartiallyFailingModels()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            video_context_output_dir = Path(tmp) / "video_context_graph_ref"
            client = PartiallyFailingClient()

            rows = scene_context_rows(scenes=[[0], [10]])
            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                video_context_output_dir=video_context_output_dir,
                show_progress=False,
            )

            profile_document = json.loads(
                (video_context_output_dir / "demo_context_graph_ref.json").read_text(encoding="utf-8")
            )
            canonical_records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            failure_records = [
                json.loads(line)
                for line in reference_failure_path(output, "demo")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(
            [record["scene_idx"] for record in canonical_records],
            [1],
        )
        self.assertEqual(
            failure_records,
            [
                {
                    "content_id": "demo",
                    "scene_idx": 0,
                    "keyframes": [0],
                    "error": "boom",
                }
            ],
        )
        self.assertEqual(profile_document["content_id"], "demo")
        self.assertEqual(profile_document["context"]["top_entities"], [])
        self.assertIn(
            "scene_ids=0",
            profile_document["aggregation_warnings"][-1],
        )

    def test_terminal_failure_resumes_without_retry_and_force_clears_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            profile_dir = Path(tmp) / "video_context_graph_ref"
            rows = scene_context_rows()
            config = ref_config(tmp, rows)

            failed = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=FailingClient(),
                config=config,
                extraction_config={},
                sleep_sec=0,
                video_context_output_dir=profile_dir,
                show_progress=False,
            )
            failed_output_exists = output.exists()
            failed_profile_exists = (
                profile_dir / "demo_context_graph_ref.json"
            ).exists()
            failed_sidecar_exists = reference_failure_path(
                output,
                "demo",
            ).exists()
            resumed_client = FakeClient({})
            resumed = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=resumed_client,
                config=config,
                extraction_config={},
                sleep_sec=0,
                video_context_output_dir=profile_dir,
                show_progress=False,
            )
            forced_client = FakeClient({})
            forced = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=forced_client,
                config=config,
                extraction_config={},
                sleep_sec=0,
                resume=False,
                video_context_output_dir=profile_dir,
                show_progress=False,
            )
            sidecar = reference_failure_path(output, "demo")
            sidecar_exists = sidecar.exists()
            output_exists = output.exists()
            profile_exists = (
                profile_dir / "demo_context_graph_ref.json"
            ).exists()

        self.assertEqual(failed["failed"], 1)
        self.assertFalse(failed_output_exists)
        self.assertFalse(failed_profile_exists)
        self.assertTrue(failed_sidecar_exists)
        self.assertEqual(resumed["failed"], 1)
        self.assertEqual(len(resumed_client.models.calls), 0)
        self.assertEqual(forced["failed"], 0)
        self.assertEqual(len(forced_client.models.calls), 1)
        self.assertFalse(sidecar_exists)
        self.assertTrue(output_exists)
        self.assertTrue(profile_exists)

    def test_invalid_json_is_terminal_without_application_retry(self) -> None:
        class InvalidJsonModels:
            def __init__(self) -> None:
                self.calls = []

            def generate_content(self, model, contents, config):
                self.calls.append(
                    {"model": model, "contents": contents, "config": config}
                )
                return FakeResponse("{")

        class InvalidJsonClient:
            def __init__(self) -> None:
                self.models = InvalidJsonModels()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            rows = scene_context_rows()
            client = InvalidJsonClient()
            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                show_progress=False,
            )

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(client.models.calls), 1)

    def test_missing_keyframe_skips_entire_content_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_rows = scene_context_rows(content_id="a-missing", scenes=[[0], [10]])
            valid_rows = scene_context_rows(content_id="b-valid", scenes=[[20]])
            client = FakeClient({})
            output_dir = Path(tmp) / "scene_context_ref"
            video_context_output_dir = Path(tmp) / "video_context_graph_ref"

            summary = extract_scene_contexts_ref(
                scene_context_rows=missing_rows + valid_rows,
                output_path=output_dir / "unused.jsonl",
                output_dir=output_dir,
                client=client,
                config=ref_config(tmp, valid_rows),
                extraction_config={},
                sleep_sec=0,
                video_context_output_dir=video_context_output_dir,
                show_progress=False,
            )

            missing_output_exists = (output_dir / "a-missing_scene_context_ref.jsonl").exists()
            valid_output_exists = (output_dir / "b-valid_scene_context_ref.jsonl").exists()
            missing_profile_exists = (video_context_output_dir / "a-missing_context_graph_ref.json").exists()
            valid_profile = json.loads(
                (video_context_output_dir / "b-valid_context_graph_ref.json").read_text(encoding="utf-8")
            )

        self.assertFalse(missing_output_exists)
        self.assertFalse(missing_profile_exists)
        self.assertTrue(valid_output_exists)
        self.assertEqual(valid_profile["content_id"], "b-valid")
        self.assertEqual(len(client.models.calls), 1)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["succeeded"], 1)

    def test_completed_scene_context_rebuilds_missing_profile_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene_context_ref.jsonl"
            video_context_output_dir = Path(tmp) / "video_context_graph_ref"
            client = FakeClient({})

            rows = scene_context_rows(scenes=[[0], [10]])
            output.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "scene_idx": row["scene_idx"],
                            "keyframes": row["keyframes"],
                            "raw_data": {"asr_text": "legacy", "ocr_text": "legacy"},
                            "vlm_visual_graph": {"scene_type": "unknown"},
                            "vlm_visual_graph_warnings": [],
                        }
                    )
                    for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
            config = ref_config(tmp, rows)
            write_ref_fingerprint(output, rows, config)
            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output,
                client=client,
                config=config,
                extraction_config={},
                sleep_sec=0,
                video_context_output_dir=video_context_output_dir,
                show_progress=False,
            )
            output_exists = output.exists()
            profile_document = json.loads(
                (video_context_output_dir / "demo_context_graph_ref.json").read_text(encoding="utf-8")
            )
            normalized_records = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(client.models.calls), 0)
        self.assertTrue(output_exists)
        self.assertEqual(profile_document["content_id"], "demo")
        self.assertTrue(all("raw_data" not in record for record in normalized_records))
        self.assertEqual(summary["skipped"], 2)
        self.assertEqual(summary["video_contexts_written"], 1)

    def test_processes_eight_scenes_in_parallel_without_overlapping_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_rows = scene_context_rows(content_id="a", scenes=[[index] for index in range(9)])
            second_rows = scene_context_rows(content_id="b", scenes=[[100], [101]])
            rows = first_rows + second_rows
            client = ConcurrencyTrackingClient()
            config = ref_config(tmp, rows)
            output_dir = Path(tmp) / "scene_context_ref"

            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=output_dir / "unused.jsonl",
                output_dir=output_dir,
                client=client,
                config=config,
                extraction_config={},
                sleep_sec=0,
                video_context_output_dir=Path(tmp) / "video_context_graph_ref",
                show_progress=False,
            )

            first_records = [
                json.loads(line)
                for line in (output_dir / "a_scene_context_ref.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(client.models.max_active, 8)
        self.assertFalse(client.models.contents_overlapped)
        self.assertEqual(client.models.calls_by_content, {"a": 9, "b": 2})
        self.assertEqual([record["scene_idx"] for record in first_records], list(range(9)))
        self.assertEqual(summary["succeeded"], 11)

    def test_max_scenes_partial_content_removes_stale_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = scene_context_rows(scenes=[[0], [10]])
            profile_dir = Path(tmp) / "video_context_graph_ref"
            profile_dir.mkdir()
            profile_path = profile_dir / "demo_context_graph_ref.json"
            profile_path.write_text("{}", encoding="utf-8")
            client = FakeClient({})

            summary = extract_scene_contexts_ref(
                scene_context_rows=rows,
                output_path=Path(tmp) / "scene_context_ref.jsonl",
                client=client,
                config=ref_config(tmp, rows),
                extraction_config={},
                sleep_sec=0,
                max_scenes=1,
                video_context_output_dir=profile_dir,
                show_progress=False,
            )
            profile_exists = profile_path.exists()

        self.assertFalse(profile_exists)
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["video_contexts_written"], 0)

    def test_uses_shared_prompt_and_validator_import_paths(self) -> None:
        normalized, _ = validate_observation({"entities": []})

        self.assertEqual(normalized["scene_type"], "unknown")
        self.assertIn("IMPORTANT entity rules", SCENE_EXTRACTION_PROMPT)


if __name__ == "__main__":
    unittest.main()
