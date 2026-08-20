from __future__ import annotations

import base64
from io import BytesIO
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.scene_context_extraction.graph_core.prompt import SCENE_EXTRACTION_PROMPT, USER_MESSAGE
from src.scene_context_extraction.ondevice.extractor import (
    MINISTRAL3,
    QWEN3_VL,
    extract_scene_graph,
    graph_extraction_config,
    image_content_for_model,
    ministral_image_data_url,
    pad_ministral_image,
    run_graph_vlm_inference,
)
from src.scene_context_extraction.ondevice.pipeline import (
    SceneContextJob,
    init_ministral3_model,
    init_vlm_model,
    model_family_from_config,
    run_scene_context_job,
    run_scene_context_job_parallel,
)


class OnDeviceVlmBackendTests(unittest.TestCase):
    def test_model_family_defaults_to_qwen_and_rejects_unknown_values(self) -> None:
        self.assertEqual(model_family_from_config({}), QWEN3_VL)
        self.assertEqual(model_family_from_config({"MODEL_FAMILY": MINISTRAL3}), MINISTRAL3)
        with self.assertRaisesRegex(ValueError, "unsupported MODEL_FAMILY"):
            model_family_from_config({"MODEL_FAMILY": "other"})

    def test_graph_config_keeps_generation_settings_and_model_family_separate(self) -> None:
        config = graph_extraction_config(
            {
                "MODEL_FAMILY": MINISTRAL3,
                "max_new_tokens": 99,
                "do_sample": False,
            }
        )

        self.assertEqual(
            config,
            {
                "model_family": MINISTRAL3,
                "max_new_tokens": 99,
                "do_sample": False,
            },
        )

    def test_ministral_padding_preserves_source_pixels_and_adds_black_rows(self) -> None:
        source = Image.new("RGB", (672, 384), color=(11, 22, 33))

        padded = pad_ministral_image(source)

        self.assertEqual(source.size, (672, 384))
        self.assertEqual(padded.size, (672, 392))
        self.assertEqual(padded.crop((0, 4, 672, 388)).tobytes(), source.tobytes())
        self.assertEqual(padded.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(padded.getpixel((671, 391)), (0, 0, 0))

    def test_ministral_payload_is_in_memory_png_and_rejects_wrong_source_size(self) -> None:
        source = Image.new("RGB", (672, 384), color=(1, 2, 3))

        data_url = ministral_image_data_url(source)
        encoded = data_url.removeprefix("data:image/png;base64,")
        with Image.open(BytesIO(base64.b64decode(encoded))) as payload_image:
            self.assertEqual(payload_image.size, (672, 392))
        self.assertEqual(
            image_content_for_model([source], MINISTRAL3),
            [{"type": "image", "base64": data_url}],
        )
        with self.assertRaisesRegex(ValueError, "must be 672x384"):
            pad_ministral_image(Image.new("RGB", (671, 384)))

    def test_qwen_payload_keeps_existing_pil_image(self) -> None:
        source = Image.new("RGB", (672, 384))
        self.assertEqual(
            image_content_for_model([source], QWEN3_VL),
            [{"type": "image", "image": source}],
        )

    def test_wrong_ministral_source_size_is_isolated_as_scene_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            Image.new("RGB", (640, 384)).save(image_path)

            observation, warnings = extract_scene_graph(
                model=None,
                processor=None,
                image_paths=[str(image_path)],
                generation_config={"model_family": MINISTRAL3},
            )

        self.assertIsNone(observation)
        self.assertEqual(len(warnings), 1)
        self.assertIn("must be 672x384", warnings[0])

    def test_ministral_loader_uses_bfloat16_without_quantization_config(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.MistralCommonBackend = mock.Mock()
        fake_transformers.Mistral3ForConditionalGeneration = mock.Mock()
        fake_transformers.MistralCommonBackend.from_pretrained.return_value = "processor"
        fake_transformers.Mistral3ForConditionalGeneration.from_pretrained.return_value = "model"

        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            model, processor = init_ministral3_model("/models/ministral-bf16")

        self.assertEqual((model, processor), ("model", "processor"))
        fake_transformers.MistralCommonBackend.from_pretrained.assert_called_once_with(
            "/models/ministral-bf16"
        )
        fake_transformers.Mistral3ForConditionalGeneration.from_pretrained.assert_called_once_with(
            "/models/ministral-bf16",
            device_map="auto",
            dtype=fake_torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.assertNotIn(
            "quantization_config",
            fake_transformers.Mistral3ForConditionalGeneration.from_pretrained.call_args.kwargs,
        )

    def test_loader_dispatch_applies_fc_patch_only_to_qwen(self) -> None:
        with mock.patch(
            "src.scene_context_extraction.ondevice.pipeline.init_qwen3vl_model",
            return_value=("qwen-model", "qwen-processor"),
        ) as qwen, mock.patch(
            "src.scene_context_extraction.ondevice.pipeline.init_ministral3_model",
            return_value=("mistral-model", "mistral-processor"),
        ) as ministral:
            self.assertEqual(
                init_vlm_model("qwen-path", QWEN3_VL),
                ("qwen-model", "qwen-processor"),
            )
            self.assertEqual(
                init_vlm_model("mistral-path", MINISTRAL3),
                ("mistral-model", "mistral-processor"),
            )

        qwen.assert_called_once_with("qwen-path", use_fc_patch=True)
        ministral.assert_called_once_with("mistral-path")

    def test_ministral_inference_preserves_per_image_sizes_and_prompt(self) -> None:
        bfloat16 = object()
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = bfloat16

        class Tensor:
            def __init__(self, values) -> None:
                self.values = values
                self.device = None
                self.dtype = None

            def to(self, *, device=None, dtype=None):
                self.device = device
                if dtype is not None:
                    self.dtype = dtype
                return self

            def __iter__(self):
                return iter(self.values)

            def tolist(self):
                return self.values

        class Inputs(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

        class Processor:
            def __init__(self) -> None:
                self.messages = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return Inputs(
                    input_ids=Tensor([[1, 2]]),
                    attention_mask=Tensor([[1, 1]]),
                    pixel_values=Tensor(None),
                    image_sizes=Tensor([[392, 672], [392, 672]]),
                )

            def batch_decode(self, sequences, **kwargs):
                return ['{"scene_type":"unknown"}']

        class Model:
            device = "cpu"

            def __init__(self) -> None:
                self.inputs = None

            def generate(self, **kwargs):
                self.inputs = kwargs
                return [[1, 2, 3]]

        processor = Processor()
        model = Model()
        images = [Image.new("RGB", (672, 384)), Image.new("RGB", (672, 384))]

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = run_graph_vlm_inference(
                model=model,
                processor=processor,
                images=images,
                generation_config={"max_new_tokens": 10, "do_sample": False},
                model_family=MINISTRAL3,
            )

        self.assertEqual(result.text, '{"scene_type":"unknown"}')
        self.assertEqual(result.token_ids, (3,))
        self.assertEqual(result.generated_tokens, 1)
        self.assertFalse(result.reached_max_new_tokens)
        self.assertFalse(result.repetition_detected)
        self.assertEqual(processor.messages[0]["content"][0]["text"], SCENE_EXTRACTION_PROMPT)
        self.assertEqual(processor.messages[1]["content"][-1]["text"], USER_MESSAGE)
        self.assertEqual(len(processor.messages[1]["content"][:-1]), 2)
        self.assertIs(model.inputs["pixel_values"].dtype, bfloat16)
        self.assertEqual(model.inputs["image_sizes"].tolist(), [[392, 672], [392, 672]])
        self.assertEqual(model.inputs["max_new_tokens"], 10)
        self.assertNotIn("model_family", model.inputs)

    def test_force_reextracts_complete_state_in_serial_and_parallel_paths(self) -> None:
        class InProcessWorkerPool:
            gpus = ["0"]

            def run_tasks(self, tasks):
                for task in tasks:
                    Path(task["final_output_path"]).write_text(
                        json.dumps(
                            {
                                "scene_idx": task["scene_order"],
                                "keyframes": [0],
                                "vlm_visual_graph": {"scene_type": "new-parallel"},
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    Path(task["failure_output_path"]).write_text("", encoding="utf-8")
                return [
                    {
                        **task,
                        "ok": True,
                        "output_jsonl": task["final_output_path"],
                        "failure_jsonl": task["failure_output_path"],
                        "summary": {"failed": 0, "warnings": 0},
                    }
                    for task in tasks
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (672, 384)).save(frames / "0000.png")
            ref_jsonl = root / "ref.jsonl"
            ref_jsonl.write_text(
                json.dumps({"scene_idx": 0, "timeline": [{"timestamp": 0}]}) + "\n",
                encoding="utf-8",
            )
            timestamps = root / "timestamps.json"
            timestamps.write_text(
                json.dumps([{"keyframe_timestamps": [0]}]),
                encoding="utf-8",
            )

            def make_job(name: str) -> SceneContextJob:
                output = root / f"{name}.jsonl"
                output.write_text(
                    json.dumps(
                        {
                            "scene_idx": 0,
                            "keyframes": [0],
                            "vlm_visual_graph": {"scene_type": "old"},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return SceneContextJob(
                    content_id=name,
                    ref_jsonl=str(ref_jsonl),
                    scene_context_jsonl=str(output),
                    frames_dir=str(frames),
                    timestamp_json=str(timestamps),
                )

            serial_job = make_job("serial")
            with mock.patch(
                "src.scene_context_extraction.ondevice.extractor.extract_scene_graph",
                return_value=({"scene_type": "new-serial"}, []),
            ) as serial_inference:
                run_scene_context_job(serial_job, None, None, force=True)

            parallel_job = make_job("parallel")
            run_scene_context_job_parallel(
                parallel_job,
                InProcessWorkerPool(),
                force=True,
            )

            serial_record = json.loads(Path(serial_job.scene_context_jsonl).read_text(encoding="utf-8"))
            parallel_record = json.loads(Path(parallel_job.scene_context_jsonl).read_text(encoding="utf-8"))

        serial_inference.assert_called_once()
        self.assertEqual(serial_record["vlm_visual_graph"]["scene_type"], "new-serial")
        self.assertEqual(parallel_record["vlm_visual_graph"]["scene_type"], "new-parallel")


if __name__ == "__main__":
    unittest.main()
