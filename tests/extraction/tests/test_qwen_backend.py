from PIL import Image

from viewing_context_pipeline.extraction.scene_context_extraction.ondevice.extractor import (
    graph_extraction_config,
    image_content_for_model,
)


def test_qwen_payload_has_native_image_parts() -> None:
    image = Image.new("RGB", (16, 16))
    assert image_content_for_model([image]) == [{"type": "image", "image": image}]


def test_qwen_generation_config_has_no_model_router() -> None:
    assert graph_extraction_config({"max_new_tokens": 32}) == {
        "max_new_tokens": 32,
        "do_sample": False,
    }
