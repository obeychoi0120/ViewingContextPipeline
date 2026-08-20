from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
from typing import Any, Callable

from PIL import Image, ImageOps

from ..graph_adapter.json_repair import extract_json, preprocess_raw_text
from ..graph_adapter.payload import (
    get_keyframe_timestamps,
    load_scene_timestamps,
    normalize_keyframe_timestamps,
    select_scene_image_paths,
)
from ..graph_adapter.writer import (
    build_graph_record,
    open_jsonl_writer,
    write_graph_record,
)
from ..graph_core.prompt import SCENE_EXTRACTION_PROMPT, USER_MESSAGE
from ..graph_core.multimodal import (
    MULTIMODAL_USER_MESSAGE,
    shot_reference_text,
    shot_references,
    validate_image_reference_alignment,
)
from ..graph_core.scene_failures import build_scene_failure_record
from ..graph_core.validator import validate_observation


QWEN3_VL = "qwen3_vl"
MINISTRAL3 = "ministral3"
MINISTRAL_SOURCE_SIZE = (672, 384)
MINISTRAL_INPUT_SIZE = (672, 392)
MIN_REPETITION_BLOCK_TOKENS = 8
MAX_REPETITION_BLOCK_TOKENS = 64
MIN_REPETITION_REPEATS = 3


@dataclass(frozen=True)
class VlmGenerationResult:
    text: str
    token_ids: tuple[int, ...]
    generated_tokens: int
    max_new_tokens: int
    reached_max_new_tokens: bool
    generation_seconds: float
    repetition_block_tokens: int = 0
    repetition_repeats: int = 0

    @property
    def repetition_detected(self) -> bool:
        return self.repetition_repeats >= MIN_REPETITION_REPEATS


def extract_visual_graphs(
    model: Any,
    processor: Any,
    frame_save_folder: str,
    scenes: list[dict[str, Any]],
    vlm_config: dict[str, Any] | None = None,
    timestamp_json_path: str | None = None,
    final_output_path: str | None = None,
    content_id: str = "",
    failure_output_path: str | None = None,
    on_scene_complete: Callable[[], None] | None = None,
    on_generation_complete: Callable[[int, float], None] | None = None,
) -> dict[str, int | float]:
    if vlm_config is None:
        vlm_config = {}

    scene_timestamps = load_scene_timestamps(timestamp_json_path)
    generation_config = graph_extraction_config(vlm_config)
    summary: dict[str, int | float] = {
        "failed": 0,
        "warnings": 0,
        "generated_tokens": 0,
        "generation_seconds": 0.0,
    }

    def record_generation(generation: VlmGenerationResult) -> None:
        summary["generated_tokens"] += generation.generated_tokens
        summary["generation_seconds"] += generation.generation_seconds
        if on_generation_complete is not None:
            on_generation_complete(
                generation.generated_tokens,
                generation.generation_seconds,
            )

    output_file = open_jsonl_writer(final_output_path) if final_output_path else None
    failure_file = (
        open_jsonl_writer(failure_output_path)
        if failure_output_path
        else None
    )
    try:
        for idx, item in enumerate(scenes):
            raw_output_text: list[str] = []
            keyframes = normalize_keyframe_timestamps(get_keyframe_timestamps(item, scene_timestamps, idx))
            image_paths = select_scene_image_paths(
                frames_dir=frame_save_folder,
                item=item,
                scene_timestamps=scene_timestamps,
                fallback_idx=idx,
            )
            observation, warnings = extract_scene_graph(
                model=model,
                processor=processor,
                image_paths=image_paths,
                scene=item,
                multimodal=bool(vlm_config.get("multimodal", False)),
                generation_config=generation_config,
                max_entities=vlm_config.get("max_entities", 5),
                on_json_repair_failure=raw_output_text.append,
                on_generation_complete=record_generation,
            )
            if len(image_paths) != len(keyframes):
                warnings.insert(0, f"Found {len(image_paths)} of {len(keyframes)} keyframe images")
            summary["failed"] += observation is None
            summary["warnings"] += len(warnings)
            if observation is None:
                scene_idx = item.get(
                    "scene_idx",
                    item.get("scene_id", idx),
                )
                error_message = "; ".join(warnings).strip() or "scene extraction failed"
                console_error = " ".join(error_message.splitlines())
                print(
                    f"[scene-error] content_id={content_id} "
                    f"scene_idx={scene_idx} error={console_error}",
                    flush=True,
                )
                if failure_file:
                    write_graph_record(
                        failure_file,
                        build_scene_failure_record(
                            content_id=content_id,
                            scene_idx=scene_idx,
                            keyframes=keyframes,
                            error=error_message,
                            raw_output_text=(
                                raw_output_text[0]
                                if raw_output_text
                                else None
                            ),
                        ),
                    )
            elif output_file:
                write_graph_record(
                    output_file,
                    build_graph_record(item, idx, keyframes, observation),
                )
            if on_scene_complete is not None:
                on_scene_complete()
    finally:
        if output_file:
            output_file.close()
        if failure_file:
            failure_file.close()

    return summary


def extract_scene_graph(
    model: Any,
    processor: Any,
    image_paths: list[str],
    generation_config: dict[str, Any],
    scene: dict[str, Any] | None = None,
    multimodal: bool = False,
    max_entities: int = 5,
    on_json_repair_failure: Callable[[str], None] | None = None,
    on_generation_complete: Callable[[VlmGenerationResult], None] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not image_paths:
        return None, ["No keyframe images found for scene"]

    try:
        images = load_images(image_paths)
        references = shot_references(scene or {}) if multimodal else []
        if multimodal:
            validate_image_reference_alignment(len(images), references)
        inference_config = dict(generation_config)
        model_family = str(inference_config.pop("model_family", QWEN3_VL))
        generation = run_graph_vlm_inference(
            model=model,
            processor=processor,
            images=images,
            shot_reference_records=references,
            generation_config=inference_config,
            model_family=model_family,
        )
        if on_generation_complete is not None:
            on_generation_complete(generation)
        # Pre-process: truncate entities array to max_entities before JSON parsing
        # This prevents truncation failures when the model generates many entities
        preprocessed = preprocess_raw_text(generation.text, max_entities=max_entities)
        raw_observation = extract_json(preprocessed)
        if raw_observation is None:
            if on_json_repair_failure is not None:
                on_json_repair_failure(generation.text)
            return None, [generation_failure_message(generation)]

        # Post-processing: force-limit entities to max_entities (safety net)
        if isinstance(raw_observation.get("entities"), list):
            entities = raw_observation["entities"]
            if len(entities) > max_entities:
                warnings.append(
                    f"Truncated entities from {len(entities)} to {max_entities}"
                )
                kept_ids = set()
                for ent in entities[:max_entities]:
                    if isinstance(ent, dict) and "local_id" in ent:
                        kept_ids.add(ent["local_id"])
                raw_observation["entities"] = entities[:max_entities]
                for ent in raw_observation["entities"]:
                    if isinstance(ent, dict) and isinstance(ent.get("relations"), dict):
                        iw = ent["relations"].get("INTERACTS_WITH")
                        if iw and iw not in kept_ids:
                            del ent["relations"]["INTERACTS_WITH"]

        observation, validation_warnings = validate_observation(raw_observation)
        warnings.extend(validation_warnings)
        return observation, warnings
    except Exception as exc:
        return None, [f"inference_exception: {type(exc).__name__}: {exc}"]


def load_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def run_graph_vlm_inference(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    generation_config: dict[str, Any],
    model_family: str = QWEN3_VL,
    shot_reference_records: list[dict[str, Any]] | None = None,
) -> VlmGenerationResult:
    image_content = image_content_for_model(images, model_family)
    references = shot_reference_records or []
    if references:
        validate_image_reference_alignment(len(image_content), references)
        user_content: list[dict[str, Any]] = []
        for image_part, reference in zip(image_content, references):
            user_content.append(image_part)
            user_content.append(
                {"type": "text", "text": shot_reference_text(reference)}
            )
        user_content.append({"type": "text", "text": MULTIMODAL_USER_MESSAGE})
        system_prompt = SCENE_EXTRACTION_PROMPT + "\n\n" + MULTIMODAL_USER_MESSAGE
    else:
        user_content = image_content
        user_content.append({"type": "text", "text": USER_MESSAGE})
        system_prompt = SCENE_EXTRACTION_PROMPT
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model(inputs, model, model_family)
    generation_started_at = perf_counter()
    generated = model.generate(
        **inputs,
        **generation_config,
    )
    generation_seconds = perf_counter() - generation_started_at
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    token_ids = token_ids_as_tuple(trimmed[0])
    max_new_tokens = int(generation_config.get("max_new_tokens", 1536))
    repetition_block_tokens, repetition_repeats = detect_repeated_token_suffix(
        token_ids
    )
    return VlmGenerationResult(
        text=text,
        token_ids=token_ids,
        generated_tokens=len(token_ids),
        max_new_tokens=max_new_tokens,
        reached_max_new_tokens=len(token_ids) >= max_new_tokens,
        generation_seconds=generation_seconds,
        repetition_block_tokens=repetition_block_tokens,
        repetition_repeats=repetition_repeats,
    )


def token_ids_as_tuple(sequence: Any) -> tuple[int, ...]:
    values = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
    return tuple(int(token_id) for token_id in values)


def detect_repeated_token_suffix(
    token_ids: tuple[int, ...],
) -> tuple[int, int]:
    max_block_tokens = min(
        MAX_REPETITION_BLOCK_TOKENS,
        len(token_ids) // MIN_REPETITION_REPEATS,
    )
    best_match: tuple[int, int, int] | None = None
    best_result = (0, 0)
    for block_tokens in range(MIN_REPETITION_BLOCK_TOKENS, max_block_tokens + 1):
        for partial_tokens in range(block_tokens):
            complete_end = len(token_ids) - partial_tokens
            if complete_end < block_tokens * MIN_REPETITION_REPEATS:
                continue
            block = token_ids[complete_end - block_tokens:complete_end]
            if (
                partial_tokens
                and token_ids[complete_end:] != block[:partial_tokens]
            ):
                continue

            repeats = 1
            cursor = complete_end - block_tokens
            while (
                cursor >= block_tokens
                and token_ids[cursor - block_tokens:cursor] == block
            ):
                repeats += 1
                cursor -= block_tokens
            if repeats < MIN_REPETITION_REPEATS:
                continue

            covered_tokens = repeats * block_tokens + partial_tokens
            match = (covered_tokens, repeats, -block_tokens)
            if best_match is None or match > best_match:
                best_match = match
                best_result = (block_tokens, repeats)
    return best_result


def generation_failure_message(generation: VlmGenerationResult) -> str:
    if generation.repetition_detected:
        return (
            "repetition_detected: "
            f"generated_tokens={generation.generated_tokens} "
            f"max_new_tokens={generation.max_new_tokens} "
            f"block_tokens={generation.repetition_block_tokens} "
            f"repeats={generation.repetition_repeats}"
        )
    if generation.reached_max_new_tokens:
        return (
            "max_tokens_reached: "
            f"generated_tokens={generation.generated_tokens} "
            f"max_new_tokens={generation.max_new_tokens}"
        )
    return f"json_repair_failed: generated_tokens={generation.generated_tokens}"


def image_content_for_model(
    images: list[Image.Image],
    model_family: str,
) -> list[dict[str, Any]]:
    if model_family == QWEN3_VL:
        return [{"type": "image", "image": image} for image in images]
    if model_family == MINISTRAL3:
        return [
            {"type": "image", "base64": ministral_image_data_url(image)}
            for image in images
        ]
    raise ValueError(f"unsupported MODEL_FAMILY: {model_family!r}")


def pad_ministral_image(image: Image.Image) -> Image.Image:
    rgb_image = image.convert("RGB")
    if rgb_image.size != MINISTRAL_SOURCE_SIZE:
        raise ValueError(
            "Ministral input image must be 672x384 before padding, "
            f"got {rgb_image.size[0]}x{rgb_image.size[1]}"
        )
    padded = ImageOps.expand(rgb_image, border=(0, 4, 0, 4), fill=(0, 0, 0))
    if padded.size != MINISTRAL_INPUT_SIZE:
        raise AssertionError(f"unexpected Ministral input size: {padded.size}")
    return padded


def ministral_image_data_url(image: Image.Image) -> str:
    with BytesIO() as buffer:
        pad_ministral_image(image).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def move_inputs_to_model(
    inputs: Any,
    model: Any,
    model_family: str,
) -> Any:
    if model_family == QWEN3_VL:
        return inputs.to(model.device)
    if model_family != MINISTRAL3:
        raise ValueError(f"unsupported MODEL_FAMILY: {model_family!r}")

    import torch

    for key, value in list(inputs.items()):
        if not hasattr(value, "to"):
            continue
        if key == "pixel_values":
            inputs[key] = value.to(device=model.device, dtype=torch.bfloat16)
        elif key != "image_sizes":
            inputs[key] = value.to(device=model.device)
    return inputs


def graph_extraction_config(vlm_config: dict[str, Any]) -> dict[str, Any]:
    do_sample = vlm_config.get("do_sample", False)
    config = {
        "max_new_tokens": vlm_config.get("max_new_tokens", 1536),
        "do_sample": do_sample,
        "model_family": vlm_config.get("MODEL_FAMILY", QWEN3_VL),
    }
    return config
