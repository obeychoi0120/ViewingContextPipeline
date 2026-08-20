from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..graph_adapter.json_repair import (
    extract_json,
    preprocess_raw_text,
)
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
from src.video_data_collection.raw_pipeline import shot_interval_from_config

from .api import GaussApiClient


@dataclass(frozen=True)
class GaussGenerationResult:
    text: str
    generated_tokens: int
    max_tokens: int
    finish_reason: str
    generation_seconds: float

    @property
    def reached_max_tokens(self) -> bool:
        return self.finish_reason == "length"


def extract_visual_graphs(
    client: GaussApiClient,
    frame_save_folder: str,
    scenes: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    timestamp_json_path: str | None = None,
    final_output_path: str | None = None,
    content_id: str = "",
    failure_output_path: str | None = None,
    on_scene_complete: Callable[[], None] | None = None,
    on_generation_complete: Callable[[int, float], None] | None = None,
) -> dict[str, int | float]:
    config = config or {}
    scene_timestamps = load_scene_timestamps(timestamp_json_path)
    generation_config = graph_extraction_config(config)
    summary: dict[str, int | float] = {
        "failed": 0,
        "warnings": 0,
        "generated_tokens": 0,
        "generation_seconds": 0.0,
    }

    def record_generation(generation: GaussGenerationResult) -> None:
        summary["generated_tokens"] += generation.generated_tokens
        summary["generation_seconds"] += generation.generation_seconds
        if on_generation_complete is not None:
            on_generation_complete(
                generation.generated_tokens,
                generation.generation_seconds,
            )

    output_file = (
        open_jsonl_writer(final_output_path) if final_output_path else None
    )
    failure_file = (
        open_jsonl_writer(failure_output_path) if failure_output_path else None
    )
    try:
        for fallback_idx, item in enumerate(scenes):
            raw_output_text: list[str] = []
            keyframes = normalize_keyframe_timestamps(
                get_keyframe_timestamps(
                    item,
                    scene_timestamps,
                    fallback_idx,
                )
            )
            image_paths = select_scene_image_paths(
                frames_dir=frame_save_folder,
                item=item,
                scene_timestamps=scene_timestamps,
                fallback_idx=fallback_idx,
            )
            observation, warnings = extract_scene_graph(
                client=client,
                image_paths=image_paths,
                content_id=content_id,
                scene=item,
                multimodal=bool(config.get("multimodal", False)),
                generation_config=generation_config,
                max_entities=config.get("max_entities", 5),
                on_json_repair_failure=raw_output_text.append,
                on_generation_complete=record_generation,
            )
            if len(image_paths) != len(keyframes):
                warnings.insert(
                    0,
                    f"Found {len(image_paths)} of {len(keyframes)} keyframe images",
                )
            summary["failed"] += observation is None
            summary["warnings"] += len(warnings)
            scene_idx = item.get(
                "scene_idx",
                item.get("scene_id", fallback_idx),
            )
            if observation is None:
                error_message = (
                    "; ".join(warnings).strip() or "scene extraction failed"
                )
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
                    build_graph_record(
                        item,
                        fallback_idx,
                        keyframes,
                        observation,
                    ),
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
    client: GaussApiClient,
    image_paths: list[str],
    content_id: str,
    generation_config: dict[str, Any],
    scene: dict[str, Any] | None = None,
    multimodal: bool = False,
    max_entities: int = 5,
    on_json_repair_failure: Callable[[str], None] | None = None,
    on_generation_complete: Callable[[GaussGenerationResult], None]
    | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not image_paths:
        return None, ["No keyframe images found for scene"]

    try:
        references = shot_references(scene or {}) if multimodal else []
        if multimodal:
            validate_image_reference_alignment(len(image_paths), references)
        generation = run_gauss_api_inference(
            client=client,
            image_paths=image_paths,
            content_id=content_id,
            generation_config=generation_config,
            shot_reference_records=references,
        )
        if on_generation_complete is not None:
            on_generation_complete(generation)
        preprocessed = preprocess_raw_text(
            generation.text,
            max_entities=max_entities,
        )
        raw_observation = extract_json(preprocessed)
        if raw_observation is None:
            if on_json_repair_failure is not None:
                on_json_repair_failure(generation.text)
            return None, [generation_failure_message(generation)]

        if isinstance(raw_observation.get("entities"), list):
            entities = raw_observation["entities"]
            if len(entities) > max_entities:
                warnings.append(
                    f"Truncated entities from {len(entities)} to {max_entities}"
                )
                kept_ids = {
                    entity["local_id"]
                    for entity in entities[:max_entities]
                    if isinstance(entity, dict) and "local_id" in entity
                }
                raw_observation["entities"] = entities[:max_entities]
                for entity in raw_observation["entities"]:
                    if not isinstance(entity, dict):
                        continue
                    relations = entity.get("relations")
                    if not isinstance(relations, dict):
                        continue
                    interacts_with = relations.get("INTERACTS_WITH")
                    if interacts_with and interacts_with not in kept_ids:
                        del relations["INTERACTS_WITH"]

        observation, validation_warnings = validate_observation(raw_observation)
        warnings.extend(validation_warnings)
        return observation, warnings
    except Exception as exc:
        return None, [f"inference_exception: {type(exc).__name__}: {exc}"]


def run_gauss_api_inference(
    client: GaussApiClient,
    image_paths: list[str],
    content_id: str,
    generation_config: dict[str, Any],
    shot_reference_records: list[dict[str, Any]] | None = None,
) -> GaussGenerationResult:
    image_urls = client.image_urls(
        image_paths,
        mode=shot_interval_from_config(generation_config),
        content_id=content_id,
    )
    max_tokens = int(generation_config.get("max_new_tokens", 1536))
    references = shot_reference_records or []
    result = client.generate(
        image_urls,
        system_prompt=(
            SCENE_EXTRACTION_PROMPT + "\n\n" + MULTIMODAL_USER_MESSAGE
            if references
            else SCENE_EXTRACTION_PROMPT
        ),
        user_message=MULTIMODAL_USER_MESSAGE if references else USER_MESSAGE,
        max_tokens=max_tokens,
        temperature=float(generation_config.get("temperature", 0.0)),
        top_p=float(generation_config.get("top_p", 0.95)),
        top_k=int(generation_config.get("top_k", 20)),
        repetition_penalty=float(
            generation_config.get("repetition_penalty", 1.0)
        ),
        shot_reference_texts=[shot_reference_text(item) for item in references],
    )
    return GaussGenerationResult(
        text=result.text,
        generated_tokens=result.generated_tokens,
        max_tokens=max_tokens,
        finish_reason=result.finish_reason,
        generation_seconds=result.generation_seconds,
    )


def generation_failure_message(generation: GaussGenerationResult) -> str:
    if generation.reached_max_tokens:
        return (
            "max_tokens_reached: "
            f"generated_tokens={generation.generated_tokens} "
            f"max_tokens={generation.max_tokens}"
        )
    return f"json_repair_failed: generated_tokens={generation.generated_tokens}"


def graph_extraction_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "shot_interval": shot_interval_from_config(config),
        "max_new_tokens": config.get("max_new_tokens", 1536),
        "temperature": config.get("temperature", 0.0),
        "top_p": config.get("top_p", 0.95),
        "top_k": config.get("top_k", 20),
        "repetition_penalty": config.get("repetition_penalty", 1.0),
    }
