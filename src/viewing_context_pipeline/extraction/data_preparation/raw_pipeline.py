from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from viewing_context_pipeline.extraction.common.manifest import validate_manifest_rows


DEFAULT_CONFIG: dict[str, Any] = {
    "shot_interval": "fixed_30s",
    "ocr_config": {
        "enabled": False,
        "OCR_MIN_HEIGHT": 0,
        "OCR_MAX_CHARS": 1000,
        "OCR_SCORE_THR": 0.6,
        "OCR_FILTER_UI": True,
        "OCR_TOP_UI_REGION_Y": 0.1,
        "OCR_RIGHT_CHAT_REGION_X": 0.8,
        "OCR_SMALL_TEXT_HEIGHT_RATIO": 0.1,
    },
    "asr_config": {
        "enabled": False,
        "BEAM_SIZE": 5,
        "CONDITION_ON_PREVIOUS_TEXT": False,
        "NO_SPEECH_THRESHOLD": 0.8,
        "VAD_THRESHOLD": 0.25,
        "VAD_MIN_SILENCE_MS": 1000,
        "VAD_SPEECH_PAD_MS": 300,
        "INITIAL_PROMPT": None,
        "REF_MODEL_PATH": "/home_nvme/shared/models/faster-whisper-large-v3",
    },
    "resized_keyframe_config": {
        "image_resolution": [672, 384],
    },
}

SHOT_INTERVAL = "fixed_30s"
FIXED_30S_SCENE_SECONDS = 30
FIXED_30S_REFERENCE_SECONDS = 10
FIXED_30S_FRAMES_PER_SCENE = 3
DATA_PREPARATION_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "extraction" / "data_preparation.json"

OCR_CONFIG_KEYS = (
    "OCR_MIN_HEIGHT",
    "OCR_MAX_CHARS",
    "OCR_SCORE_THR",
    "OCR_FILTER_UI",
    "OCR_TOP_UI_REGION_Y",
    "OCR_RIGHT_CHAT_REGION_X",
    "OCR_SMALL_TEXT_HEIGHT_RATIO",
)

@dataclass(frozen=True)
class RawPipelinePaths:
    content_id: str
    save_path: str
    assets_path: str
    resized_keyframes_dir: str
    metadata_json: str
    multimodal_ref: str
    timestamp_json: str
    filtered_timestamp_json: str


def build_content_paths(
    data_root: str | Path,
    content_id: str,
    output_root: str | Path | None = None,
    shot_interval: str = SHOT_INTERVAL,
) -> RawPipelinePaths:
    shot_interval = normalize_shot_interval(shot_interval)
    safe_content_id = normalize_content_id(content_id)
    save_path = posix_path(data_root, safe_content_id)
    assets_path = posix_path(save_path, "assets")
    output_path = output_save_path(output_root)
    return RawPipelinePaths(
        content_id=safe_content_id,
        save_path=save_path,
        assets_path=assets_path,
        resized_keyframes_dir=posix_path(
            output_path,
            resized_keyframes_relative_path(shot_interval),
            safe_content_id,
        ),
        metadata_json=posix_path(
            output_path, "data", "cohort", "metadata", f"{safe_content_id}.json"
        ),
        multimodal_ref=posix_path(
            output_path,
            multimodal_ref_relative_path(shot_interval),
            f"{safe_content_id}_multimodal_ref.jsonl",
        ),
        timestamp_json=posix_path(assets_path, "timestamp.json"),
        filtered_timestamp_json=posix_path(
            assets_path,
            timestamp_filename(shot_interval),
        ),
    )


def posix_path(*parts: str | Path) -> str:
    normalized_parts = [str(part).replace("\\", "/") for part in parts if str(part)]
    return str(PurePosixPath(*normalized_parts))


def output_save_path(output_root: str | Path | None = None) -> Path:
    if output_root:
        return Path(output_root)
    value = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not value:
        raise ValueError("OUTPUT_SAVE_PATH is required in config/.env or the environment")
    return Path(value)


def selected_filtered_timestamp_json(paths: RawPipelinePaths) -> str:
    return paths.filtered_timestamp_json


def normalize_shot_interval(value: object) -> str:
    mode = str(value or "").strip()
    if mode != SHOT_INTERVAL:
        raise ValueError(f"shot_interval must be {SHOT_INTERVAL}; got {value!r}")
    return mode


def shot_interval_from_config(config: dict[str, Any]) -> str:
    return normalize_shot_interval(config.get("shot_interval", DEFAULT_CONFIG["shot_interval"]))


def shot_interval_output_dirname(shot_interval: str) -> str:
    mode = normalize_shot_interval(shot_interval)
    return mode


def resized_keyframes_relative_path(shot_interval: str) -> str:
    return posix_path(
        "data", shot_interval_output_dirname(shot_interval), "resized_keyframes"
    )


def multimodal_ref_relative_path(shot_interval: str) -> str:
    return posix_path("data", shot_interval_output_dirname(shot_interval), "multimodal_ref")


def multimodal_from_config(config: dict[str, Any]) -> bool:
    if "multimodal" not in config:
        raise ValueError("multimodal must be explicitly set to true or false")
    value = config["multimodal"]
    if type(value) is not bool:
        raise ValueError("multimodal must be a boolean")
    return value


def modality_dirname(multimodal: bool) -> str:
    if type(multimodal) is not bool:
        raise ValueError("multimodal must be a boolean")
    return "multimodal" if multimodal else "img_only"


def viewing_context_relative_path(multimodal: bool, shot_interval: str) -> str:
    return posix_path(
        "viewing_context",
        modality_dirname(multimodal),
        shot_interval_output_dirname(shot_interval),
    )


def configured_shot_interval() -> str:
    return shot_interval_from_config(load_processing_config(DATA_PREPARATION_CONFIG_PATH))


def configured_resized_keyframes_relative_path() -> str:
    return resized_keyframes_relative_path(configured_shot_interval())


def configured_multimodal_ref_relative_path() -> str:
    return multimodal_ref_relative_path(configured_shot_interval())


def timestamp_filename(shot_interval: str) -> str:
    mode = normalize_shot_interval(shot_interval)
    return f"timestamp_{mode}.json"


def configured_timestamp_filename() -> str:
    return timestamp_filename(configured_shot_interval())


def normalize_content_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("content_id/name is required")
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    normalized = "".join(keep).strip("._")
    if not normalized:
        raise ValueError(f"content_id has no filesystem-safe characters: {text}")
    return normalized


def row_value(row: dict[str, object], target_key: str) -> object:
    normalized_target = normalize_row_key(target_key)
    for key, value in row.items():
        if normalize_row_key(str(key)) == normalized_target:
            return value
    return ""


def normalize_row_key(key: str) -> str:
    return key.strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_")


def load_processing_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if not config_path:
        raise ValueError("data preparation config path is required")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"data preparation config not found: {path}")
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    return deep_merge(config, user_config)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def grouped_config(config: dict[str, Any], group_name: str, keys: tuple[str, ...]) -> dict[str, Any]:
    group = dict(config.get(group_name, {}))
    for key in keys:
        if key in config:
            group[key] = config[key]
    return group


def process_local_source(
    *,
    name: str,
    source_video_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    metadata: dict[str, object],
    config_path: str | Path = DATA_PREPARATION_CONFIG_PATH,
    frames_per_window: int = FIXED_30S_FRAMES_PER_SCENE,
    lang: str | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Process a caller-owned local video without mutating the source file."""

    source_path = Path(source_video_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"local video source not found: {source_path}")
    config = load_processing_config(config_path)
    shot_interval = shot_interval_from_config(config)
    normalize_shot_interval(shot_interval)
    if frames_per_window != FIXED_30S_FRAMES_PER_SCENE:
        raise ValueError("fixed_30s requires frames_per_window=3")
    paths = build_content_paths(
        data_root=data_root,
        content_id=name,
        output_root=output_root,
        shot_interval=shot_interval,
    )
    Path(paths.assets_path).mkdir(parents=True, exist_ok=True)
    write_metadata_json(metadata_from_row(metadata, paths.content_id), paths.metadata_json)

    from . import video_processor

    contract = {
        "schema_version": "local-video-processing-contract/v3",
        "source_sha256": file_sha256(source_path),
        "shot_interval": shot_interval,
        "frames_per_window": frames_per_window,
        "frame_source": "original_video",
        "keyframe_extractor": (
            f"direct-selected-timestamps/v{video_processor.DIRECT_KEYFRAME_EXTRACTION_VERSION}"
        ),
        "keyframe_resize": "contain_pad_black",
        "keyframe_resolution": list(resized_keyframe_resolution(config)),
        "asr_enabled": config.get("asr_config", {}).get("enabled") is True,
        "ocr_enabled": config.get("ocr_config", {}).get("enabled") is True,
    }
    contract.update({
        "sampling": fixed_sampling_contract(shot_interval, frames_per_window),
        "keyframe_offsets_seconds": [5, 15, 25],
        "ocr_sampling_fps": 1,
        "ocr_interval_max_gap_seconds": 2.0,
        "ocr_interval_similarity_threshold": 0.75,
        "ocr_dedup_similarity_threshold": 0.8,
        "ocr_max_chars": config.get("ocr_config", {}).get("OCR_MAX_CHARS", 1000),
    })
    contract_path = processing_contract_path(paths, shot_interval)
    if force or read_json_object(contract_path) != contract:
        (Path(paths.save_path) / f"{paths.content_id}.wav").unlink(missing_ok=True)
        invalidate_pts_dependent_artifacts(paths, modes=(shot_interval,))

    result = process_prepared_source(
        paths=paths,
        config=config,
        lang=lang,
        frames_per_window=frames_per_window,
        direct_video_path=source_path,
    )
    write_json_atomic(contract_path, contract)
    return result


def process_prepared_source(
    *,
    paths: RawPipelinePaths,
    config: dict[str, Any],
    lang: str | None,
    frames_per_window: int = FIXED_30S_FRAMES_PER_SCENE,
    direct_video_path: str | Path,
) -> dict[str, str]:
    """Run fixed-30s keyframe, ASR/OCR, and multimodal-reference stages."""

    from . import data_processor, video_processor

    shot_interval = shot_interval_from_config(config)
    if frames_per_window != FIXED_30S_FRAMES_PER_SCENE:
        raise ValueError("fixed_30s requires frames_per_window=3")
    asr_config = config.get("asr_config", {})

    selected_timestamp_json = selected_filtered_timestamp_json(paths)
    image_size = resized_keyframe_resolution(config)
    media_path = str(direct_video_path)
    video_duration = max(1, math.ceil(video_processor.get_video_duration_seconds(media_path)))
    direct_complete = resized_keyframes_match_timestamps(selected_timestamp_json, paths.resized_keyframes_dir, image_size)
    if not direct_complete:
        invalidate_pts_dependent_artifacts(paths, modes=(shot_interval,))
        write_fixed_interval_timestamps(selected_timestamp_json, video_duration, frames_per_window=frames_per_window, shot_interval=shot_interval)
        video_processor.extract_resized_keyframes(media_path, selected_keyframe_timestamps(selected_timestamp_json), paths.resized_keyframes_dir, image_size)

    asr_enabled = asr_config.get("enabled", False) is True
    audio_path = str(Path(paths.save_path) / f"{paths.content_id}.wav")
    asr_words_path = str(Path(paths.assets_path) / "ASR_words.json")
    ref_model_path = str(asr_config.get("REF_MODEL_PATH") or DEFAULT_CONFIG["asr_config"]["REF_MODEL_PATH"])
    if asr_enabled and not Path(audio_path).exists():
        from . import audio_extractor

        audio_extractor.extract_audio(media_path, out_audio_path=audio_path)
    if asr_enabled:
        from . import asr_processor

        common_asr_kwargs = {
            "lang": lang,
            "beam_size": asr_config.get("BEAM_SIZE", 5),
            "condition_on_previous_text": asr_config.get("CONDITION_ON_PREVIOUS_TEXT", False),
            "no_speech_threshold": asr_config.get("NO_SPEECH_THRESHOLD", 0.8),
            "vad_threshold": asr_config.get("VAD_THRESHOLD", 0.25),
            "vad_min_silence_duration_ms": asr_config.get("VAD_MIN_SILENCE_MS", 1000),
            "vad_speech_pad_ms": asr_config.get("VAD_SPEECH_PAD_MS", 300),
            "initial_prompt": asr_config.get("INITIAL_PROMPT"),
        }
        if not is_nonempty_file(asr_words_path):
            Path(asr_words_path).unlink(missing_ok=True)
            asr_processor.process_asr(
                audio_path,
                asr_words_path,
                model_name=ref_model_path,
                **common_asr_kwargs,
            )
            require_nonempty_asr_output(asr_words_path)

    ocr_frames_path, ocr_cleaned_path = ocr_artifact_paths(paths, shot_interval)
    ocr_config = grouped_config(config, "ocr_config", OCR_CONFIG_KEYS)
    ocr_enabled = ocr_config.get("enabled", False) is True

    if ocr_enabled and not Path(ocr_frames_path).exists():
        from . import ocr_processor

        ocr_processor.patch_paddlex_predictor()
        ocr_kwargs = {
            "score_thr": ocr_config.get("OCR_SCORE_THR", 0.6),
            "min_height": ocr_config.get("OCR_MIN_HEIGHT", 0),
            "filter_ui": ocr_config.get("OCR_FILTER_UI", True),
            "top_ui_region_y": ocr_config.get("OCR_TOP_UI_REGION_Y", 0.1),
            "right_chat_region_x": ocr_config.get("OCR_RIGHT_CHAT_REGION_X", 0.8),
            "small_text_height_ratio": ocr_config.get("OCR_SMALL_TEXT_HEIGHT_RATIO", 0.1),
        }
        ocr_frames_dir = Path(paths.save_path) / ".ocr_frames_fixed_30s"
        shutil.rmtree(ocr_frames_dir, ignore_errors=True)
        try:
            video_processor.extract_frames(media_path, ocr_frames_dir)
            ocr_processor.process_ocr(
                str(ocr_frames_dir),
                ocr_frames_path,
                use_server_detector=True,
                **ocr_kwargs,
            )
        finally:
            shutil.rmtree(ocr_frames_dir, ignore_errors=True)

    if ocr_enabled and not Path(ocr_cleaned_path).exists():
        data_processor.intervalize_ocr(ocr_frames_path, ocr_cleaned_path)

    multimodal = config.get("multimodal", False)
    if type(multimodal) is not bool:
        raise ValueError("multimodal must be a boolean")
    if multimodal:
        Path(paths.multimodal_ref).parent.mkdir(parents=True, exist_ok=True)
        data_processor.merge_scene_data(
            timestamp_file=selected_timestamp_json,
            asr_words_file=asr_words_path if asr_enabled else "",
            ocr_cleaned_file=ocr_cleaned_path if ocr_enabled else "",
            output_file=paths.multimodal_ref,
            video_duration=video_duration,
            ocr_max_length=ocr_config.get("OCR_MAX_CHARS", 1000),
        )
    else:
        Path(paths.multimodal_ref).unlink(missing_ok=True)
    return manifest_row(paths)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def processing_contract_path(
    paths: RawPipelinePaths, shot_interval: str
) -> Path:
    mode = normalize_shot_interval(shot_interval)
    return Path(paths.assets_path) / f"processing_contract_{mode}.json"


def ocr_artifact_paths(
    paths: RawPipelinePaths, shot_interval: str
) -> tuple[str, str]:
    mode = normalize_shot_interval(shot_interval)
    root = Path(paths.assets_path) / mode
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "OCR_frames.json"), str(root / "OCR_cleaned.json")


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def is_nonempty_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.is_file() and file_path.stat().st_size > 0


def require_nonempty_asr_output(asr_output_path: str | Path) -> None:
    if not is_nonempty_file(asr_output_path):
        raise RuntimeError(f"ASR processing did not create a non-empty output: {asr_output_path}")


def invalidate_pts_dependent_artifacts(
    paths: RawPipelinePaths,
    *,
    modes: tuple[str, ...] = (SHOT_INTERVAL,),
) -> None:
    output_root = Path(paths.metadata_json).parents[3]
    normalized_modes = tuple(normalize_shot_interval(mode) for mode in modes)
    for mode in normalized_modes:
        shutil.rmtree(
            output_root / resized_keyframes_relative_path(mode) / paths.content_id,
            ignore_errors=True,
        )
        (
            output_root
            / multimodal_ref_relative_path(mode)
            / f"{paths.content_id}_multimodal_ref.jsonl"
        ).unlink(missing_ok=True)

    assets_path = Path(paths.assets_path)
    stale_files = [Path(paths.timestamp_json)]
    stale_files.extend(assets_path / timestamp_filename(mode) for mode in normalized_modes)
    stale_files.extend(assets_path / mode / name for mode in normalized_modes for name in ("OCR_frames.json", "OCR_cleaned.json"))
    for stale_file in stale_files:
        stale_file.unlink(missing_ok=True)



def read_metadata_json(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_from_row(row: dict[str, object], content_id: str) -> dict[str, Any]:
    metadata = {key: value for key, value in row.items()}
    metadata["title"] = str(metadata.get("title") or "")
    metadata["tags"] = str(metadata.get("tags") or "")
    metadata["content_id"] = content_id
    return metadata


def resized_keyframe_resolution(config: dict[str, Any]) -> tuple[int, int]:
    resized_config = config.get("resized_keyframe_config", {})
    resolution = resized_config.get("image_resolution") if isinstance(resized_config, dict) else None
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError(f"resized_keyframe_config.image_resolution must be [width, height], got {resolution!r}")
    return int(resolution[0]), int(resolution[1])


def build_fixed_interval_windows(
    video_duration: int,
    frames_per_window: int = FIXED_30S_FRAMES_PER_SCENE,
    shot_interval: str = SHOT_INTERVAL,
) -> list[dict[str, Any]]:
    duration = int(video_duration)
    if duration <= 0:
        raise ValueError(f"video_duration must be positive, got {video_duration!r}")

    if (
        isinstance(frames_per_window, bool)
        or not isinstance(frames_per_window, int)
        or frames_per_window <= 0
    ):
        raise ValueError("frames_per_window must be a positive integer")
    normalize_shot_interval(shot_interval)
    if frames_per_window != FIXED_30S_FRAMES_PER_SCENE:
        raise ValueError("fixed_30s requires frames_per_window=3")
    return build_fixed_30s_windows(duration)


def build_fixed_30s_windows(video_duration: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for scene_start in range(0, video_duration, FIXED_30S_SCENE_SECONDS):
        scene_end = min(scene_start + FIXED_30S_SCENE_SECONDS, video_duration)
        boundaries = list(
            range(scene_start, scene_end, FIXED_30S_REFERENCE_SECONDS)
        )
        keyframes = [
            (reference_start + min(
                reference_start + FIXED_30S_REFERENCE_SECONDS,
                scene_end,
            ))
            // 2
            for reference_start in boundaries
        ]
        windows.append(
            {
                "scene_start": scene_start,
                "scene_end": scene_end,
                "duration": scene_end - scene_start,
                "shot_change_timestamps": boundaries,
                "keyframe_timestamps": keyframes,
            }
        )
    return windows


def fixed_sampling_contract(
    shot_interval: str, frames_per_window: int
) -> dict[str, Any]:
    normalize_shot_interval(shot_interval)
    if frames_per_window != FIXED_30S_FRAMES_PER_SCENE:
        raise ValueError("fixed_30s requires frames_per_window=3")
    return {
        "scene_seconds": FIXED_30S_SCENE_SECONDS,
        "reference_seconds": FIXED_30S_REFERENCE_SECONDS,
        "frames_per_scene": FIXED_30S_FRAMES_PER_SCENE,
        "keyframe_position": "reference_midpoint_floor_seconds",
    }


def write_fixed_interval_timestamps(
    output_path: str | Path,
    video_duration: int,
    frames_per_window: int = FIXED_30S_FRAMES_PER_SCENE,
    shot_interval: str = SHOT_INTERVAL,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            build_fixed_interval_windows(
                video_duration,
                frames_per_window,
                shot_interval=shot_interval,
            ),
            f,
            indent=4,
            ensure_ascii=False,
        )


def selected_keyframe_timestamps(timestamp_file: str | Path) -> list[int]:
    with Path(timestamp_file).open("r", encoding="utf-8") as f:
        scenes = json.load(f)
    if not isinstance(scenes, list):
        raise ValueError(f"filtered timestamp JSON must contain a list: {timestamp_file}")

    timestamps: list[int] = []
    seen: set[int] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError(f"invalid scene entry in filtered timestamp JSON: {timestamp_file}")
        for timestamp in scene.get("keyframe_timestamps", []):
            timestamp_sec = int(round(float(timestamp)))
            if timestamp_sec not in seen:
                timestamps.append(timestamp_sec)
                seen.add(timestamp_sec)
    if not timestamps:
        raise ValueError(f"filtered timestamp JSON has no keyframes: {timestamp_file}")
    return timestamps


def resized_keyframes_match_timestamps(
    timestamp_file: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int],
) -> bool:
    from PIL import Image

    try:
        expected_names = {
            f"{timestamp:04d}.png"
            for timestamp in selected_keyframe_timestamps(timestamp_file)
        }
        output_path = Path(output_dir)
        if not output_path.is_dir():
            return False
        actual_names = {
            path.name
            for path in output_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }
        if actual_names != expected_names:
            return False
        for name in expected_names:
            with Image.open(output_path / name) as image:
                if image.size != image_size:
                    return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def write_metadata_json(metadata: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def manifest_row(paths: RawPipelinePaths) -> dict[str, str]:
    return {"content_id": paths.content_id}


def write_manifest(rows: list[dict[str, str]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [
        {"content_id": str(row.get("content_id") or "").strip()}
        for row in rows
    ]
    normalized_rows = validate_manifest_rows(
        normalized_rows,
        source=str(output_path),
    )
    fieldnames = ["content_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
