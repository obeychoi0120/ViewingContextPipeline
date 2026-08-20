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
from urllib.parse import parse_qs, urlparse

from src.common.manifest import validate_manifest_rows
from src.common.output_paths import custom_output_root


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
    "scene_config": {
        "MIN_LENGTH": 20.0,
        "SPLIT_THRESOLD": 60.0,
        "MAX_LENGTH": 80.0,
        "MAX_SHOTS": 5,
        "MIN_SHOT_OFFSET": 5.0,
    },
    "resized_keyframe_config": {
        "image_resolution": [672, 384],
    },
}

SHOT_INTERVAL_MODES = frozenset({"shot_wise", "fixed_15s", "fixed_30s"})
FIXED_INTERVAL_SECONDS = 15
FIXED_FRAMES_PER_WINDOW = 4
FIXED_30S_SCENE_SECONDS = 30
FIXED_30S_REFERENCE_SECONDS = 10
FIXED_30S_FRAMES_PER_SCENE = 3
VIDEO_DATA_COLLECTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "video_data_collection.json"

OCR_CONFIG_KEYS = (
    "OCR_MIN_HEIGHT",
    "OCR_MAX_CHARS",
    "OCR_SCORE_THR",
    "OCR_FILTER_UI",
    "OCR_TOP_UI_REGION_Y",
    "OCR_RIGHT_CHAT_REGION_X",
    "OCR_SMALL_TEXT_HEIGHT_RATIO",
)

URL_KEYS = (
    "url",
    "video_url",
    "video url",
    "youtube_url",
    "youtube url",
    "video_link",
    "video link",
    "link",
    "href",
    "webpage_url",
    "webpage url",
    "watch_url",
    "watch url",
)
@dataclass(frozen=True)
class RawPipelinePaths:
    content_id: str
    url: str
    save_path: str
    assets_path: str
    video_path: str
    video_480p_path: str
    all_frames_dir: str
    resized_keyframes_dir: str
    metadata_json: str
    ref_jsonl: str
    timestamp_json: str
    filtered_timestamp_json: str


def build_content_paths(
    data_root: str | Path,
    content_id: str,
    url: str = "",
    output_root: str | Path | None = None,
    shot_interval: str = "fixed_15s",
) -> RawPipelinePaths:
    shot_interval = normalize_shot_interval(shot_interval)
    safe_content_id = normalize_content_id(content_id)
    save_path = posix_path(data_root, safe_content_id)
    assets_path = posix_path(save_path, "assets")
    output_path = output_save_path(output_root)
    return RawPipelinePaths(
        content_id=safe_content_id,
        url=url,
        save_path=save_path,
        assets_path=assets_path,
        video_path=posix_path(save_path, f"{safe_content_id}.mp4"),
        video_480p_path=posix_path(save_path, f"{safe_content_id}_480p.mp4"),
        all_frames_dir=posix_path(save_path, "all_frames"),
        resized_keyframes_dir=posix_path(
            output_path,
            resized_keyframes_relative_path(shot_interval),
            safe_content_id,
        ),
        metadata_json=posix_path(
            output_path, "asset", "metadata", f"{safe_content_id}.json"
        ),
        ref_jsonl=posix_path(
            output_path,
            ref_jsonl_relative_path(shot_interval),
            f"{safe_content_id}_ref.jsonl",
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
    return custom_output_root(value)


def selected_filtered_timestamp_json(paths: RawPipelinePaths) -> str:
    return paths.filtered_timestamp_json


def normalize_shot_interval(value: object) -> str:
    mode = str(value or "").strip()
    if mode not in SHOT_INTERVAL_MODES:
        supported = ", ".join(sorted(SHOT_INTERVAL_MODES))
        raise ValueError(f"shot_interval must be one of: {supported}; got {value!r}")
    return mode


def shot_interval_from_config(config: dict[str, Any]) -> str:
    return normalize_shot_interval(config.get("shot_interval", DEFAULT_CONFIG["shot_interval"]))


def shot_interval_output_dirname(shot_interval: str) -> str:
    mode = normalize_shot_interval(shot_interval)
    return mode


def resized_keyframes_relative_path(shot_interval: str) -> str:
    return posix_path(
        "asset", shot_interval_output_dirname(shot_interval), "resized_keyframes"
    )


def ref_jsonl_relative_path(shot_interval: str) -> str:
    return posix_path("asset", shot_interval_output_dirname(shot_interval), "ref_jsonl")


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
    return shot_interval_from_config(load_processing_config(VIDEO_DATA_COLLECTION_CONFIG_PATH))


def configured_resized_keyframes_relative_path() -> str:
    return resized_keyframes_relative_path(configured_shot_interval())


def configured_ref_jsonl_relative_path() -> str:
    return ref_jsonl_relative_path(configured_shot_interval())


def timestamp_filename(shot_interval: str) -> str:
    mode = normalize_shot_interval(shot_interval)
    if mode == "shot_wise":
        return "timestamp_filtered.json"
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


def row_to_url(row: dict[str, object]) -> str:
    for key in URL_KEYS:
        value = str(row_value(row, key) or "").strip()
        if value:
            return value
    video_id = row_to_video_id(row)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    raise ValueError(f"batch row must contain a URL column or video_id. columns={list(row.keys())}")


def row_to_video_id(row: dict[str, object]) -> str:
    video_id = str(row_value(row, "video_id") or row_value(row, "video id") or "").strip()
    if video_id:
        return video_id_from_url(video_id) or video_id
    for key in URL_KEYS:
        parsed = video_id_from_url(str(row_value(row, key) or "").strip())
        if parsed:
            return parsed
    return ""


def row_value(row: dict[str, object], target_key: str) -> object:
    normalized_target = normalize_row_key(target_key)
    for key, value in row.items():
        if normalize_row_key(str(key)) == normalized_target:
            return value
    return ""


def normalize_row_key(key: str) -> str:
    return key.strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_")


def video_id_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0]
    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id
    path_parts = [part for part in parsed.path.split("/") if part]
    for marker in ("shorts", "embed", "live"):
        if marker in path_parts:
            index = path_parts.index(marker)
            if index + 1 < len(path_parts):
                return path_parts[index + 1]
    return ""


def load_processing_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if not config_path:
        raise ValueError("video data collection config path is required")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"video data collection config not found: {path}")
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


def process_one(
    name: str,
    url: str,
    data_root: str | Path,
    lang: str | None = None,
    config_path: str | Path | None = "config/video_data_collection.json",
    metadata: dict[str, object] | None = None,
    download_metadata: bool = True,
    write_missing_metadata: bool = True,
    bot_check_max_retries: int | None = None,
    bot_check_retry_delay_sec: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, str]:
    config = load_processing_config(config_path)
    shot_interval = shot_interval_from_config(config)
    paths = build_content_paths(
        data_root=data_root,
        content_id=name,
        url=url,
        output_root=output_root,
        shot_interval=shot_interval,
    )
    Path(paths.assets_path).mkdir(parents=True, exist_ok=True)

    from . import utils, video_processor

    utils.proxy_setup()
    load_or_create_metadata(
        paths.metadata_json,
        url,
        paths.content_id,
        metadata,
        allow_download=download_metadata,
        write_missing=write_missing_metadata,
    )

    if not Path(paths.video_480p_path).exists() and not Path(paths.video_path).exists():
        download_kwargs = {}
        if bot_check_max_retries is not None:
            download_kwargs["bot_check_max_retries"] = bot_check_max_retries
        if bot_check_retry_delay_sec is not None:
            download_kwargs["bot_check_retry_delay_sec"] = bot_check_retry_delay_sec
        video_processor.download_video(url, paths.video_path, **download_kwargs)
    ensure_canonical_480p_video(
        video_processor,
        paths.video_path,
        paths.video_480p_path,
    )

    return process_prepared_source(
        paths=paths,
        config=config,
        lang=lang,
        url=url,
    )


def process_local_source(
    *,
    name: str,
    source_video_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    metadata: dict[str, object],
    config_path: str | Path = "config/video_data_collection.json",
    frames_per_window: int = 1,
    lang: str | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Process a caller-owned local video without mutating the source file."""

    source_path = Path(source_video_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"local video source not found: {source_path}")
    config = load_processing_config(config_path)
    shot_interval = shot_interval_from_config(config)
    if shot_interval not in {"fixed_15s", "fixed_30s"}:
        raise ValueError(
            "local MicroLens processing requires shot_interval=fixed_15s or fixed_30s"
        )
    paths = build_content_paths(
        data_root=data_root,
        content_id=name,
        url=f"microlens://100k/{str(name).rsplit('_', 1)[-1].lstrip('0') or '0'}",
        output_root=output_root,
        shot_interval=shot_interval,
    )
    Path(paths.assets_path).mkdir(parents=True, exist_ok=True)
    write_metadata_json(metadata_from_row(metadata, paths.url, paths.content_id), paths.metadata_json)

    from . import video_processor

    contract = {
        "schema_version": (
            "local-video-processing-contract/v3"
            if shot_interval == "fixed_30s"
            else "local-video-processing-contract/v2"
        ),
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
    if shot_interval == "fixed_30s":
        contract.update(
            {
                "sampling": fixed_sampling_contract(
                    shot_interval, frames_per_window
                ),
                "keyframe_offsets_seconds": [5, 15, 25],
                "ocr_sampling_fps": 1,
                "ocr_interval_max_gap_seconds": 2.0,
                "ocr_interval_similarity_threshold": 0.75,
                "ocr_dedup_similarity_threshold": 0.8,
                "ocr_max_chars": config.get("ocr_config", {}).get(
                    "OCR_MAX_CHARS", 1000
                ),
            }
        )
    else:
        contract["interval_seconds"] = FIXED_INTERVAL_SECONDS
    contract_path = processing_contract_path(paths, shot_interval)
    if force or read_json_object(contract_path) != contract:
        Path(paths.video_480p_path).unlink(missing_ok=True)
        shutil.rmtree(paths.all_frames_dir, ignore_errors=True)
        (Path(paths.save_path) / f"{paths.content_id}.wav").unlink(missing_ok=True)
        invalidate_pts_dependent_artifacts(paths, modes=(shot_interval,))

    Path(paths.video_480p_path).unlink(missing_ok=True)
    shutil.rmtree(paths.all_frames_dir, ignore_errors=True)
    result = process_prepared_source(
        paths=paths,
        config=config,
        lang=lang,
        url=paths.url,
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
    url: str,
    frames_per_window: int | None = None,
    preserve_keyframe_aspect_ratio: bool = False,
    direct_video_path: str | Path | None = None,
) -> dict[str, str]:
    """Run the shared frame, ASR/OCR, timeline, and Ref JSONL stages."""

    from . import data_processor, utils, video_processor

    shot_interval = shot_interval_from_config(config)
    if frames_per_window is None:
        frames_per_window = (
            FIXED_30S_FRAMES_PER_SCENE
            if shot_interval == "fixed_30s"
            else FIXED_FRAMES_PER_WINDOW
        )
    asr_config = config.get("asr_config", {})
    scene_config = config.get("scene_config", {})

    selected_timestamp_json = selected_filtered_timestamp_json(paths)
    image_size = resized_keyframe_resolution(config)
    if direct_video_path is None:
        media_path = paths.video_480p_path
        frames_rebuilt = not video_processor.extracted_frames_are_current(
            media_path,
            paths.all_frames_dir,
        )
        if frames_rebuilt:
            invalidate_pts_dependent_artifacts(paths)
            video_processor.extract_frames(media_path, paths.all_frames_dir)
        video_duration = video_processor.get_extracted_video_duration(paths.all_frames_dir)
        should_build_timestamps = frames_rebuilt or not Path(selected_timestamp_json).exists()
        if should_build_timestamps:
            if shot_interval == "shot_wise":
                from . import scene_segmenter

                candidate_timestamps = scene_segmenter.extract_scene_timestamps(
                    media_path,
                    use_adaptive=False,
                    output_json=paths.timestamp_json,
                )
                scene_segmenter.run_scene_segmentation(
                    candidate_timestamps,
                    selected_timestamp_json,
                    min_scene_length=scene_config.get("MIN_LENGTH", 20.0),
                    split_threshold=scene_config.get("SPLIT_THRESOLD", 60.0),
                    max_scene_length=scene_config.get("MAX_LENGTH", 80.0),
                    max_shots_per_scene=scene_config.get("MAX_SHOTS", 5),
                    min_shot_length=scene_config.get("MIN_SHOT_OFFSET", 5.0),
                    video_duration=video_duration,
                )
            else:
                write_fixed_interval_timestamps(
                    selected_timestamp_json,
                    video_duration,
                    frames_per_window=frames_per_window,
                    shot_interval=shot_interval,
                )
        if Path(selected_timestamp_json).exists():
            clamp_trailing_keyframe_timestamps(
                timestamp_file=selected_timestamp_json,
                all_frames_dir=paths.all_frames_dir,
                video_duration=video_duration,
            )
            export_resized_keyframes(
                timestamp_file=selected_timestamp_json,
                all_frames_dir=paths.all_frames_dir,
                output_dir=paths.resized_keyframes_dir,
                image_size=image_size,
                preserve_aspect_ratio=preserve_keyframe_aspect_ratio,
            )
    else:
        if shot_interval not in {"fixed_15s", "fixed_30s"}:
            raise ValueError(
                "direct local-video extraction requires shot_interval=fixed_15s or fixed_30s"
            )
        media_path = str(direct_video_path)
        video_duration = max(
            1,
            math.ceil(video_processor.get_video_duration_seconds(media_path)),
        )
        direct_complete = resized_keyframes_match_timestamps(
            selected_timestamp_json,
            paths.resized_keyframes_dir,
            image_size,
        )
        if not direct_complete:
            invalidate_pts_dependent_artifacts(paths, modes=(shot_interval,))
            write_fixed_interval_timestamps(
                selected_timestamp_json,
                video_duration,
                frames_per_window=frames_per_window,
                shot_interval=shot_interval,
            )
            video_processor.extract_resized_keyframes(
                media_path,
                selected_keyframe_timestamps(selected_timestamp_json),
                paths.resized_keyframes_dir,
                image_size,
            )

    asr_enabled = asr_config.get("enabled", False) is True
    audio_path = str(Path(paths.save_path) / f"{paths.content_id}.wav")
    asr_ref_path = str(Path(paths.assets_path) / "ASR_Ref.json")
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
        if not is_nonempty_file(asr_ref_path):
            Path(asr_ref_path).unlink(missing_ok=True)
            asr_processor.process_asr(
                audio_path,
                asr_ref_path,
                model_name=ref_model_path,
                **common_asr_kwargs,
            )
            require_nonempty_asr_output(asr_ref_path)

    ocr_ref_path, ocr_ref_cleaned_path = ocr_artifact_paths(paths, shot_interval)
    ocr_config = grouped_config(config, "ocr_config", OCR_CONFIG_KEYS)
    ocr_enabled = ocr_config.get("enabled", False) is True

    if ocr_enabled and not Path(ocr_ref_path).exists():
        from . import ocr_processor

        utils.patch_paddlex_predictor()
        ocr_kwargs = {
            "score_thr": ocr_config.get("OCR_SCORE_THR", 0.6),
            "min_height": ocr_config.get("OCR_MIN_HEIGHT", 0),
            "filter_ui": ocr_config.get("OCR_FILTER_UI", True),
            "top_ui_region_y": ocr_config.get("OCR_TOP_UI_REGION_Y", 0.1),
            "right_chat_region_x": ocr_config.get("OCR_RIGHT_CHAT_REGION_X", 0.8),
            "small_text_height_ratio": ocr_config.get("OCR_SMALL_TEXT_HEIGHT_RATIO", 0.1),
        }
        if shot_interval == "fixed_30s" and direct_video_path is not None:
            ocr_frames_dir = Path(paths.save_path) / ".ocr_frames_fixed_30s"
            shutil.rmtree(ocr_frames_dir, ignore_errors=True)
            try:
                video_processor.extract_frames(media_path, ocr_frames_dir)
                ocr_processor.process_ocr(
                    str(ocr_frames_dir),
                    ocr_ref_path,
                    generate_ref=True,
                    **ocr_kwargs,
                )
            finally:
                shutil.rmtree(ocr_frames_dir, ignore_errors=True)
        else:
            ocr_frame_dir = (
                paths.all_frames_dir
                if shot_interval == "fixed_30s"
                else paths.resized_keyframes_dir
            )
            ocr_processor.process_ocr(
                ocr_frame_dir,
                ocr_ref_path,
                generate_ref=True,
                **ocr_kwargs,
            )

    if ocr_enabled and not Path(ocr_ref_cleaned_path).exists():
        data_processor.intervalize_ocr(ocr_ref_path, ocr_ref_cleaned_path)

    Path(paths.ref_jsonl).parent.mkdir(parents=True, exist_ok=True)
    data_processor.merge_scene_data(
        timestamp_file=selected_timestamp_json,
        asr_words_file=asr_ref_path if asr_enabled else "",
        ocr_cleaned_file=ocr_ref_cleaned_path if ocr_enabled else "",
        output_file=paths.ref_jsonl,
        video_duration=video_duration,
        ocr_max_length=ocr_config.get("OCR_MAX_CHARS", 1000),
    )
    remove_legacy_raw_artifacts(paths)
    write_url_file(paths.save_path, url)
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
    filename = (
        "processing_contract.json"
        if mode == "fixed_15s"
        else f"processing_contract_{mode}.json"
    )
    return Path(paths.assets_path) / filename


def ocr_artifact_paths(
    paths: RawPipelinePaths, shot_interval: str
) -> tuple[str, str]:
    mode = normalize_shot_interval(shot_interval)
    if mode == "fixed_30s":
        root = Path(paths.assets_path) / mode
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(paths.assets_path)
    return str(root / "OCR_Ref.json"), str(root / "OCR_Ref_Cleaned.json")


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def collect_video_metadata(url: str, fallback_name: str) -> dict[str, Any]:
    try:
        import yt_dlp
        from .ytdlp_utils import ytdlp_base_opts

        ydl_opts = {**ytdlp_base_opts(), "quiet": True, "skip_download": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        info = {}
    return {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_id": info.get("channel_id") or "",
        "upload_date": info.get("upload_date") or "",
        "duration": info.get("duration") or "",
        "view_count": info.get("view_count") or "",
        "description": str(info.get("description") or ""),
        "url": url,
    }


def ensure_canonical_480p_video(video_processor: Any, source_path: str | Path, canonical_path: str | Path) -> None:
    video_processor.ensure_480p_video(source_path, canonical_path)
    source = Path(source_path)
    if source.exists():
        source.unlink()


def is_nonempty_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.is_file() and file_path.stat().st_size > 0


def require_nonempty_asr_output(asr_output_path: str | Path) -> None:
    if not is_nonempty_file(asr_output_path):
        raise RuntimeError(f"ASR processing did not create a non-empty output: {asr_output_path}")


def remove_legacy_raw_artifacts(paths: RawPipelinePaths) -> None:
    assets_path = Path(paths.assets_path)
    legacy_paths = (
        assets_path / f"{paths.content_id}_raw.jsonl",
        assets_path / "ASR.json",
        assets_path / "ASR.model.json",
        assets_path / "ASR_Ref.model.json",
        assets_path / "OCR.json",
        assets_path / "OCR_Cleaned.json",
    )
    for legacy_path in legacy_paths:
        legacy_path.unlink(missing_ok=True)


def invalidate_pts_dependent_artifacts(
    paths: RawPipelinePaths,
    *,
    modes: tuple[str, ...] | frozenset[str] = SHOT_INTERVAL_MODES,
) -> None:
    output_root = Path(paths.metadata_json).parents[2]
    normalized_modes = tuple(normalize_shot_interval(mode) for mode in modes)
    for mode in normalized_modes:
        shutil.rmtree(
            output_root / resized_keyframes_relative_path(mode) / paths.content_id,
            ignore_errors=True,
        )
        (
            output_root
            / ref_jsonl_relative_path(mode)
            / f"{paths.content_id}_ref.jsonl"
        ).unlink(missing_ok=True)

    assets_path = Path(paths.assets_path)
    stale_files = [Path(paths.timestamp_json), assets_path / "timestamp_candidates_analysis.json"]
    stale_files.extend(assets_path / timestamp_filename(mode) for mode in normalized_modes)
    if "shot_wise" in normalized_modes or "fixed_15s" in normalized_modes:
        stale_files.extend(
            [assets_path / "OCR_Ref.json", assets_path / "OCR_Ref_Cleaned.json"]
        )
    if "fixed_30s" in normalized_modes:
        stale_files.extend(
            [
                assets_path / "fixed_30s" / "OCR_Ref.json",
                assets_path / "fixed_30s" / "OCR_Ref_Cleaned.json",
            ]
        )
    for stale_file in stale_files:
        stale_file.unlink(missing_ok=True)

    for mode in normalized_modes:
        (output_root / "video_profile" / mode / f"{paths.content_id}_profile.json").unlink(
            missing_ok=True
        )
        for multimodal in (False, True):
            mode_root = output_root / viewing_context_relative_path(multimodal, mode)
            failure_root = (
                output_root
                / "failures"
                / "viewing_context"
                / modality_dirname(multimodal)
                / mode
            )
            for postfix in ("qwen", "mistral", "gaussa_gemma4_e2b_v0_3", "ref"):
                scene_dir = f"scene_context_graph_{postfix}"
                (mode_root / scene_dir / f"{paths.content_id}_scene_context.jsonl").unlink(
                    missing_ok=True
                )
                (failure_root / scene_dir / f"{paths.content_id}_failures.jsonl").unlink(
                    missing_ok=True
                )
                context_dir = f"video_context_graph_{postfix}"
                suffix = "ref" if postfix == "ref" else "ond"
                (mode_root / context_dir / f"{paths.content_id}_context_graph_{suffix}.json").unlink(
                    missing_ok=True
                )


def load_or_create_metadata(
    metadata_path: str | Path,
    url: str,
    fallback_name: str,
    row_metadata: dict[str, object] | None = None,
    allow_download: bool = True,
    write_missing: bool = True,
) -> dict[str, Any]:
    path = Path(metadata_path)
    if path.exists():
        return read_metadata_json(path)

    if row_metadata is not None:
        metadata = metadata_from_row(row_metadata, url=url, fallback_name=fallback_name)
    elif allow_download:
        metadata = collect_video_metadata(url=url, fallback_name=fallback_name)
    else:
        metadata = metadata_from_row({}, url=url, fallback_name=fallback_name)
    if write_missing:
        write_metadata_json(metadata, path)
    return metadata


def read_metadata_json(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_from_row(row: dict[str, object], url: str, fallback_name: str) -> dict[str, Any]:
    metadata = {key: value for key, value in row.items()}
    metadata["title"] = str(metadata.get("title") or "")
    metadata["channel"] = str(metadata.get("channel") or row.get("channel_title") or row.get("uploader") or "")
    metadata["channel_id"] = str(metadata.get("channel_id") or "")
    metadata["upload_date"] = str(metadata.get("upload_date") or row.get("published_at") or "")
    metadata["duration"] = metadata.get("duration") or row.get("duration_sec") or ""
    metadata["view_count"] = metadata.get("view_count") or ""
    metadata["description"] = str(metadata.get("description") or "")
    metadata["url"] = url
    return metadata


def resized_keyframe_resolution(config: dict[str, Any]) -> tuple[int, int]:
    resized_config = config.get("resized_keyframe_config", {})
    resolution = resized_config.get("image_resolution") if isinstance(resized_config, dict) else None
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError(f"resized_keyframe_config.image_resolution must be [width, height], got {resolution!r}")
    return int(resolution[0]), int(resolution[1])


def build_fixed_interval_windows(
    video_duration: int,
    frames_per_window: int = FIXED_FRAMES_PER_WINDOW,
    shot_interval: str = "fixed_15s",
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
    mode = normalize_shot_interval(shot_interval)
    if mode == "shot_wise":
        raise ValueError("fixed interval windows do not support shot_interval=shot_wise")
    if mode == "fixed_30s":
        if frames_per_window != FIXED_30S_FRAMES_PER_SCENE:
            raise ValueError("fixed_30s requires frames_per_window=3")
        return build_fixed_30s_windows(duration)

    window_seconds = FIXED_INTERVAL_SECONDS * int(frames_per_window)
    windows: list[dict[str, Any]] = []
    for window_start in range(0, duration, window_seconds):
        window_end = min(window_start + window_seconds, duration)
        timestamps = list(range(window_start, window_end, FIXED_INTERVAL_SECONDS))
        windows.append(
            {
                "scene_start": window_start,
                "scene_end": window_end,
                "duration": window_end - window_start,
                "shot_change_timestamps": timestamps,
                "keyframe_timestamps": timestamps,
            }
        )
    return windows


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
    mode = normalize_shot_interval(shot_interval)
    if mode == "fixed_30s":
        return {
            "scene_seconds": FIXED_30S_SCENE_SECONDS,
            "reference_seconds": FIXED_30S_REFERENCE_SECONDS,
            "frames_per_scene": FIXED_30S_FRAMES_PER_SCENE,
            "keyframe_position": "reference_midpoint_floor_seconds",
        }
    if mode == "fixed_15s":
        return {
            "scene_seconds": FIXED_INTERVAL_SECONDS * frames_per_window,
            "reference_seconds": FIXED_INTERVAL_SECONDS,
            "frames_per_scene": frames_per_window,
            "keyframe_position": "reference_start_seconds",
        }
    raise ValueError("shot_wise does not have a fixed sampling contract")


def write_fixed_interval_timestamps(
    output_path: str | Path,
    video_duration: int,
    frames_per_window: int = FIXED_FRAMES_PER_WINDOW,
    shot_interval: str = "fixed_15s",
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


def export_resized_keyframes(
    timestamp_file: str | Path,
    all_frames_dir: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int],
    preserve_aspect_ratio: bool = False,
) -> None:
    from PIL import Image

    timestamps = selected_keyframe_timestamps(timestamp_file)

    source_path = Path(all_frames_dir)
    output_path = Path(output_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"all_frames directory not found: {source_path}")

    missing_sources = [source_path / f"{timestamp:04d}.png" for timestamp in timestamps]
    missing_sources = [source for source in missing_sources if not source.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"selected all-frame is missing: {missing_sources[0]}")

    output_path.mkdir(parents=True, exist_ok=True)

    expected_names = {f"{timestamp:04d}.png" for timestamp in timestamps}
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    for existing in output_path.iterdir():
        if not existing.is_file() or existing.suffix.lower() not in image_extensions:
            continue
        if existing.name not in expected_names:
            existing.unlink()

    for timestamp in timestamps:
        frame_name = f"{timestamp:04d}.png"
        source = source_path / frame_name
        destination = output_path / frame_name
        if destination.exists():
            try:
                with Image.open(destination) as existing:
                    if existing.size == image_size:
                        continue
            except OSError:
                pass
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            if preserve_aspect_ratio:
                rgb.thumbnail(image_size, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", image_size, "black")
                offset = (
                    (image_size[0] - rgb.width) // 2,
                    (image_size[1] - rgb.height) // 2,
                )
                canvas.paste(rgb, offset)
                canvas.save(destination)
            else:
                rgb.resize(image_size).save(destination)


def clamp_trailing_keyframe_timestamps(
    timestamp_file: str | Path,
    all_frames_dir: str | Path,
    video_duration: int,
) -> bool:
    with Path(timestamp_file).open("r", encoding="utf-8") as f:
        scenes = json.load(f)
    if not isinstance(scenes, list) or not scenes or not isinstance(scenes[-1], dict):
        return False

    frame_timestamps = [
        int(path.stem)
        for path in Path(all_frames_dir).glob("*.png")
        if path.stem.isdigit()
    ]
    if not frame_timestamps:
        return False

    last_frame_timestamp = max(frame_timestamps)
    if video_duration <= 0 or last_frame_timestamp < video_duration - 1:
        return False

    keyframes = scenes[-1].get("keyframe_timestamps")
    if not isinstance(keyframes, list):
        return False

    adjusted_keyframes: list[int] = []
    seen: set[int] = set()
    clamped_timestamps: list[int] = []
    changed = False
    for value in keyframes:
        timestamp = int(round(float(value)))
        if last_frame_timestamp < timestamp <= video_duration + 1:
            clamped_timestamps.append(timestamp)
            timestamp = last_frame_timestamp
            changed = True
        if timestamp not in seen:
            adjusted_keyframes.append(timestamp)
            seen.add(timestamp)

    if not changed:
        return False

    scenes[-1]["keyframe_timestamps"] = adjusted_keyframes
    with Path(timestamp_file).open("w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=4, ensure_ascii=False)
    print(
        f"[Info] Clamped trailing keyframe timestamp(s) {clamped_timestamps} "
        f"to last available frame {last_frame_timestamp}."
    )
    return True


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


def resized_keyframes_are_complete(
    timestamp_file: str | Path,
    all_frames_dir: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int],
) -> bool:
    try:
        expected_names = {f"{timestamp:04d}.png" for timestamp in selected_keyframe_timestamps(timestamp_file)}
        all_frames_path = Path(all_frames_dir)
        if not all_frames_path.is_dir():
            return False
        if any(not (all_frames_path / name).is_file() for name in expected_names):
            return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return resized_keyframes_match_timestamps(timestamp_file, output_dir, image_size)


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


def write_url_file(save_path: str | Path, url: str) -> None:
    path = Path(save_path) / "URL.txt"
    if not path.exists():
        path.write_text(url, encoding="utf-8")


def manifest_row(paths: RawPipelinePaths) -> dict[str, str]:
    return {
        "content_id": paths.content_id,
        "url": paths.url,
    }


def write_manifest(rows: list[dict[str, str]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [
        {
            "content_id": str(row.get("content_id") or "").strip(),
            "url": str(row.get("url") or "").strip(),
        }
        for row in rows
    ]
    normalized_rows = validate_manifest_rows(
        normalized_rows,
        source=str(output_path),
    )
    fieldnames = ["content_id", "url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
