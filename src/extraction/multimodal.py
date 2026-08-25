from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from extraction.data_preparation.fixed30 import file_sha256


def shot_references(scene: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = scene.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("multimodal scene timeline must be a non-empty list")
    references: list[dict[str, Any]] = []
    previous = -1
    for index, shot in enumerate(timeline):
        location = f"timeline[{index}]"
        if not isinstance(shot, dict):
            raise ValueError(f"{location} must be an object")
        timestamp = shot.get("timestamp")
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError(f"{location}.timestamp must be a non-negative integer")
        if timestamp <= previous:
            raise ValueError("multimodal timeline timestamps must be strictly increasing")
        previous = timestamp
        raw_asr = shot.get("raw_asr")
        raw_ocr = shot.get("raw_ocr")
        if not isinstance(raw_asr, str) or not isinstance(raw_ocr, str):
            raise ValueError(f"{location}.raw_asr and raw_ocr must be strings")
        references.append(
            {
                "kind": "shot_reference",
                "timestamp_seconds": timestamp,
                "asr_text": raw_asr,
                "ocr_text": raw_ocr,
            }
        )
    return references


def shot_reference_text(reference: dict[str, Any]) -> str:
    return json.dumps(reference, ensure_ascii=False, separators=(",", ":"))


def validate_image_reference_alignment(
    image_count: int,
    references: list[dict[str, Any]],
) -> None:
    if image_count != len(references):
        raise ValueError(
            "multimodal scene requires one shot_reference per keyframe: "
            f"images={image_count}, references={len(references)}"
        )


def prepare_multimodal_evidence(
    video_path: str | Path,
    timestamp_json_path: str | Path,
    output_dir: str | Path,
    *,
    asr_model: str | None = "small",
    language: str | None = None,
    ocr_model_root: str | Path | None = None,
    ocr_max_chars: int = 1000,
    force: bool = False,
) -> dict[str, Any]:
    """Create ASR/OCR evidence and a fixed-30s multimodal scene timeline."""

    video = Path(video_path)
    timestamps = Path(timestamp_json_path)
    output = Path(output_dir)
    if not video.is_file() or not timestamps.is_file():
        raise FileNotFoundError("video and fixed-30s timestamp JSON are required")
    if asr_model is None and ocr_model_root is None:
        raise ValueError("enable at least one of ASR or OCR")
    if ocr_max_chars <= 0:
        raise ValueError("ocr_max_chars must be positive")
    output.mkdir(parents=True, exist_ok=True)
    timeline_path = output / "multimodal_timeline.jsonl"
    manifest_path = output / "manifest.json"
    inputs = {
        "video": file_sha256(video),
        "timestamps": file_sha256(timestamps),
        "asr_model": asr_model,
        "language": language,
        "ocr_model_root": str(ocr_model_root) if ocr_model_root else None,
        "ocr_max_chars": ocr_max_chars,
    }
    input_fingerprint = _fingerprint(inputs)
    current = _read_json(manifest_path)
    if (
        not force
        and current
        and current.get("input_fingerprint") == input_fingerprint
        and timeline_path.is_file()
    ):
        return current

    asr_path = output / "asr_words.json"
    ocr_intervals_path = output / "ocr_intervals.json"
    if asr_model is not None:
        from extraction.data_preparation.audio_extractor import extract_audio
        from extraction.data_preparation.asr_processor import process_asr

        audio_path = output / "audio.wav"
        extract_audio(video, audio_path)
        process_asr(
            audio_path,
            asr_path,
            model_name=asr_model,
            lang=language,
        )
        if not asr_path.is_file() or not asr_path.stat().st_size:
            raise RuntimeError("ASR did not produce an output")
    else:
        _write_json(asr_path, [])

    if ocr_model_root is not None:
        from extraction.data_preparation.ocr_processor import process_ocr
        from extraction.data_preparation.video_processor import extract_frames

        frame_dir = output / ".ocr_frames"
        raw_ocr_path = output / "ocr_frames.json"
        shutil.rmtree(frame_dir, ignore_errors=True)
        try:
            extract_frames(video, frame_dir)
            process_ocr(
                frame_dir,
                raw_ocr_path,
                model_root=ocr_model_root,
            )
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
        intervalize_ocr(raw_ocr_path, ocr_intervals_path)
    else:
        _write_json(ocr_intervals_path, [])

    scenes = json.loads(timestamps.read_text(encoding="utf-8"))
    asr_words = json.loads(asr_path.read_text(encoding="utf-8"))
    ocr_intervals = json.loads(ocr_intervals_path.read_text(encoding="utf-8"))
    records = merge_scene_evidence(
        scenes,
        asr_words,
        ocr_intervals,
        ocr_max_chars=ocr_max_chars,
    )
    _write_jsonl(timeline_path, records)
    manifest = {
        "schema_version": "multimodal-evidence/v1",
        "status": "complete",
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": _fingerprint(records),
        "timeline_path": str(timeline_path),
        "scene_count": len(records),
    }
    _write_json(manifest_path, manifest)
    return manifest


def intervalize_ocr(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_gap: float = 2.0,
    similarity_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    frames = json.loads(Path(input_path).read_text(encoding="utf-8"))
    completed: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for frame in frames:
        timestamp = frame["frame_time"]
        remaining: list[dict[str, Any]] = []
        for track in active:
            if timestamp - track["last_seen_time"] > max_gap:
                completed.append(track)
            else:
                remaining.append(track)
        active = remaining
        for raw_text in frame.get("texts", []):
            normalized = _clean_text(raw_text)
            if not normalized:
                continue
            best_index = -1
            best_ratio = -1.0
            for index, track in enumerate(active):
                ratio = difflib.SequenceMatcher(
                    None, normalized, track["clean_text"]
                ).ratio()
                if normalized in track["clean_text"] or track["clean_text"] in normalized:
                    ratio = max(ratio, 0.9)
                if ratio > best_ratio:
                    best_index, best_ratio = index, ratio
            if best_index >= 0 and best_ratio >= similarity_threshold:
                track = active[best_index]
                track["last_seen_time"] = timestamp
                track["end_time"] = timestamp
                if len(raw_text) > len(track["text"]):
                    track["text"] = raw_text
                    track["clean_text"] = normalized
            else:
                active.append(
                    {
                        "start_time": timestamp,
                        "end_time": timestamp,
                        "last_seen_time": timestamp,
                        "text": raw_text,
                        "clean_text": normalized,
                    }
                )
    completed.extend(active)
    result = [
        {
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "texts": [row["text"]],
        }
        for row in sorted(completed, key=lambda row: row["start_time"])
    ]
    _write_json(Path(output_path), result)
    return result


def merge_scene_evidence(
    scenes: list[dict[str, Any]],
    asr_words: list[dict[str, Any]],
    ocr_intervals: list[dict[str, Any]],
    *,
    ocr_max_chars: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene_idx, scene in enumerate(scenes):
        start = float(scene["scene_start"])
        end = float(scene["scene_end"])
        boundaries = sorted(set([start, *scene.get("shot_change_timestamps", []), end]))
        keyframes = [int(round(float(value))) for value in scene.get("keyframe_timestamps", [])]
        if len(keyframes) != len(boundaries) - 1 or len(set(keyframes)) != len(keyframes):
            raise ValueError(f"scene {scene_idx} has invalid fixed-30s keyframes")
        timeline: list[dict[str, Any]] = []
        for shot_idx, (shot_start, shot_end) in enumerate(zip(boundaries, boundaries[1:])):
            words = [
                str(word["word"])
                for word in asr_words
                if max(shot_start, float(word["start"]))
                < min(shot_end, float(word["end"]))
            ]
            ocr_values: list[str] = []
            for interval in ocr_intervals:
                if max(shot_start, float(interval["start_time"])) < min(
                    shot_end, float(interval["end_time"]) + 1.0
                ):
                    ocr_values.extend(str(value) for value in interval.get("texts", []))
            ocr_text = ", ".join(_deduplicate(ocr_values))
            if len(ocr_text) > ocr_max_chars:
                keep = max(1, (ocr_max_chars - 16) // 2)
                ocr_text = f"{ocr_text[:keep]} ... [snip] ... {ocr_text[-keep:]}"
            timeline.append(
                {
                    "shot_idx": shot_idx,
                    "timestamp": keyframes[shot_idx],
                    "raw_asr": " ".join(words),
                    "raw_ocr": ocr_text,
                }
            )
        result.append({"scene_idx": scene_idx, "timeline": timeline})
    if not result:
        raise ValueError("multimodal evidence contains no scenes")
    return result


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(value))).strip().casefold()


def _deduplicate(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _clean_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(value.strip())
    return result


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
