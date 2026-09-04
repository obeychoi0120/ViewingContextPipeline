from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

from extraction.image_validation import verified_image_size


def _decodable_frame_tail_timestamps_seconds(video_path: Path) -> tuple[float, float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    last_timestamp: float | None = None
    previous_timestamp: float | None = None
    for line in completed.stdout.splitlines():
        try:
            timestamp = float(line.strip())
        except ValueError:
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            if last_timestamp is None or timestamp > last_timestamp:
                previous_timestamp, last_timestamp = last_timestamp, timestamp
            elif timestamp < last_timestamp and (
                previous_timestamp is None or timestamp > previous_timestamp
            ):
                previous_timestamp = timestamp
    if last_timestamp is None:
        detail = completed.stderr.strip() or "no decodable video frames found"
        raise RuntimeError(f"Could not determine last decodable frame: {video_path}: {detail}")
    # Seeking at the exact final timestamp can round beyond EOF in ffmpeg's input
    # time base. The preceding distinct frame leaves one decodable frame after
    # the seek point; a single-frame video is safely sought from zero.
    safe_seek_timestamp = previous_timestamp if previous_timestamp is not None else 0.0
    return safe_seek_timestamp, last_timestamp


def _last_decodable_frame_timestamp_seconds(video_path: Path) -> float:
    return _decodable_frame_tail_timestamps_seconds(video_path)[1]


def extract_resized_keyframes(
    video_path: str | Path,
    timestamps: list[int],
    output_folder: str | Path,
    image_size: tuple[int, int],
) -> None:
    source = Path(video_path)
    output = Path(output_folder)
    staging = output.with_name(f".{output.name}.direct_tmp")
    width, height = image_size
    if not source.is_file():
        raise FileNotFoundError(f"Video source is missing: {source}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size!r}")
    if (
        not timestamps
        or timestamps != sorted(set(timestamps))
        or any(type(value) is not int or value < 0 for value in timestamps)
    ):
        raise ValueError("timestamps must be sorted unique non-negative integers")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for timestamp in timestamps:
            destination = staging / f"{timestamp:04d}.png"
            filter_graph = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            clamp_message: str | None = None

            def extract_at(seek_timestamp: int | float) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        str(seek_timestamp),
                        "-i",
                        str(source),
                        "-vf",
                        filter_graph,
                        "-frames:v",
                        "1",
                        str(destination),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )

            completed = extract_at(timestamp)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Failed to extract keyframe at {timestamp}s: {completed.stderr.strip()}"
                )
            image_size_actual = verified_image_size(destination)
            if image_size_actual is None:
                safe_timestamp, last_timestamp = _decodable_frame_tail_timestamps_seconds(source)
                if last_timestamp < timestamp:
                    destination.unlink(missing_ok=True)
                    completed = extract_at(safe_timestamp)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"Failed to clamp keyframe at {timestamp}s using safe seek "
                            f"{safe_timestamp}s before last frame {last_timestamp}s: "
                            f"{completed.stderr.strip()}"
                        )
                    image_size_actual = verified_image_size(destination)
                    clamp_message = (
                        f"[Info] Clamped trailing keyframe at {timestamp}s "
                        f"using safe seek at {safe_timestamp}s before last decodable "
                        f"frame at {last_timestamp}s."
                    )
            if image_size_actual is None:
                raise RuntimeError(f"Could not read extracted keyframe: {destination}")
            if image_size_actual != (width, height):
                raise RuntimeError(f"Invalid keyframe dimensions: {destination}")
            if clamp_message is not None:
                print(clamp_message, flush=True)
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
