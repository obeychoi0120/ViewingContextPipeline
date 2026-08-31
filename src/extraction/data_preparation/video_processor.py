from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

from extraction.image_validation import verified_image_size


def _last_decodable_frame_timestamp_seconds(video_path: Path) -> float:
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
    frame_timestamps = []
    for line in completed.stdout.splitlines():
        try:
            timestamp = float(line.strip())
        except ValueError:
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            frame_timestamps.append(timestamp)
    if not frame_timestamps:
        detail = completed.stderr.strip() or "no decodable video frames found"
        raise RuntimeError(f"Could not determine last decodable frame: {video_path}: {detail}")
    return max(frame_timestamps)


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
    if not timestamps or timestamps != sorted(set(timestamps)) or any(type(value) is not int or value < 0 for value in timestamps):
        raise ValueError("timestamps must be sorted unique non-negative integers")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for timestamp in timestamps:
            destination = staging / f"{timestamp:04d}.png"
            filter_graph = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            def extract_at(seek_timestamp: int | float) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(seek_timestamp), "-i", str(source), "-vf", filter_graph, "-frames:v", "1", str(destination)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )

            completed = extract_at(timestamp)
            if completed.returncode != 0:
                raise RuntimeError(f"Failed to extract keyframe at {timestamp}s: {completed.stderr.strip()}")
            image_size_actual = verified_image_size(destination)
            if image_size_actual is None:
                last_timestamp = _last_decodable_frame_timestamp_seconds(source)
                if last_timestamp < timestamp:
                    destination.unlink(missing_ok=True)
                    completed = extract_at(last_timestamp)
                    if completed.returncode != 0:
                        raise RuntimeError(f"Failed to clamp keyframe at {timestamp}s to {last_timestamp}s: {completed.stderr.strip()}")
                    image_size_actual = verified_image_size(destination)
                    print(
                        f"[Info] Clamped trailing keyframe at {timestamp}s "
                        f"to last decodable frame at {last_timestamp}s."
                    )
            if image_size_actual is None:
                raise RuntimeError(f"Could not read extracted keyframe: {destination}")
            if image_size_actual != (width, height):
                raise RuntimeError(f"Invalid keyframe dimensions: {destination}")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
