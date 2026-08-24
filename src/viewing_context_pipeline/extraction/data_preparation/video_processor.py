from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import av
import cv2
from tqdm import tqdm


FRAME_EXTRACTION_VERSION = 3
FRAME_EXTRACTION_METADATA_FILENAME = ".pts_extraction.json"
DIRECT_KEYFRAME_EXTRACTION_VERSION = 2


def get_video_duration_seconds(video_path: str | Path) -> float:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video source is missing: {path}")
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        duration = float(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Could not determine video duration: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Video duration must be positive: {path}")
    return duration


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
            completed = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(timestamp), "-i", str(source), "-vf", filter_graph, "-frames:v", "1", str(destination)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Failed to extract keyframe at {timestamp}s: {completed.stderr.strip()}")
            image = cv2.imread(str(destination))
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"Invalid keyframe dimensions: {destination}")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def extract_frames(video_path: str | Path, output_folder: str | Path) -> None:
    """Extract the nearest display-oriented frame for every integer PTS second."""

    source = Path(video_path)
    output = Path(output_folder)
    staging = output.with_name(f".{output.name}.pts_tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    container = av.open(str(source))
    saved = 0
    target = 0
    first_time = None
    previous = None
    previous_time = None
    try:
        stream = container.streams.video[0]
        with tqdm(desc="[Video] Extracting by PTS", unit="frames") as progress:
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise RuntimeError(f"Decoded frame has no PTS: {source}")
                current = float(frame.pts * frame.time_base)
                if first_time is None:
                    first_time = current
                current -= first_time
                while target <= current:
                    selected = frame
                    if previous is not None and target - previous_time <= current - target:
                        selected = previous
                    cv2.imwrite(str(staging / f"{target:04d}.png"), selected.reformat(format="bgr24").to_ndarray())
                    saved += 1
                    target += 1
                    progress.update(1)
                previous = frame
                previous_time = current
    finally:
        container.close()
    if not saved:
        shutil.rmtree(staging)
        raise RuntimeError(f"No frames extracted: {source}")
    source_stat = source.stat()
    (staging / FRAME_EXTRACTION_METADATA_FILENAME).write_text(
        json.dumps({"version": FRAME_EXTRACTION_VERSION, "source_size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns, "duration_seconds": target - 1, "frame_count": saved}, indent=2) + "\n",
        encoding="utf-8",
    )
    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
