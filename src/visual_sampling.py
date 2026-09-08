from __future__ import annotations

from fractions import Fraction
import math
from typing import Any


def validate_sampling(scene_duration: int, num_keyframes: int) -> None:
    for name, value in (("scene_duration", scene_duration), ("num_keyframes", num_keyframes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_keyframes > scene_duration * 10:
        raise ValueError("num_keyframes exceeds the 0.1-second timestamp precision")


def truncate_timestamp(value: Any) -> int | float:
    """Floor a non-negative timestamp to tenths without binary-float rounding."""
    timestamp = value if isinstance(value, Fraction) else Fraction(str(value))
    if timestamp < 0:
        raise ValueError("timestamp must be non-negative")
    ticks = math.floor(timestamp * 10)
    return ticks // 10 if ticks % 10 == 0 else ticks / 10


def timestamp_stem(value: Any) -> str:
    timestamp = truncate_timestamp(value)
    ticks = int(Fraction(str(timestamp)) * 10)
    seconds, tenth = divmod(ticks, 10)
    return f"{seconds:04d}" + (f"_{tenth}" if tenth else "")


def build_fixed_windows(
    video_duration: object, *, scene_duration: int, num_keyframes: int,
) -> list[dict[str, Any]]:
    validate_sampling(scene_duration, num_keyframes)
    duration = Fraction(str(video_duration))
    if duration <= 0:
        raise ValueError("video_duration must be a positive finite number")
    interval = Fraction(scene_duration, num_keyframes)
    windows: list[dict[str, Any]] = []
    for scene_start in range(0, math.ceil(duration), scene_duration):
        scene_end = min(scene_start + scene_duration, duration)
        boundaries = [
            scene_start + index * interval
            for index in range(num_keyframes)
            if scene_start + index * interval < scene_end
        ]
        # Keep full sampling bins; only clip the final bin to the actual video end.
        keyframes = list(dict.fromkeys(
            truncate_timestamp((start + min(start + interval, scene_end)) / 2)
            for start in boundaries
        ))
        windows.append({
            "scene_start": scene_start,
            "scene_end": float(scene_end),
            "duration": float(scene_end - scene_start),
            "shot_change_timestamps": [float(start) for start in boundaries],
            "keyframe_timestamps": keyframes,
        })
    return windows
