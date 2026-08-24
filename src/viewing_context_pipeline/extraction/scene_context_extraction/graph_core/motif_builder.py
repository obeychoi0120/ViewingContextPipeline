"""
graph_v2/motif_builder.py - Deterministic motif generation.

Motifs prioritize 4D content axes, entities, settings, and activities.
Supplemental style/mood motifs are retained but no longer dominate clustering.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .scoring import bucket_content_axes
from .ontology import MOTIF_EXCLUDE_ENTITIES, MOTIF_EXCLUDE_ROLES, MOTIF_EXCLUDE_VALUES


@dataclass
class Motif:
    """A single motif node."""
    key: str
    motif_type: str
    parts: Dict[str, str]


def build_motifs(observation: Dict[str, Any]) -> List[Motif]:
    """Build deterministic motifs from a validated observation."""
    setting = observation.get("setting", "unknown")
    scene_function = observation.get("scene_function", "unknown")
    style = observation.get("style", "mixed")
    mood = observation.get("mood", "mixed")
    content_axes = observation.get("content_axes_4d", {})
    axis_buckets = bucket_content_axes(content_axes)
    entities = observation.get("entities", [])
    main_entity = _pick_main_entity(entities)
    main_action = _pick_main_action(entities)

    motifs: List[Motif] = []

    # 1. Main content signature: the strongest non-neutral content axes.
    content_parts = {
        axis: bucket
        for axis, bucket in axis_buckets.items()
        if _valid(bucket)
    }
    if len(content_parts) >= 2:
        motifs.append(Motif(
            key=_make_key(content_parts),
            motif_type="content_axis_signature",
            parts=content_parts,
        ))

    # 2. Setting + entity: concrete subject matter.
    if _valid(setting) and main_entity:
        parts = {"setting": setting, "entity": main_entity}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="setting_entity",
            parts=parts,
        ))

    # 3. Content + entity: what kind of content the subject belongs to.
    strongest_axis = _strongest_axis(content_axes, axis_buckets)
    if strongest_axis and main_entity:
        axis_name, bucket = strongest_axis
        parts = {axis_name: bucket, "entity": main_entity}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="content_entity",
            parts=parts,
        ))

    # 4. Content + setting: repeatable context preference.
    if strongest_axis and _valid(setting):
        axis_name, bucket = strongest_axis
        parts = {axis_name: bucket, "setting": setting}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="content_setting",
            parts=parts,
        ))

    # 5. Action motif: useful for talk/review/play/cook/sport themes.
    if main_action and _valid(setting):
        parts = {"setting": setting, "action": main_action}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="setting_action",
            parts=parts,
        ))
    elif main_action and strongest_axis:
        axis_name, bucket = strongest_axis
        parts = {axis_name: bucket, "action": main_action}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="content_action",
            parts=parts,
        ))

    # 6. Function motif: useful for tutorial/review/documentary/game patterns.
    if _valid(scene_function) and _valid(setting):
        parts = {"scene_function": scene_function, "setting": setting}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="function_setting",
            parts=parts,
        ))
    elif _valid(scene_function) and main_entity:
        parts = {"scene_function": scene_function, "entity": main_entity}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="function_entity",
            parts=parts,
        ))

    # 7. Supplemental visual atmosphere. Lower weight in clustering.
    if _valid(style) and _valid(mood):
        parts = {"style": style, "mood": mood}
        motifs.append(Motif(
            key=_make_key(parts),
            motif_type="supplemental_style_mood",
            parts=parts,
        ))

    return motifs


def _valid(value: str) -> bool:
    return bool(value) and value not in MOTIF_EXCLUDE_VALUES


def _pick_main_entity(entities: List[Dict[str, Any]]) -> str:
    candidates = []
    for ent in entities:
        role = ent.get("role", "object")
        name = ent.get("name", "")
        if role in MOTIF_EXCLUDE_ROLES:
            continue
        if name in MOTIF_EXCLUDE_ENTITIES:
            continue
        if not _valid(name):
            continue
        priority = 0 if role == "main_subject" else 1
        candidates.append((priority, name))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _pick_main_action(entities: List[Dict[str, Any]]) -> str:
    for ent in entities:
        if ent.get("role") == "background":
            continue
        doing = ent.get("relations", {}).get("DOING", "")
        if doing:
            return " ".join(doing.split()[:3])
    return ""


def _strongest_axis(
    content_axes: Dict[str, float],
    axis_buckets: Dict[str, str],
) -> Optional[Tuple[str, str]]:
    candidates = [
        (axis, abs(content_axes.get(axis, 0.0)), axis_buckets.get(axis, "neutral"))
        for axis in axis_buckets
        if _valid(axis_buckets.get(axis, "neutral"))
    ]
    if not candidates:
        return None
    axis, _, bucket = max(candidates, key=lambda item: item[1])
    return axis, bucket


def _make_key(parts: Dict[str, str]) -> str:
    return "|".join(f"{key}={value}" for key, value in sorted(parts.items()))
