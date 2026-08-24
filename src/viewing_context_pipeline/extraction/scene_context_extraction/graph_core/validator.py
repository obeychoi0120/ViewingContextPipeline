"""
graph_v4/validator.py - Validate and enrich Qwen scene observation JSON.

The model emits observable atoms only. This validator clamps those atoms to the
ontology and derives content_axes_4d, visual_style, and mood_state.
"""

from typing import Any, Dict, List, Optional, Tuple

from .scoring import compute_content_axes, derive_mood, derive_visual_style
from .ontology import (
    AFFECT_CUES,
    ENTITY_CATEGORIES,
    ENTITY_ROLES,
    FACE_PROMINENCE,
    MOOD_BINS,
    PEOPLE_DENSITY,
    RELATION_TYPES,
    SCENE_FUNCTIONS,
    SCENE_TYPES,
    SETTINGS,
    VISUAL_STYLE_CUES,
)

OBSERVABLE_GRAPH_FIELDS = (
    "scene_type",
    "visual_style_cues",
    "people_density",
    "face_prominence",
    "mood_bin",
    "affect_cues",
    "scene_function",
    "setting",
    "entities",
)

DERIVED_GRAPH_FIELDS = frozenset({"content_axes_4d", "visual_style", "style", "mood_state", "mood"})


_FIELD_DEFAULTS = {
    "scene_type": "unknown",
    "people_density": "unknown",
    "face_prominence": "unknown",
    "mood_bin": "unknown",
    "scene_function": "unknown",
    "setting": "unknown",
}

_STYLE_CUE_DEFAULTS = {
    "media_form": "unknown",
    "fantasy_element": "unknown",
    "shot_scale": "unknown",
    "graphic_density": "unknown",
    "composition_density": "unknown",
}


def validate_observation(obs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a raw Qwen scene observation and add derived fields."""
    warnings: List[str] = []
    result: Dict[str, Any] = {}

    if not isinstance(obs, dict):
        warnings.append("observation is not a dict - using empty observation")
        obs = {}

    result["scene_type"] = _validate_enum(
        obs.get("scene_type"), SCENE_TYPES, "scene_type",
        _FIELD_DEFAULTS["scene_type"], warnings,
    )

    result["visual_style_cues"] = _validate_style_cues(
        obs.get("visual_style_cues"), warnings,
    )

    result["people_density"] = _validate_enum(
        obs.get("people_density"), PEOPLE_DENSITY, "people_density",
        _FIELD_DEFAULTS["people_density"], warnings,
    )

    result["face_prominence"] = _validate_enum(
        obs.get("face_prominence"), FACE_PROMINENCE, "face_prominence",
        _FIELD_DEFAULTS["face_prominence"], warnings,
    )

    result["mood_bin"] = _validate_enum(
        obs.get("mood_bin", obs.get("mood")), MOOD_BINS, "mood_bin",
        _FIELD_DEFAULTS["mood_bin"], warnings,
    )

    result["affect_cues"] = _validate_affect_cues(
        obs.get("affect_cues"), warnings,
    )

    result["scene_function"] = _validate_enum(
        obs.get("scene_function"), SCENE_FUNCTIONS, "scene_function",
        _FIELD_DEFAULTS["scene_function"], warnings,
    )

    result["setting"] = _validate_enum(
        obs.get("setting"), SETTINGS, "setting",
        _FIELD_DEFAULTS["setting"], warnings,
    )

    raw_entities = obs.get("entities") or []
    if not isinstance(raw_entities, list):
        warnings.append("entities is not a list - using empty list")
        raw_entities = []

    seen_ids: set = set()
    result["entities"] = []
    for i, ent in enumerate(raw_entities):
        if not isinstance(ent, dict):
            warnings.append(f"entities[{i}] is not a dict - skipped")
            continue
        norm_ent = _validate_entity(ent, i, seen_ids, warnings)
        if norm_ent is not None:
            result["entities"].append(norm_ent)

    # Derived fields. Qwen does not emit these scores.
    result["content_axes_4d"] = compute_content_axes(result)
    result["visual_style"] = derive_visual_style(result)
    result["style"] = result["visual_style"]["primary"]
    result["mood_state"] = derive_mood(result)
    result["mood"] = result["mood_state"]["primary"]

    return result, warnings


def compact_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {field: observation[field] for field in OBSERVABLE_GRAPH_FIELDS if field in observation}


def _validate_style_cues(raw: Any, warnings: List[str]) -> Dict[str, str]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        warnings.append("visual_style_cues is not a dict - using defaults")
        raw = {}

    cues: Dict[str, str] = {}
    for cue_name, allowed in VISUAL_STYLE_CUES.items():
        cues[cue_name] = _validate_enum(
            raw.get(cue_name), allowed, f"visual_style_cues.{cue_name}",
            _STYLE_CUE_DEFAULTS[cue_name], warnings,
        )
    return cues


def _validate_affect_cues(raw: Any, warnings: List[str]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        warnings.append("affect_cues is not a list - using empty list")
        return []

    result: List[str] = []
    for i, value in enumerate(raw):
        cue = _validate_enum(
            value, AFFECT_CUES, f"affect_cues[{i}]", "unknown", warnings,
        )
        if cue not in result:
            result.append(cue)
    return result


def _validate_enum(
    value: Any,
    allowed: frozenset,
    field_name: str,
    default: str,
    warnings: List[str],
) -> str:
    if value is None or not isinstance(value, str):
        warnings.append(f"{field_name}: missing or non-string - defaulting to '{default}'")
        return default

    val = value.strip().lower()
    if val in allowed:
        return val

    normalized = (
        val.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )
    if normalized in allowed:
        warnings.append(f"{field_name}: '{value}' normalized to '{normalized}'")
        return normalized

    if normalized in {"n_a", "na", "not_applicable", "unclear"} and "unknown" in allowed:
        warnings.append(f"{field_name}: '{value}' normalized to 'unknown'")
        return "unknown"

    alias_key = field_name.split("[", 1)[0]
    alias = (
        _ENUM_ALIASES.get(field_name, {}).get(normalized)
        or _ENUM_ALIASES.get(alias_key, {}).get(normalized)
    )
    if alias and alias in allowed:
        warnings.append(f"{field_name}: '{value}' aliased to '{alias}'")
        return alias

    warnings.append(f"{field_name}: '{value}' not in allowed set - defaulting to '{default}'")
    return default


_ENUM_ALIASES: Dict[str, Dict[str, str]] = {
    "scene_type": {
        "graphic_text": "graphic_information",
        "graphic_info": "graphic_information",
        "graphic": "graphic_information",
        "activity_sport": "sport_fitness",
        "sports": "sport_fitness",
        "sport": "sport_fitness",
        "vehicle_travel": "vehicle_transport",
        "vehicle": "vehicle_transport",
        "fashion_beauty": "object_product",
        "space_place": "nature_landscape",
        "mixed": "unknown",
    },
    "visual_style_cues.media_form": {
        "animated": "animation",
        "cartoon": "animation",
        "cgi": "animation",
        "cg": "animation",
        "3d_animation": "animation",
        "2d_animation": "animation",
        "animation_2d": "animation",
        "animation_3d": "animation",
        "game_graphics": "graphics",
        "motion_graphics": "graphics",
        "live_action_plus_cg": "live_action_cg",
        "mixed": "unknown",
        "mixed_media": "unknown",
    },
    "visual_style_cues.fantasy_element": {
        "low": "none",
        "medium": "mid",
        "med": "mid",
        "moderate": "mid",
    },
    "visual_style_cues.shot_scale": {
        "detail": "close",
        "macro": "close",
        "extreme_close": "close",
        "close_up": "close",
        "closeup": "close",
        "full": "wide",
        "long": "wide",
        "mixed": "unknown",
    },
    "visual_style_cues.graphic_density": {
        "low": "none",
        "mixed": "unknown",
    },
    "visual_style_cues.composition_density": {
        "mixed": "unknown",
    },
    "people_density": {
        "no_people": "none",
        "no_person": "none",
        "zero": "none",
        "single": "one",
        "solo": "one",
        "one_person": "one",
        "two": "few",
        "pair": "few",
        "duo": "few",
        "couple": "few",
        "group": "few",
        "small_group": "few",
        "crowd": "many",
    },
    "face_prominence": {
        "medium": "mid",
        "med": "mid",
        "moderate": "mid",
        "prominent": "high",
        "dominant": "high",
        "close": "high",
        "close_up": "high",
        "closeup": "high",
    },
    "mood_bin": {
        "joyful": "cheerful",
        "happy": "cheerful",
        "playful": "cheerful",
        "excited": "cheerful",
        "joyful_energetic": "cheerful",
        "playful_light": "cheerful",
        "busy_dynamic": "cheerful",
        "calm": "peaceful",
        "peaceful_content": "peaceful",
        "quiet_neutral": "neutral",
        "serious_focused": "serious",
        "tense_alarming": "tense",
        "ominous": "dark",
        "ominous_uncanny": "dark",
        "melancholic": "dark",
        "melancholic_lonely": "dark",
        "sad": "dark",
        "mixed": "unknown",
    },
    "affect_cues": {
        "happy": "cheerful",
        "joyful": "cheerful",
        "playful": "cheerful",
        "calm_stillness": "calm",
        "calm_relaxed": "calm",
        "peaceful": "calm",
        "serene": "calm",
        "active_motion": "excited",
        "excitement": "excited",
        "excitement_celebration": "excited",
        "celebration": "excited",
        "surprise": "excited",
        "surprised": "excited",
        "surprise_shock": "excited",
        "shock": "excited",
        "focused": "serious",
        "serious_information": "serious",
        "serious_focus": "serious",
        "alarmed": "tense",
        "conflict_danger": "tense",
        "danger": "tense",
        "tension_danger": "tense",
        "anger_conflict": "tense",
        "isolation_sadness": "sad",
        "sadness": "sad",
        "sadness_isolation": "sad",
        "uncanny_darkness": "sad",
        "uncanny": "sad",
        "fear": "sad",
        "fear_uncanny": "sad",
        "laughter": "cheerful",
        "smile": "cheerful",
        "smile_laughter": "cheerful",
        "warmth": "cheerful",
        "affection": "cheerful",
        "warmth_affection": "cheerful",
        "mixed": "unknown",
    },
    "scene_function": {
        "tutorial": "instructional",
        "how_to": "instructional",
        "instruction": "instructional",
        "manufacturing": "instructional",
        "making": "instructional",
        "process": "instructional",
        "review": "review_comparison",
        "comparison": "review_comparison",
        "report": "information_report",
        "news": "information_report",
        "interview": "information_report",
        "interview_conversation": "information_report",
        "talk": "information_report",
        "documentary": "observation_documentary",
        "observation": "observation_documentary",
        "game": "competition_game",
        "competition": "competition_game",
        "sports_game": "competition_game",
        "story": "narrative",
        "concert": "performance",
        "social": "emotional_social",
        "aesthetic": "performance",
        "aesthetic_showcase": "performance",
        "showcase": "performance",
    },
    "entities": {
        # Entity category aliases for removed/merged categories
        "pet": "animal",
        "animal_pet": "animal",
        "general_object": "object",
        "face": "person",
        "outfit": "person",
        "plant": "animal",
        "drink": "food",
        "car": "vehicle",
        "train": "vehicle",
        "airplane": "vehicle",
        "phone": "device",
        "laptop": "device",
        "camera": "device",
        "furniture": "object",
        "book": "object",
        "cosmetic": "object",
        "sports_equipment": "object",
        "music_instrument": "object",
        "artwork": "object",
        "product": "object",
        "urban_structure": "building",
        "street": "building",
        "sky": "nature",
        "water": "nature",
        "mountain": "nature",
        "screen_text": "text",
        "graphic": "text",
        "bag": "person",
        "shoes": "person",
        "accessory": "person",
        "ui_element": "text",
        "virtual_element": "text",
        "information_media": "text",
    },
    "setting": {
        "golf_course": "sports_venue",
        "sports_field": "sports_venue",
        "soccer_field": "sports_venue",
        "baseball_field": "sports_venue",
        "football_field": "sports_venue",
        "news_studio": "studio",
        "tv_studio": "studio",
        "recording_studio": "studio",
        "indoor": "home",
        "indoors": "home",
        "inside": "home",
        "house": "home",
        "room": "home",
        "bedroom": "home",
        "living_room": "home",
        "kitchen": "home",
        "bathroom": "home",
        "cafe": "dining",
        "restaurant": "dining",
        "bar": "dining",
        "market": "dining",
        "office": "workplace",
        "school": "workplace",
        "hospital": "workplace",
        "laboratory": "workplace",
        "lab": "workplace",
        "library": "workplace",
        "gym": "sports_venue",
        "gym_fitness": "sports_venue",
        "swimming_pool": "sports_venue",
        "stadium": "sports_venue",
        "stage": "studio",
        "theater": "studio",
        "concert": "studio",
        "store": "commercial",
        "mall": "commercial",
        "supermarket": "commercial",
        "convenience_store": "commercial",
        "factory": "commercial",
        "warehouse": "commercial",
        "airport": "transport",
        "station": "transport",
        "subway": "transport",
        "train_station": "transport",
        "bus_station": "transport",
        "harbor": "transport",
        "pier": "transport",
        "dock": "transport",
        "port": "transport",
        "vehicle_interior": "transport",
        "car": "transport",
        "bus": "transport",
        "train_interior": "transport",
        "airplane_interior": "transport",
        "street": "urban",
        "city": "urban",
        "rooftop": "urban",
        "plaza": "urban",
        "balcony": "urban",
        "terrace": "urban",
        "parking_lot": "urban",
        "nature": "nature_green",
        "outdoor": "nature_green",
        "outdoors": "nature_green",
        "outside": "nature_green",
        "forest": "nature_green",
        "mountain": "nature_green",
        "park": "nature_green",
        "garden": "nature_green",
        "field": "nature_green",
        "rural": "nature_green",
        "farm": "nature_green",
        "countryside": "nature_green",
        "jungle": "nature_green",
        "woods": "nature_green",
        "courtyard": "nature_green",
        "backyard": "nature_green",
        "beach": "nature_water",
        "ocean": "nature_water",
        "sea": "nature_water",
        "underwater": "nature_water",
        "river": "nature_water",
        "lake": "nature_water",
        "snow": "arid",
        "desert": "arid",
        "abstract": "virtual",
        "museum": "commercial",
        "cinema": "commercial",
        "church": "commercial",
        "temple": "commercial",
    },
}


def _validate_entity(
    ent: Dict[str, Any],
    index: int,
    seen_ids: set,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    local_id = ent.get("local_id", "")
    local_id = local_id.strip() if isinstance(local_id, str) else ""
    if not local_id:
        local_id = f"e{index + 1}"
        warnings.append(f"entities[{index}]: missing local_id - assigned '{local_id}'")

    if local_id in seen_ids:
        warnings.append(f"entities[{index}]: duplicate local_id '{local_id}' - skipped")
        return None
    seen_ids.add(local_id)

    category = _validate_enum(
        ent.get("category"), ENTITY_CATEGORIES,
        f"entities[{index}].category", "unknown", warnings,
    )

    name = ent.get("name", "")
    if not isinstance(name, str) or not name.strip():
        name = category
        warnings.append(f"entities[{index}]: missing name - using category '{category}'")
    else:
        name = name.strip().lower()

    role = _validate_enum(
        ent.get("role"), ENTITY_ROLES,
        f"entities[{index}].role", "object", warnings,
    )

    relations = _validate_entity_relations(
        ent.get("relations"), index, warnings,
    )

    return {
        "local_id": local_id,
        "category": category,
        "name": name,
        "role": role,
        "relations": relations,
    }


def _validate_entity_relations(
    raw_relations: Any,
    entity_index: int,
    warnings: List[str],
) -> Dict[str, str]:
    if raw_relations is None:
        return {}
    if not isinstance(raw_relations, dict):
        warnings.append(f"entities[{entity_index}].relations: not a dict - using empty")
        return {}

    valid_relations: Dict[str, str] = {}
    for key, value in raw_relations.items():
        if not isinstance(key, str):
            warnings.append(f"entities[{entity_index}].relations: non-string key skipped")
            continue
        norm_key = key.strip().upper()
        if norm_key in _RELATION_SLOT_ALIASES:
            alias = _RELATION_SLOT_ALIASES[norm_key]
            if alias is None:
                warnings.append(
                    f"entities[{entity_index}].relations: legacy slot '{key}' skipped"
                )
                continue
            warnings.append(
                f"entities[{entity_index}].relations: '{key}' aliased to '{alias}'"
            )
            norm_key = alias
        if norm_key not in RELATION_TYPES:
            warnings.append(
                f"entities[{entity_index}].relations: unknown slot '{key}' - skipped"
            )
            continue
        if not isinstance(value, str) or not value.strip():
            warnings.append(
                f"entities[{entity_index}].relations.{norm_key}: "
                "empty or non-string value - skipped"
            )
            continue
        if norm_key not in valid_relations:
            valid_relations[norm_key] = value.strip().lower()

    return valid_relations


_RELATION_SLOT_ALIASES: Dict[str, Optional[str]] = {
    "INTERACT_WITH": "INTERACTS_WITH",
    "WITH": "INTERACTS_WITH",
    "ABOUT": "INTERACTS_WITH",
}
