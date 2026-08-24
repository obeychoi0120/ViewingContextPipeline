"""
graph_v4/scoring.py - Deterministic mapping from observations to profiles.

Qwen outputs visible labels and short relation phrases. This module maps those
atoms into 4 content axes, visual style archetypes, and a normalized mood state.
"""

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from .ontology import (
    ENTITY_CATEGORIES,
    MOOD_BINS,
    SCENE_FUNCTIONS,
    SCENE_TYPES,
    SETTINGS,
    STYLES,
    VISUAL_STYLE_CUES,
)


CONTENT_AXIS_ORDER = [
    "subject_sociality",
    "media_syntheticity",
    "setting_context",
    "utility_orientation",
]

BUILT_SETTINGS = SETTINGS & {
    "home", "dining", "workplace", "studio", "sports_venue",
    "commercial", "transport", "urban",
}

NATURE_SETTINGS = SETTINGS & {"nature_green", "nature_water", "arid"}

VIRTUAL_SETTINGS = SETTINGS & {"virtual"}

PEOPLE_SCENE_TYPES = SCENE_TYPES & {
    "people_social", "person_portrait", "sport_fitness",
}

WORLD_SCENE_TYPES = SCENE_TYPES & {
    "nature_landscape", "animal_pet", "object_product", "food_drink",
    "vehicle_transport",
}

SYNTHETIC_MEDIA_FORMS = VISUAL_STYLE_CUES["media_form"] & {
    "live_action_cg", "animation", "graphics",
}

SYNTHETIC_CATEGORIES = ENTITY_CATEGORIES & {"text"}

NATURE_CATEGORIES = ENTITY_CATEGORIES & {"animal", "nature"}

PRODUCT_INFO_CATEGORIES = ENTITY_CATEGORIES & {
    "device", "food", "object", "text",
}

SOCIAL_ACTION_WORDS = {
    "talking", "speaking", "laughing", "smiling", "interviewing",
    "presenting", "chatting", "singing", "dancing", "performing",
    "hugging", "celebrating",
}

UTILITY_WORDS = {
    "explain", "explaining", "review", "reviewing", "tutorial", "guide",
    "how to", "news", "reporting", "price", "stock", "data", "chart",
    "text", "recipe", "cooking", "instruction", "lecture", "teaching",
    "demonstrating", "unboxing", "product", "advertisement", "commercial",
    "comparison", "process", "making", "repairing",
}

EXPERIENCE_WORDS = {
    "travel", "walking", "playing", "laughing", "performance", "concert",
    "swimming", "running", "exploring", "landscape", "wildlife", "story",
}

UTILITY_FUNCTIONS = SCENE_FUNCTIONS & {
    "instructional", "review_comparison", "information_report",
}

EXPERIENCE_FUNCTIONS = SCENE_FUNCTIONS & {
    "observation_documentary", "narrative", "performance",
    "emotional_social",
}


def compute_content_axes(observation: Dict[str, Any]) -> Dict[str, float]:
    """Compute the 4D content axis vector in [-1, +1]."""
    scene_type = observation.get("scene_type", "unknown")
    setting = observation.get("setting", "unknown")
    scene_function = observation.get("scene_function", "unknown")
    people_density = observation.get("people_density", "unknown")
    face_prominence = observation.get("face_prominence", "unknown")
    entities = observation.get("entities", [])
    cues = observation.get("visual_style_cues", {})
    text_blob = _relation_text(entities)

    categories = Counter(e.get("category", "") for e in entities)
    names = [e.get("name", "") for e in entities]
    person_count = categories.get("person", 0)
    entity_count = max(len(entities), 1)
    person_ratio = person_count / entity_count

    subject = 0.0
    if scene_type in PEOPLE_SCENE_TYPES:
        subject += 0.45
    elif scene_type in WORLD_SCENE_TYPES:
        subject -= 0.25
    subject += {
        "none": -0.25,
        "one": 0.25,
        "few": 0.35,
        "many": 0.45,
    }.get(people_density, 0.0)
    subject += {
        "low": 0.05,
        "mid": 0.18,
        "high": 0.30,
    }.get(face_prominence, 0.0)
    if person_count:
        subject += min(0.25, person_ratio * 0.45)
    if _contains_any(text_blob, SOCIAL_ACTION_WORDS):
        subject += 0.15
    if categories.get("object", 0) + categories.get("device", 0) >= 2:
        subject -= 0.15

    synthetic = 0.0
    media_form = cues.get("media_form", "unknown")
    if media_form == "live_action":
        synthetic -= 0.25
    elif media_form == "live_action_cg":
        synthetic += 0.35
    elif media_form in {"animation", "graphics"}:
        synthetic += 0.60
    if scene_type == "graphic_information":
        synthetic += 0.45
    graphic_density = cues.get("graphic_density", "unknown")
    synthetic += {
        "none": -0.05,
        "medium": 0.20,
        "high": 0.35,
    }.get(graphic_density, 0.0)
    fantasy_element = cues.get("fantasy_element", "unknown")
    synthetic += {
        "mid": 0.18,
        "high": 0.32,
    }.get(fantasy_element, 0.0)
    synthetic_count = sum(categories.get(c, 0) for c in SYNTHETIC_CATEGORIES)
    if synthetic_count:
        synthetic += min(0.4, synthetic_count * 0.15)
    if any(_contains_any(n, {"screen", "overlay", "chat", "subtitle", "ui"}) for n in names):
        synthetic += 0.15

    setting_context = 0.0
    if setting in BUILT_SETTINGS:
        setting_context += 0.55
    elif setting in NATURE_SETTINGS:
        setting_context -= 0.55
    elif setting in VIRTUAL_SETTINGS:
        setting_context += 0.10
    nature_count = sum(categories.get(c, 0) for c in NATURE_CATEGORIES)
    if nature_count:
        setting_context -= min(0.35, nature_count * 0.15)
    if categories.get("building", 0):
        setting_context += 0.15

    utility = 0.0
    if scene_function in UTILITY_FUNCTIONS:
        utility += 0.45
    elif scene_function in EXPERIENCE_FUNCTIONS:
        utility -= 0.20
    elif scene_function == "competition_game":
        utility += 0.10
    if scene_type in {"graphic_information", "object_product"}:
        utility += 0.35
    elif scene_type in {"people_social", "nature_landscape", "animal_pet", "vehicle_transport"}:
        utility -= 0.15
    info_count = sum(categories.get(c, 0) for c in PRODUCT_INFO_CATEGORIES)
    if info_count:
        utility += min(0.35, info_count * 0.12)
    if _contains_any(text_blob, UTILITY_WORDS):
        utility += 0.35
    if _contains_any(text_blob, EXPERIENCE_WORDS):
        utility -= 0.20
    if scene_type == "sport_fitness":
        utility += 0.10

    return {
        "subject_sociality": _clamp_round(subject),
        "media_syntheticity": _clamp_round(synthetic),
        "setting_context": _clamp_round(setting_context),
        "utility_orientation": _clamp_round(utility),
    }


def bucket_content_axes(axes: Dict[str, float], threshold: float = 0.25) -> Dict[str, str]:
    """Discretize content axes for motifs and graph nodes."""
    buckets: Dict[str, str] = {}
    for axis_name in CONTENT_AXIS_ORDER:
        value = axes.get(axis_name, 0.0)
        if value >= threshold:
            buckets[axis_name] = "positive"
        elif value <= -threshold:
            buckets[axis_name] = "negative"
        else:
            buckets[axis_name] = "neutral"
    return buckets


def derive_visual_style(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Derive visual style archetype membership from observable cues."""
    cues = observation.get("visual_style_cues", {})
    scene_type = observation.get("scene_type", "unknown")
    setting = observation.get("setting", "unknown")
    mood_bin = observation.get("mood_bin", "unknown")
    scene_function = observation.get("scene_function", "unknown")
    face_prominence = observation.get("face_prominence", "unknown")
    entities = observation.get("entities", [])
    categories = Counter(e.get("category", "") for e in entities)

    media_form = cues.get("media_form", "unknown")
    fantasy = cues.get("fantasy_element", "unknown")
    shot = cues.get("shot_scale", "unknown")
    graphic_density = cues.get("graphic_density", "unknown")
    composition = cues.get("composition_density", "unknown")

    scores = {style: 0.0 for style in STYLES if style != "mixed"}
    evidence: Dict[str, List[str]] = {style: [] for style in scores}

    def add(style: str, amount: float, reason: str) -> None:
        scores[style] += amount
        evidence[style].append(reason)

    if media_form == "live_action":
        add("natural_documentary", 0.25, "live_action")
    if media_form == "live_action_cg":
        add("neon_synthetic", 0.22, "live_action_cg")
        add("low_key_cinematic", 0.10, "live_action_cg")
    if media_form == "animation":
        add("flat_graphic_playful", 0.36, "animation")
        add("bright_social_pop", 0.12, "animation")
        add("neon_synthetic", 0.14, "animation")
    if media_form == "graphics":
        add("flat_graphic_playful", 0.38, "graphics")
        add("clean_minimal_product", 0.16, "graphics")

    if fantasy in {"mid", "high"}:
        add("neon_synthetic", 0.18 if fantasy == "mid" else 0.28, f"{fantasy}_fantasy")
        add("low_key_cinematic", 0.10, f"{fantasy}_fantasy")

    if graphic_density in {"medium", "high"}:
        add("flat_graphic_playful", 0.22, f"{graphic_density}_graphics")
        add("bright_social_pop", 0.12, f"{graphic_density}_graphics")
    if graphic_density == "none":
        add("natural_documentary", 0.12, f"{graphic_density}_graphics")

    if composition == "minimal":
        add("clean_minimal_product", 0.25, "minimal_composition")
    if composition == "balanced":
        add("natural_documentary", 0.08, "balanced_composition")
    if composition == "busy":
        add("bright_social_pop", 0.16, "busy_composition")

    if shot == "close" and (scene_type == "object_product" or categories.get("object", 0)):
        add("clean_minimal_product", 0.22, f"{shot}_product")
    if shot in {"close", "medium"} and categories.get("person", 0):
        add("editorial_portrait", 0.22, f"{shot}_person")
        add("soft_warm_intimate", 0.10, f"{shot}_person")
    if face_prominence in {"mid", "high"}:
        add("editorial_portrait", 0.12 if face_prominence == "mid" else 0.24, f"{face_prominence}_face")

    if scene_type == "object_product" or categories.get("object", 0) or categories.get("device", 0):
        add("clean_minimal_product", 0.28, "product_or_object_focus")
    if scene_type == "person_portrait":
        add("editorial_portrait", 0.28, scene_type)
    if scene_type == "graphic_information":
        add("flat_graphic_playful", 0.20, "graphic_information")
        add("clean_minimal_product", 0.16, "graphic_information")
    if scene_type in {"people_social", "food_drink"} and mood_bin in {"cheerful", "peaceful"}:
        add("soft_warm_intimate", 0.16, f"{scene_type}_{mood_bin}")
        add("bright_social_pop", 0.10, f"{scene_type}_{mood_bin}")
    if scene_function == "performance":
        add("clean_minimal_product", 0.08, "performance")
        add("editorial_portrait", 0.08, "performance")
    if scene_function == "information_report":
        add("natural_documentary", 0.12, "information_report")
        add("clean_minimal_product", 0.10, "information_report")
    if setting in NATURE_SETTINGS:
        add("natural_documentary", 0.20, "nature_setting")
    if setting == "urban" and media_form in SYNTHETIC_MEDIA_FORMS:
        add("neon_synthetic", 0.12, "synthetic_urban")

    normalized = {k: round(min(1.0, v), 3) for k, v in scores.items()}
    primary = max(normalized, key=normalized.get) if normalized else "mixed"
    if normalized.get(primary, 0.0) < 0.2:
        primary = "mixed"

    return {
        "primary": primary,
        "scores": normalized,
        "evidence": evidence.get(primary, [])[:5] if primary != "mixed" else [],
    }


def derive_mood(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize mood from the VLM mood_bin plus affect/action evidence."""
    mood_bin = observation.get("mood_bin", "unknown")
    affect_cues = set(observation.get("affect_cues", []))
    axes = observation.get("content_axes_4d", {})
    entities = observation.get("entities", [])
    relation_text = _relation_text(entities)

    valence, arousal = _mood_seed(mood_bin)
    evidence: List[str] = [] if mood_bin == "unknown" else [f"mood_bin:{mood_bin}"]

    def bump(dv: float, da: float, reason: str) -> None:
        nonlocal valence, arousal
        valence += dv
        arousal += da
        evidence.append(reason)

    if "cheerful" in affect_cues:
        bump(0.35, 0.15, "cheerful")
    if "excited" in affect_cues:
        bump(0.20, 0.36, "excited")
    if "calm" in affect_cues:
        bump(0.18, -0.32, "calm")
    if "serious" in affect_cues:
        bump(-0.02, -0.06, "serious")
    if "tense" in affect_cues:
        bump(-0.40, 0.34, "tense")
    if "sad" in affect_cues:
        bump(-0.38, -0.18, "sad")

    if axes.get("utility_orientation", 0.0) > 0.35:
        bump(0.0, -0.08, "utility_oriented")
    if _contains_any(relation_text, {"laughing", "smiling", "playing", "dancing"}):
        bump(0.18, 0.12, "positive_action")
    if _contains_any(relation_text, {"fighting", "crying", "danger", "storm", "fire"}):
        bump(-0.22, 0.22, "negative_action")

    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))
    primary = mood_bin if mood_bin in MOOD_BINS and mood_bin != "unknown" else _mood_from_valence_arousal(
        valence, arousal, affect_cues,
    )

    return {
        "primary": primary,
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "evidence": evidence[:6],
    }


def _mood_seed(mood_bin: str) -> Tuple[float, float]:
    return {
        "cheerful": (0.45, 0.30),
        "peaceful": (0.25, -0.35),
        "neutral": (0.0, 0.0),
        "serious": (-0.05, -0.05),
        "tense": (-0.35, 0.40),
        "dark": (-0.45, -0.05),
    }.get(mood_bin, (0.0, 0.0))


def _mood_from_valence_arousal(
    valence: float,
    arousal: float,
    affect_cues: Iterable[str],
) -> str:
    cues = set(affect_cues)
    if "serious" in cues and abs(valence) < 0.35 and arousal < 0.25:
        return "serious"
    if valence >= 0.22 and arousal >= 0.18:
        return "cheerful"
    if valence >= 0.15 and arousal <= 0.05:
        return "peaceful"
    if valence <= -0.25 and arousal >= 0.18:
        return "tense"
    if valence <= -0.25 or "sad" in cues:
        return "dark"
    if arousal >= 0.30 and valence >= 0.0:
        return "cheerful"
    return "neutral"


def _relation_text(entities: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for ent in entities:
        chunks.append(str(ent.get("name", "")))
        chunks.append(str(ent.get("category", "")))
        for val in ent.get("relations", {}).values():
            chunks.append(str(val))
    return " ".join(chunks).lower()


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def _clamp_round(value: float) -> float:
    return round(max(-1.0, min(1.0, value)), 3)
