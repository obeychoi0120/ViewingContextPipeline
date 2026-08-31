from __future__ import annotations

from typing import Final


SETTING_CONTEXTS: Final[tuple[str, ...]] = (
    "indoor",
    "outdoor_urban",
    "outdoor_nature",
    "transport",
    "unknown",
)
ENTITY_ROLES: Final[tuple[str, ...]] = (
    "primary",
    "secondary",
    "context",
    "background",
)
EVENT_REFERENCE_SLOTS: Final[tuple[str, ...]] = (
    "actor_id",
    "target_id",
    "instrument_id",
    "location_id",
)
STATIC_RELATION_TYPES: Final[tuple[str, ...]] = ("WEARING",)
AFFECT_VALENCE: Final[tuple[str, ...]] = (
    "positive",
    "neutral",
    "negative",
    "unknown",
)
AFFECT_AROUSAL: Final[tuple[str, ...]] = (
    "low",
    "medium",
    "high",
    "unknown",
)

ENTITY_GUIDANCE_MAX: Final[int] = 6
MAX_EVENTS: Final[int] = 4
MAX_STATIC_RELATIONS: Final[int] = 4
MAX_SEMANTIC_TOPICS: Final[int] = 3
MAX_TOPIC_WORDS: Final[int] = 4

DISALLOWED_GENERIC_ENTITY_NAMES: Final[frozenset[str]] = frozenset(
    {"object", "thing", "item", "something", "stuff"}
)
