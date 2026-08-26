from __future__ import annotations

from typing import Any, Final


SCENE_SCHEMA_VERSION: Final[str] = "minimal-semantic-scene/v1"

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

MAX_ENTITIES: Final[int] = 6
MAX_EVENTS: Final[int] = 4
MAX_STATIC_RELATIONS: Final[int] = 4
MAX_SEMANTIC_TOPICS: Final[int] = 3
MAX_TOPIC_WORDS: Final[int] = 4

DISALLOWED_GENERIC_ENTITY_NAMES: Final[frozenset[str]] = frozenset(
    {"object", "thing", "item", "something", "stuff"}
)


def taxonomy_contract() -> dict[str, Any]:
    """Return the code-owned mapping used to render and fingerprint the prompt."""
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "setting_contexts": list(SETTING_CONTEXTS),
        "entity_roles": list(ENTITY_ROLES),
        "event_reference_slots": list(EVENT_REFERENCE_SLOTS),
        "static_relation_types": list(STATIC_RELATION_TYPES),
        "affect_valence": list(AFFECT_VALENCE),
        "affect_arousal": list(AFFECT_AROUSAL),
        "limits": {
            "entities": MAX_ENTITIES,
            "events": MAX_EVENTS,
            "static_relations": MAX_STATIC_RELATIONS,
            "semantic_topics": MAX_SEMANTIC_TOPICS,
            "topic_words": MAX_TOPIC_WORDS,
        },
        "disallowed_generic_entity_names": sorted(
            DISALLOWED_GENERIC_ENTITY_NAMES
        ),
    }
