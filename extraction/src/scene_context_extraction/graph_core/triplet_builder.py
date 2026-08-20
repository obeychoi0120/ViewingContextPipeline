"""
graph_v2/triplet_builder.py — Deterministic triplet generation from validated observations.

Converts a validated scene observation into a list of (subject, predicate, object) triplets
following the graph node design from the specification.

Entity relations are per-entity attribute slots (DOING, WEARING, IS_A, INTERACTS_WITH).
Each slot is converted into a triplet linking the entity instance to a value node,
with intelligent resolution when the value matches another entity in the same scene.
"""

from typing import List, Dict, Any, NamedTuple

from .scoring import bucket_content_axes


class Triplet(NamedTuple):
    """A single graph triplet with provenance context."""
    subject: str
    predicate: str
    object: str


def build_triplets(
    session_id: str,
    scene_id: str,
    observation: Dict[str, Any],
    forest_id: str = "current",
) -> List[Triplet]:
    """Build all deterministic triplets from a validated observation.

    Args:
        session_id: e.g. "session_001"
        scene_id: e.g. "scene_003"
        observation: Validated observation dict.
        forest_id: Forest key, default "current".

    Returns:
        List of Triplet named tuples.
    """
    triplets: List[Triplet] = []
    scene_key = f"scene:{session_id}:{scene_id}"
    session_key = f"session:{session_id}"
    forest_key = f"forest:{forest_id}"

    # ------------------------------------------------------------------
    # 1. Structure triplets
    # ------------------------------------------------------------------
    triplets.append(Triplet(forest_key, "HAS_SESSION", session_key))
    triplets.append(Triplet(session_key, "HAS_SCENE", scene_key))

    # ------------------------------------------------------------------
    # 2. Content axis bucket triplets
    # Scores stay in observation_json; graph nodes use stable buckets.
    content_axes = observation.get("content_axes_4d", {})
    for axis_name, bucket in bucket_content_axes(content_axes).items():
        axis_node = f"content_axis:{axis_name}={bucket}"
        triplets.append(Triplet(scene_key, "HAS_CONTENT_AXIS", axis_node))

    # ------------------------------------------------------------------
    # 3. Observable visual style cue triplets
    # ------------------------------------------------------------------
    style_cues = observation.get("visual_style_cues", {})
    for cue_name, cue_value in style_cues.items():
        cue_node = f"style_cue:{cue_name}={cue_value}"
        triplets.append(Triplet(scene_key, "HAS_STYLE_CUE", cue_node))

    # ------------------------------------------------------------------
    # 4. Scene-level observable triplets
    # ------------------------------------------------------------------
    scene_type = observation.get("scene_type", "unknown")
    triplets.append(Triplet(scene_key, "HAS_SCENE_TYPE", f"scene_type:{scene_type}"))

    people_density = observation.get("people_density", "unknown")
    triplets.append(Triplet(
        scene_key, "HAS_PEOPLE_DENSITY", f"people_density:{people_density}"
    ))

    face_prominence = observation.get("face_prominence", "unknown")
    triplets.append(Triplet(
        scene_key, "HAS_FACE_PROMINENCE", f"face_prominence:{face_prominence}"
    ))

    mood_bin = observation.get("mood_bin", "unknown")
    triplets.append(Triplet(scene_key, "HAS_MOOD_BIN", f"mood_bin:{mood_bin}"))

    scene_function = observation.get("scene_function", "unknown")
    triplets.append(Triplet(
        scene_key, "HAS_SCENE_FUNCTION", f"scene_function:{scene_function}"
    ))

    # ------------------------------------------------------------------
    # 5. Derived style / mood / setting triplets
    # ------------------------------------------------------------------

    style = observation.get("style", "mixed")
    triplets.append(Triplet(scene_key, "HAS_STYLE", f"style:{style}"))

    mood = observation.get("mood", "unknown")
    triplets.append(Triplet(scene_key, "HAS_MOOD", f"mood:{mood}"))

    setting = observation.get("setting", "unknown")
    triplets.append(Triplet(scene_key, "HAS_SETTING", f"setting:{setting}"))

    # ------------------------------------------------------------------
    # 6. Legacy visual axis triplets, if a caller still provides them
    # ------------------------------------------------------------------
    visual_axes = observation.get("visual_axes", {})
    for axis_name, axis_value in visual_axes.items():
        axis_node = f"axis:{axis_name}:{axis_value}"
        triplets.append(Triplet(scene_key, "HAS_AXIS", axis_node))

    # ------------------------------------------------------------------
    # 7. Entity triplets (structural)
    # ------------------------------------------------------------------
    entities = observation.get("entities", [])

    # Build lookup for entity resolution
    entity_names = {}   # local_id → name
    name_to_id = {}     # name → local_id (first match)
    for ent in entities:
        lid = ent["local_id"]
        name = ent.get("name", "")
        entity_names[lid] = name
        if name and name not in name_to_id:
            name_to_id[name] = lid

    for ent in entities:
        local_id = ent["local_id"]
        instance_key = f"entity_instance:{session_id}:{scene_id}:{local_id}"
        entity_key = f"entity:{ent['name']}"
        category_key = f"category:{ent['category']}"
        role_key = f"role:{ent['role']}"

        triplets.append(Triplet(scene_key, "CONTAINS_INSTANCE", instance_key))
        triplets.append(Triplet(instance_key, "INSTANCE_OF", entity_key))
        triplets.append(Triplet(instance_key, "HAS_CATEGORY", category_key))
        triplets.append(Triplet(instance_key, "HAS_ROLE", role_key))

    # ------------------------------------------------------------------
    # 8. Per-entity relation slot triplets
    # ------------------------------------------------------------------
    for ent in entities:
        local_id = ent["local_id"]
        instance_key = f"entity_instance:{session_id}:{scene_id}:{local_id}"
        relations = ent.get("relations", {})

        for slot_key, slot_value in relations.items():
            predicate = slot_key.upper()

            # Smart resolution: check if value references another entity
            resolved_target = _resolve_relation_target(
                slot_value, local_id, entity_names, name_to_id,
                session_id, scene_id,
            )
            triplets.append(Triplet(instance_key, predicate, resolved_target))

    return triplets


def _resolve_relation_target(
    value: str,
    source_local_id: str,
    entity_names: Dict[str, str],
    name_to_id: Dict[str, str],
    session_id: str,
    scene_id: str,
) -> str:
    """Resolve a relation value to the best target node.

    Resolution priority:
        1. If value is a local_id reference (e.g., "e2") → entity instance node
        2. If value matches another entity's name → entity instance node
        3. Otherwise → literal value node
    """
    val_lower = value.strip().lower()

    # 1. Direct local_id reference (e.g., "e2", "e3")
    if val_lower in entity_names and val_lower != source_local_id:
        return f"entity_instance:{session_id}:{scene_id}:{val_lower}"

    # 2. Text matches another entity name
    if val_lower in name_to_id:
        target_id = name_to_id[val_lower]
        if target_id != source_local_id:
            return f"entity_instance:{session_id}:{scene_id}:{target_id}"

    # 3. Literal value node
    return f"value:{val_lower}"
