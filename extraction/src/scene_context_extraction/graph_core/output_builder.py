"""
graph_v2/output_builder.py — Build the final 4-layer output JSON.

Layers:
    1. sessions → scenes (observations + motifs)
    2. graph (nodes + triplets)
    3. forest_aggregate
    4. macro_interests
"""

import json
from typing import Dict, Any, List

from .graph_store import GraphStore
from .aggregator import build_session_aggregate, build_forest_aggregate
from .clusterer import cluster_motifs, apply_behavior_adjustment


def build_final_output(
    store: GraphStore,
    forest_id: int,
    clustering_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Build the complete final output JSON.

    Args:
        store: GraphStore instance with all data loaded.
        forest_id: Forest to build output for.
        clustering_threshold: Similarity threshold for motif clustering.

    Returns:
        Complete output dict matching the specification schema.
    """
    forest = store.get_forest(forest_id)
    if not forest:
        return {"error": f"Forest {forest_id} not found"}

    loyalty = forest.get("behavior_loyalty", 0.5)
    exploration = forest.get("behavior_exploration", 0.5)

    # --- Sessions layer ---
    sessions_output: List[Dict[str, Any]] = []
    sessions = store.get_sessions(forest_id)

    for session in sessions:
        session_data = _build_session_output(store, session)
        sessions_output.append(session_data)

    # --- Graph layer ---
    graph = store.export_graph_dict()

    # --- Forest aggregate ---
    forest_aggregate = build_forest_aggregate(store, forest_id)

    # --- Macro interests ---
    macro_interests = cluster_motifs(store, forest_id, threshold=clustering_threshold)
    macro_interests = apply_behavior_adjustment(macro_interests, loyalty, exploration)

    macro_output = [
        {
            "macro_id": m.macro_id,
            "label": m.label,
            "top_components": m.top_components,
            "member_motifs": m.member_motifs,
            "evidence_sessions": m.evidence_sessions,
            "evidence_scenes": m.evidence_scenes,
            "visual_support": m.visual_support,
            "behavior_adjusted": m.behavior_adjusted,
        }
        for m in macro_interests
    ]

    return {
        "forest_id": forest.get("forest_key", "current"),
        "behavior_profile": {
            "loyalty": loyalty,
            "exploration": exploration,
        },
        "sessions": sessions_output,
        "graph": graph,
        "forest_aggregate": forest_aggregate,
        "macro_interests": macro_output,
    }


def _build_session_output(
    store: GraphStore,
    session: Dict,
) -> Dict[str, Any]:
    """Build output for a single session including scenes and aggregate."""
    session_id = session["id"]
    scenes = store.get_scenes(session_id)

    scenes_output: List[Dict[str, Any]] = []
    for scene in scenes:
        obs = json.loads(scene["observation_json"]) if scene.get("observation_json") else {}

        # Get motif keys for this scene
        scene_motifs = store.get_motifs_for_scene(scene["id"])
        motif_keys = [m["motif_key"] for m in scene_motifs]

        scenes_output.append({
            "scene_id": scene["scene_key"],
            "observation": obs,
            "motifs": motif_keys,
        })

    # Session aggregate
    session_aggregate = build_session_aggregate(store, session_id)

    return {
        "session_id": session["session_key"],
        "scenes": scenes_output,
        "session_aggregate": session_aggregate,
    }
