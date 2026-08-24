"""
graph_v2/aggregator.py — Session and Forest level aggregate statistics.

Computes supplemental style/mood distributions, top entities/motifs, and
4D content axis distributions from the graph store.
"""

import json
from collections import Counter
from typing import Dict, List, Any

from .graph_store import GraphStore
from .scoring import CONTENT_AXIS_ORDER, bucket_content_axes


def build_session_aggregate(store: GraphStore, session_id: int) -> Dict[str, Any]:
    """Build aggregate statistics for a single session.

    Args:
        store: GraphStore instance.
        session_id: SQLite session id.

    Returns:
        Dict with content axis, supplemental style/mood, entities, and motifs.
    """
    scenes = store.get_scenes(session_id)
    return _aggregate_from_scenes(store, scenes)


def build_forest_aggregate(store: GraphStore, forest_id: int) -> Dict[str, Any]:
    """Build aggregate statistics across all sessions in a forest.

    Args:
        store: GraphStore instance.
        forest_id: SQLite forest id.

    Returns:
        Dict with the same structure as session aggregate, but forest-wide.
    """
    scenes = store.get_all_scenes(forest_id)
    return _aggregate_from_scenes(store, scenes)


def _aggregate_from_scenes(
    store: GraphStore,
    scenes: List[Dict],
) -> Dict[str, Any]:
    """Common aggregation logic over a set of scenes."""
    style_counter: Counter = Counter()
    mood_counter: Counter = Counter()
    scene_function_counter: Counter = Counter()
    entity_counter: Counter = Counter()
    motif_counter: Counter = Counter()
    axis_bucket_counters: Dict[str, Counter] = {
        axis: Counter() for axis in CONTENT_AXIS_ORDER
    }
    axis_value_sums = {axis: 0.0 for axis in CONTENT_AXIS_ORDER}

    for scene in scenes:
        obs = json.loads(scene["observation_json"]) if scene.get("observation_json") else {}

        # Style
        style = obs.get("style", "mixed")
        style_counter[f"style:{style}"] += 1

        # Mood
        mood = obs.get("mood", "unknown")
        mood_counter[f"mood:{mood}"] += 1

        # Scene function
        scene_function = obs.get("scene_function", "unknown")
        scene_function_counter[f"scene_function:{scene_function}"] += 1

        # Entities (non-background, non-person)
        for ent in obs.get("entities", []):
            name = ent.get("name", "")
            role = ent.get("role", "object")
            if role != "background" and name and name != "person":
                entity_counter[f"entity:{name}"] += 1

        # Motifs
        for m in store.get_motifs_for_scene(scene["id"]):
            motif_counter[f"motif:{m['motif_key']}"] += 1

        # 4D content axes
        axes = obs.get("content_axes_4d", {})
        buckets = bucket_content_axes(axes)
        for axis_name in CONTENT_AXIS_ORDER:
            val = float(axes.get(axis_name, 0.0) or 0.0)
            axis_value_sums[axis_name] += val
            axis_bucket_counters[axis_name][buckets.get(axis_name, "neutral")] += 1

    # Build distribution
    total_scenes = len(scenes) or 1
    axis_bucket_dist = {}
    for axis_name, counter in axis_bucket_counters.items():
        dist = {}
        for val, count in counter.items():
            dist[val] = round(count / total_scenes, 3)
        axis_bucket_dist[axis_name] = dist
    axis_avg = {
        axis_name: round(axis_value_sums[axis_name] / total_scenes, 3)
        for axis_name in CONTENT_AXIS_ORDER
    }

    return {
        "content_axes_4d": axis_avg,
        "content_axis_distribution": axis_bucket_dist,
        "top_styles": _top_items(style_counter),
        "top_moods": _top_items(mood_counter),
        "top_scene_functions": _top_items(scene_function_counter),
        "top_entities": _top_items(entity_counter),
        "top_motifs": _top_items(motif_counter),
    }


def _top_items(counter: Counter, limit: int = 10) -> List[Dict[str, Any]]:
    """Return top items from counter as list of {id, count} dicts."""
    return [
        {"id": item, "count": count}
        for item, count in counter.most_common(limit)
    ]
