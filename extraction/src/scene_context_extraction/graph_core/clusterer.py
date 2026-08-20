"""
graph_v2/clusterer.py — Macro Interest clustering from motifs.

Uses weighted Jaccard similarity on motif components, then connected
components to form macro interest clusters.

No GNN or graph embedding required for MVP.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple

from .graph_store import GraphStore


# ---------------------------------------------------------------------------
# Component weight table
# ---------------------------------------------------------------------------
COMPONENT_WEIGHTS = {
    "subject_sociality": 1.25,
    "media_syntheticity": 1.25,
    "setting_context": 1.25,
    "utility_orientation": 1.25,
    "setting": 1.0,
    "scene_function": 0.9,
    "entity": 1.0,
    "action": 0.9,
    "style": 0.45,
    "mood": 0.4,
    "emotion": 0.7,
    "lighting": 0.7,
    "palette": 0.7,
    "temperature": 0.7,
}

DEFAULT_WEIGHT = 0.7


@dataclass
class MacroInterest:
    """A macro interest cluster."""
    macro_id: str
    label: str
    top_components: List[str]
    member_motifs: List[str]
    evidence_sessions: List[str]
    evidence_scenes: List[str]
    visual_support: float
    behavior_adjusted: Dict[str, float] = field(default_factory=dict)


def cluster_motifs(
    store: GraphStore,
    forest_id: int,
    threshold: float = 0.35,
) -> List[MacroInterest]:
    """Cluster motifs into macro interests using weighted Jaccard similarity.

    Steps:
        1. Extract motif → component set
        2. Compute pairwise weighted Jaccard similarity
        3. Connect motifs with similarity >= threshold
        4. Find connected components
        5. Generate cluster labels from top components

    Args:
        store: GraphStore instance.
        forest_id: Forest to cluster within.
        threshold: Minimum similarity to connect two motifs.

    Returns:
        List of MacroInterest clusters.
    """
    # Step 1: Gather all motifs and their components
    motifs = store.get_all_motifs(forest_id)
    if not motifs:
        return []

    motif_components: Dict[str, Dict[str, str]] = {}  # motif_key → parts
    for m in motifs:
        parts = json.loads(m["parts_json"]) if m.get("parts_json") else {}
        motif_components[m["motif_key"]] = parts

    motif_keys = list(motif_components.keys())
    n = len(motif_keys)

    if n == 0:
        return []

    # Step 2 & 3: Build adjacency via weighted Jaccard
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _weighted_jaccard(
                motif_components[motif_keys[i]],
                motif_components[motif_keys[j]],
            )
            if sim >= threshold:
                adjacency[motif_keys[i]].add(motif_keys[j])
                adjacency[motif_keys[j]].add(motif_keys[i])

    # Step 4: Connected components
    visited: Set[str] = set()
    clusters: List[Set[str]] = []

    for key in motif_keys:
        if key in visited:
            continue
        cluster: Set[str] = set()
        stack = [key]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        clusters.append(cluster)

    # Step 5: Build MacroInterest for each cluster
    total_scenes = len(store.get_all_scenes(forest_id)) or 1
    results: List[MacroInterest] = []

    for idx, cluster in enumerate(clusters):
        # Merge all components across cluster members
        component_counter: Counter = Counter()
        for motif_key in cluster:
            parts = motif_components.get(motif_key, {})
            for comp_type, comp_value in parts.items():
                # Use the prefixed format for top_components
                prefix = _component_prefix(comp_type)
                component_counter[f"{prefix}:{comp_value}"] += 1

        top_components = [
            comp for comp, _ in component_counter.most_common(8)
        ]

        # Generate label from top components
        label = _generate_label(top_components)

        # Gather evidence sessions and scenes
        evidence_sessions, evidence_scenes = _gather_evidence(
            store, forest_id, cluster,
        )

        # Visual support = fraction of scenes that contain cluster motifs
        scene_count = len(evidence_scenes)
        visual_support = round(scene_count / total_scenes, 3)

        results.append(MacroInterest(
            macro_id=f"macro_{idx + 1:03d}",
            label=label,
            top_components=top_components,
            member_motifs=[f"motif:{k}" for k in sorted(cluster)],
            evidence_sessions=sorted(evidence_sessions),
            evidence_scenes=sorted(evidence_scenes),
            visual_support=visual_support,
        ))

    # Sort by visual_support descending
    results.sort(key=lambda m: m.visual_support, reverse=True)

    return results


def apply_behavior_adjustment(
    macro_interests: List[MacroInterest],
    loyalty: float = 0.5,
    exploration: float = 0.5,
) -> List[MacroInterest]:
    """Apply behavior profile to compute core/exploration interest scores.

    Logic:
        - High loyalty → boost high-support clusters as core interest
        - High exploration → preserve low-support but coherent clusters

    Args:
        macro_interests: List from cluster_motifs().
        loyalty: Behavior loyalty score (0-1).
        exploration: Behavior exploration score (0-1).

    Returns:
        Same list with behavior_adjusted populated.
    """
    if not macro_interests:
        return macro_interests

    max_support = max(m.visual_support for m in macro_interests) or 1.0

    for m in macro_interests:
        # Normalized position (0 = weakest, 1 = strongest)
        relative_strength = m.visual_support / max_support

        # Core interest: boosted by loyalty × visual support
        core_score = relative_strength * (0.3 + 0.7 * loyalty)

        # Exploration interest: boosted for weaker clusters when exploration is high
        exploration_score = (1.0 - relative_strength) * exploration * 0.5

        m.behavior_adjusted = {
            "core_interest_score": round(core_score, 3),
            "exploration_interest_score": round(exploration_score, 3),
        }

    return macro_interests


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_jaccard(
    parts_a: Dict[str, str],
    parts_b: Dict[str, str],
) -> float:
    """Compute weighted Jaccard similarity between two motif component sets.

    Each component type has a weight. Shared components contribute their weight
    to the intersection; all components contribute to the union.
    """
    all_keys = set(parts_a.keys()) | set(parts_b.keys())
    if not all_keys:
        return 0.0

    intersection_w = 0.0
    union_w = 0.0

    for key in all_keys:
        weight = COMPONENT_WEIGHTS.get(key, DEFAULT_WEIGHT)
        val_a = parts_a.get(key)
        val_b = parts_b.get(key)

        if val_a is not None and val_b is not None:
            union_w += weight
            if val_a == val_b:
                intersection_w += weight
        else:
            union_w += weight

    return intersection_w / union_w if union_w > 0 else 0.0


def _component_prefix(comp_type: str) -> str:
    """Map component type to display prefix."""
    content_axis_types = {
        "subject_sociality",
        "media_syntheticity",
        "setting_context",
        "utility_orientation",
    }
    if comp_type in content_axis_types:
        return f"content_axis:{comp_type}"

    axis_types = {"emotion", "lighting", "palette", "temperature"}
    if comp_type in axis_types:
        # Map back to full axis name
        axis_map = {
            "emotion": "emotion_register",
            "lighting": "lighting_key",
            "palette": "palette_energy",
            "temperature": "color_temperature",
        }
        return f"axis:{axis_map.get(comp_type, comp_type)}"
    return comp_type


def _generate_label(top_components: List[str]) -> str:
    """Generate a human-readable label from top components.

    Example: ["style:soft_warm_diary", "mood:soft_positive", "setting:cafe"]
    → "soft warm cafe diary"
    """
    words = []
    for comp in top_components[:4]:  # Use top 4
        if comp.startswith("content_axis:"):
            _, axis_name, bucket = comp.split(":", 2)
            for word in (axis_name + "_" + bucket).split("_"):
                if word and word not in words:
                    words.append(word)
            continue
        # Extract just the value part
        parts = comp.split(":")
        value = parts[-1] if parts else comp
        # Clean up underscores and deduplicate
        for word in value.split("_"):
            if word and word not in words:
                words.append(word)
    return " ".join(words[:6]) if words else "unnamed_cluster"


def _gather_evidence(
    store: GraphStore,
    forest_id: int,
    cluster_motif_keys: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Find which sessions and scenes contain the cluster's motifs."""
    evidence_sessions: Set[str] = set()
    evidence_scenes: Set[str] = set()

    sessions = store.get_sessions(forest_id)
    for session in sessions:
        scenes = store.get_scenes(session["id"])
        for scene in scenes:
            scene_motifs = store.get_motifs_for_scene(scene["id"])
            for m in scene_motifs:
                if m["motif_key"] in cluster_motif_keys:
                    evidence_sessions.add(session["session_key"])
                    evidence_scenes.add(f"scene:{session['session_key']}:{scene['scene_key']}")
                    break  # One match is enough per scene

    return evidence_sessions, evidence_scenes
