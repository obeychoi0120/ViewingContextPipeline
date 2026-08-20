"""
graph_v2/build_graph.py — CLI for building the full Session Visual Interest Graph.

Takes scene observation JSON files and runs the complete pipeline:
    validate → triplet → motif → SQLite store → aggregate → cluster → output

Usage:
    # From observation files (no Qwen needed)
    python -m graph_v2.build_graph \
        --session-id session_001 \
        --observations scene_001.json scene_002.json \
        --out final_output.json

    # With custom DB path
    python -m graph_v2.build_graph \
        --session-id session_demo \
        --observations graph_v2/example_observation.json \
        --db my_graph.db \
        --out demo_output.json

    # Multiple sessions
    python -m graph_v2.build_graph \
        --session-id session_001 session_002 \
        --observations obs_s1.json obs_s2.json \
        --out multi_session_output.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any

from .validator import validate_observation
from .triplet_builder import build_triplets, Triplet
from .motif_builder import build_motifs
from .graph_store import GraphStore
from .output_builder import build_final_output


def _node_type_from_id(node_id: str) -> str:
    """Infer node type from the string id prefix."""
    prefix_map = {
        "forest:": "SessionForest",
        "session:": "Session",
        "scene:": "Scene",
        "content_axis:": "ContentAxis",
        "style_cue:": "StyleCue",
        "axis:": "VisualAxis",
        "scene_type:": "SceneType",
        "people_density:": "PeopleDensity",
        "face_prominence:": "FaceProminence",
        "mood_bin:": "MoodBin",
        "scene_function:": "SceneFunction",
        "style:": "Style",
        "mood:": "Mood",
        "setting:": "Setting",
        "entity_instance:": "EntityInstance",
        "entity:": "Entity",
        "category:": "EntityCategory",
        "role:": "EntityRole",
        "motif:": "Motif",
    }
    for prefix, ntype in prefix_map.items():
        if node_id.startswith(prefix):
            return ntype
    return "Unknown"


def _label_from_id(node_id: str) -> str:
    """Extract human-readable label from a node id string."""
    parts = node_id.split(":")
    return parts[-1] if parts else node_id


def ingest_scene(
    store: GraphStore,
    forest_id: int,
    session_db_id: int,
    session_key: str,
    scene_key: str,
    scene_index: int,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Ingest a single scene observation into the graph store.

    Args:
        store: GraphStore instance.
        forest_id: SQLite forest id.
        session_db_id: SQLite session id.
        session_key: Session key string (e.g. "session_001").
        scene_key: Scene key string (e.g. "scene_001").
        scene_index: 0-based index of scene within session.
        observation: Validated observation dict.

    Returns:
        Dict with stats: n_triplets, n_motifs, warnings.
    """
    warnings: List[str] = []

    # 1. Validate
    obs, val_warnings = validate_observation(observation)
    warnings.extend(val_warnings)

    # 2. Store scene with raw observation
    scene_db_id = store.add_scene(
        session_id=session_db_id,
        scene_key=scene_key,
        scene_index=scene_index,
        scene_summary=obs.get("scene_summary", ""),
        observation_json=json.dumps(obs, ensure_ascii=False),
    )

    # 3. Build triplets
    triplets = build_triplets(
        session_id=session_key,
        scene_id=scene_key,
        observation=obs,
        forest_id="current",
    )

    # 4. Create nodes and edges in store
    node_cache: Dict[str, int] = {}  # string id → SQLite node id

    for t in triplets:
        subj_db_id = _ensure_node(store, t.subject, node_cache,
                                   forest_id, session_db_id, scene_db_id, obs)
        obj_db_id = _ensure_node(store, t.object, node_cache,
                                  forest_id, session_db_id, scene_db_id, obs)
        store.add_edge(
            subject_node_id=subj_db_id,
            predicate=t.predicate,
            object_node_id=obj_db_id,
            forest_id=forest_id,
            session_id=session_db_id,
            scene_id=scene_db_id,
        )

    # 5. Build motifs
    motifs = build_motifs(obs)
    for motif in motifs:
        # Create motif node
        motif_unique_key = f"motif::{motif.key}"
        motif_node_id = store.get_or_create_node(
            node_type="Motif",
            label=motif.key,
            unique_key=motif_unique_key,
            forest_id=forest_id,
        )

        # Create motif record
        motif_db_id = store.get_or_create_motif(
            motif_key=motif.key,
            motif_type=motif.motif_type,
            parts=motif.parts,
            node_id=motif_node_id,
        )

        # Link scene ↔ motif
        store.link_scene_motif(scene_db_id, motif_db_id)

        # Create HAS_COMPONENT edges from motif to component nodes
        scene_node_key = f"scene:{session_key}:{scene_key}"
        scene_node_id = node_cache.get(scene_node_key)
        if scene_node_id:
            store.add_edge(
                subject_node_id=scene_node_id,
                predicate="HAS_MOTIF",
                object_node_id=motif_node_id,
                forest_id=forest_id,
                session_id=session_db_id,
                scene_id=scene_db_id,
                source="motif_builder",
            )

        # HAS_COMPONENT edges
        for comp_type, comp_value in motif.parts.items():
            comp_key = f"{comp_type}:{comp_value}"
            # Map to existing node types where possible
            comp_node_type = _component_node_type(comp_type)
            if comp_node_type == "ContentAxis":
                comp_label = f"{comp_type}={comp_value}"
                comp_unique_key = f"{comp_node_type.lower()}::{comp_label}"
            else:
                comp_label = comp_value
                comp_unique_key = f"{comp_node_type.lower()}::{comp_value}"
            comp_node_id = store.get_or_create_node(
                node_type=comp_node_type,
                label=comp_label,
                unique_key=comp_unique_key,
                forest_id=forest_id,
            )
            store.add_edge(
                subject_node_id=motif_node_id,
                predicate="HAS_COMPONENT",
                object_node_id=comp_node_id,
                forest_id=forest_id,
                source="motif_builder",
            )

    return {
        "n_triplets": len(triplets),
        "n_motifs": len(motifs),
        "warnings": warnings,
    }


def _ensure_node(
    store: GraphStore,
    node_str_id: str,
    cache: Dict[str, int],
    forest_id: int,
    session_db_id: int,
    scene_db_id: int,
    observation: Dict[str, Any],
) -> int:
    """Get or create a node, using cache for de-duplication."""
    if node_str_id in cache:
        return cache[node_str_id]

    node_type = _node_type_from_id(node_str_id)
    label = _label_from_id(node_str_id)

    # Build unique key for de-duplication
    # Global nodes (style, mood, entity, etc.) share across scenes
    # Instance nodes (entity_instance) are scene-specific
    if node_type == "EntityInstance":
        unique_key = f"entity_instance::{node_str_id}"
        # Extract local_id from the string
        parts = node_str_id.split(":")
        local_id = parts[-1] if len(parts) >= 4 else label

        # Get entity properties from observation
        props = None
        for ent in observation.get("entities", []):
            if ent.get("local_id") == local_id:
                props = {
                    "name": ent.get("name", ""),
                    "category": ent.get("category", ""),
                    "role": ent.get("role", ""),
                }
                break

        db_id = store.get_or_create_node(
            node_type=node_type,
            label=local_id,
            unique_key=unique_key,
            forest_id=forest_id,
            session_id=session_db_id,
            scene_id=scene_db_id,
            local_id=local_id,
            properties=props,
        )
    elif node_type in ("SessionForest", "Session", "Scene"):
        unique_key = f"{node_type.lower()}::{node_str_id}"
        db_id = store.get_or_create_node(
            node_type=node_type,
            label=label,
            unique_key=unique_key,
            forest_id=forest_id,
            session_id=session_db_id if node_type != "SessionForest" else None,
            scene_id=scene_db_id if node_type == "Scene" else None,
        )
    else:
        # Global nodes: style, mood, setting, entity, category, role, axis, etc.
        unique_key = f"{node_type.lower()}::{label}"
        db_id = store.get_or_create_node(
            node_type=node_type,
            label=label,
            unique_key=unique_key,
            forest_id=forest_id,
        )

    cache[node_str_id] = db_id
    return db_id


def _component_node_type(comp_type: str) -> str:
    """Map motif component type to graph node type."""
    return {
        "style": "Style",
        "mood": "Mood",
        "setting": "Setting",
        "scene_function": "SceneFunction",
        "entity": "Entity",
        "action": "Action",
        "subject_sociality": "ContentAxis",
        "media_syntheticity": "ContentAxis",
        "setting_context": "ContentAxis",
        "utility_orientation": "ContentAxis",
        "emotion": "VisualAxis",
        "lighting": "VisualAxis",
        "palette": "VisualAxis",
        "temperature": "VisualAxis",
    }.get(comp_type, "Unknown")


def load_observations(file_path: str) -> List[Dict[str, Any]]:
    """Load observations from a visual graph JSONL file.

    The input is expected to be *_visual_graph.jsonl:
        - Optional first-line video_metadata record
        - One visual graph record per scene with vlm_visual_graph
    """
    observations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("_type") == "video_metadata":
                continue
            observations.append(_visual_graph_record_to_observation(item, len(observations)))
    return observations


def _visual_graph_record_to_observation(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    if "vlm_visual_graph" not in item:
        raise ValueError("visual graph JSONL record is missing vlm_visual_graph")
    scene_idx = item.get("scene_idx", index)
    if isinstance(scene_idx, int):
        scene_id = f"scene_{scene_idx + 1:03d}"
    else:
        scene_id = str(scene_idx)
    return {
        "scene_id": str(item.get("scene_id") or scene_id),
        "observation": item.get("vlm_visual_graph"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build Session Visual Interest Graph from observations (graph_v2)"
    )
    parser.add_argument(
        "--session-id", nargs="+", required=True,
        help="Session ID(s). If multiple, pair with multiple --observations files.",
    )
    parser.add_argument(
        "--observations", nargs="+", required=True,
        help="Observation JSON file(s). Each file contains scenes for one session.",
    )
    parser.add_argument(
        "--db", default=":memory:",
        help="SQLite database path (default: in-memory)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--forest-id", default="current",
        help="Forest key (default: 'current')",
    )
    parser.add_argument(
        "--loyalty", type=float, default=0.5,
        help="Behavior loyalty score (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--exploration", type=float, default=0.5,
        help="Behavior exploration score (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Clustering similarity threshold (default: 0.35)",
    )

    args = parser.parse_args()

    # Pair session IDs with observation files
    session_ids = args.session_id
    obs_files = args.observations

    # If one session with multiple obs files, treat all as same session
    if len(session_ids) == 1 and len(obs_files) > 1:
        session_ids = session_ids * len(obs_files)
    elif len(session_ids) != len(obs_files):
        # If counts differ, each obs file gets its session_id or we expand
        if len(obs_files) == 1:
            obs_files = obs_files * len(session_ids)
        else:
            parser.error(
                f"Number of --session-id ({len(session_ids)}) must match "
                f"--observations ({len(obs_files)}), or provide exactly 1 of either."
            )

    # Initialize store
    store = GraphStore(args.db)
    forest_db_id = store.get_or_create_forest(
        forest_key=args.forest_id,
        behavior_loyalty=args.loyalty,
        behavior_exploration=args.exploration,
    )

    print(f"=== graph_v2 Build Graph ===")
    print(f"  Forest: {args.forest_id}")
    print(f"  DB: {args.db}")
    print(f"  Sessions: {len(set(session_ids))}")
    print(f"  Observation files: {len(obs_files)}")
    print()

    total_scenes = 0
    total_triplets = 0
    total_motifs = 0

    for session_key, obs_file in zip(session_ids, obs_files):
        print(f"--- Session: {session_key} ---")
        print(f"  Loading: {obs_file}")

        scenes = load_observations(obs_file)
        session_db_id = store.get_or_create_session(
            forest_id=forest_db_id,
            session_key=session_key,
        )

        for idx, scene_data in enumerate(scenes):
            scene_key = scene_data.get("scene_id", f"scene_{idx + 1:03d}")
            obs = scene_data.get("observation")

            if obs is None:
                print(f"  [SKIP] {scene_key}: no observation")
                continue

            stats = ingest_scene(
                store=store,
                forest_id=forest_db_id,
                session_db_id=session_db_id,
                session_key=session_key,
                scene_key=scene_key,
                scene_index=idx,
                observation=obs,
            )

            total_scenes += 1
            total_triplets += stats["n_triplets"]
            total_motifs += stats["n_motifs"]

            if stats["warnings"]:
                for w in stats["warnings"]:
                    print(f"    [WARN] {w}")

            print(f"  {scene_key}: {stats['n_triplets']} triplets, {stats['n_motifs']} motifs")

    # Build final output
    print()
    print(f"=== Building final output ===")
    output = build_final_output(
        store=store,
        forest_id=forest_db_id,
        clustering_threshold=args.threshold,
    )

    # Write output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    n_nodes = store.count_nodes()
    n_edges = store.count_edges()
    n_motifs_db = store.count_motifs()
    n_macros = len(output.get("macro_interests", []))

    print(f"\n=== SUMMARY ===")
    print(f"  Scenes:    {total_scenes}")
    print(f"  Nodes:     {n_nodes}")
    print(f"  Edges:     {n_edges}")
    print(f"  Motifs:    {n_motifs_db}")
    print(f"  Macros:    {n_macros}")
    print(f"  Output:    {args.out}")

    if args.db != ":memory:":
        print(f"  Database:  {args.db}")

    store.close()


if __name__ == "__main__":
    main()
