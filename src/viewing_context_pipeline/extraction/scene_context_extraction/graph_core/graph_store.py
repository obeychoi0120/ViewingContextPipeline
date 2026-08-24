"""
graph_v2/graph_store.py — SQLite-based graph storage.

Stores nodes, edges, motifs, sessions, scenes, and forests in a relational
schema designed for the Session Visual Interest Graph.

Provides a triplet_view for human-readable debugging.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Any


class GraphStore:
    """SQLite graph store for the Session Visual Interest Graph."""

    def __init__(self, db_path: str = ":memory:"):
        """Open or create the graph database.

        Args:
            db_path: Path to SQLite file, or ":memory:" for in-memory DB.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_tables(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS forests (
            id                    INTEGER PRIMARY KEY,
            forest_key            TEXT UNIQUE,
            behavior_loyalty      REAL DEFAULT 0.5,
            behavior_exploration  REAL DEFAULT 0.5,
            properties_json       TEXT
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY,
            forest_id       INTEGER REFERENCES forests(id),
            session_key     TEXT,
            session_index   INTEGER,
            properties_json TEXT,
            UNIQUE(forest_id, session_key)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scenes (
            id                INTEGER PRIMARY KEY,
            session_id        INTEGER REFERENCES sessions(id),
            scene_key         TEXT,
            scene_index       INTEGER,
            scene_summary     TEXT,
            observation_json  TEXT,
            UNIQUE(session_id, scene_key)
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id              INTEGER PRIMARY KEY,
            node_type       TEXT NOT NULL,
            label           TEXT NOT NULL,
            unique_key      TEXT UNIQUE NOT NULL,
            forest_id       INTEGER,
            session_id      INTEGER,
            scene_id        INTEGER,
            local_id        TEXT,
            properties_json TEXT
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id              INTEGER PRIMARY KEY,
            subject_node_id INTEGER NOT NULL REFERENCES nodes(id),
            predicate       TEXT NOT NULL,
            object_node_id  INTEGER NOT NULL REFERENCES nodes(id),
            forest_id       INTEGER,
            session_id      INTEGER,
            scene_id        INTEGER,
            source          TEXT DEFAULT 'qwen_observation',
            properties_json TEXT
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS motifs (
            id          INTEGER PRIMARY KEY,
            node_id     INTEGER REFERENCES nodes(id),
            motif_key   TEXT,
            motif_type  TEXT,
            parts_json  TEXT,
            UNIQUE(motif_key)
        )""")

        # Scene-motif linking table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS scene_motifs (
            id        INTEGER PRIMARY KEY,
            scene_id  INTEGER REFERENCES scenes(id),
            motif_id  INTEGER REFERENCES motifs(id),
            UNIQUE(scene_id, motif_id)
        )""")

        # Triplet view for debugging
        cur.execute("""
        CREATE VIEW IF NOT EXISTS triplet_view AS
        SELECT
            e.id          AS edge_id,
            sn.id         AS subject_id,
            sn.node_type  AS subject_type,
            sn.label      AS subject_label,
            e.predicate,
            ob.id         AS object_id,
            ob.node_type  AS object_type,
            ob.label      AS object_label,
            e.forest_id,
            e.session_id,
            e.scene_id,
            e.source
        FROM edges e
        JOIN nodes sn ON e.subject_node_id = sn.id
        JOIN nodes ob ON e.object_node_id = ob.id
        """)

        self.conn.commit()

    # ------------------------------------------------------------------
    # Forest CRUD
    # ------------------------------------------------------------------
    def get_or_create_forest(
        self,
        forest_key: str = "current",
        behavior_loyalty: float = 0.5,
        behavior_exploration: float = 0.5,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM forests WHERE forest_key = ?", (forest_key,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute(
            "INSERT INTO forests (forest_key, behavior_loyalty, behavior_exploration) "
            "VALUES (?, ?, ?)",
            (forest_key, behavior_loyalty, behavior_exploration),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_forest(self, forest_id: int) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM forests WHERE id = ?", (forest_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------
    def get_or_create_session(
        self,
        forest_id: int,
        session_key: str,
        session_index: int = 0,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id FROM sessions WHERE forest_id = ? AND session_key = ?",
            (forest_id, session_key),
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute(
            "INSERT INTO sessions (forest_id, session_key, session_index) "
            "VALUES (?, ?, ?)",
            (forest_id, session_key, session_index),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_sessions(self, forest_id: int) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE forest_id = ? ORDER BY session_index", (forest_id,))
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Scene CRUD
    # ------------------------------------------------------------------
    def add_scene(
        self,
        session_id: int,
        scene_key: str,
        scene_index: int,
        scene_summary: str,
        observation_json: str,
    ) -> int:
        cur = self.conn.cursor()
        # Check if already exists
        cur.execute(
            "SELECT id FROM scenes WHERE session_id = ? AND scene_key = ?",
            (session_id, scene_key),
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute(
            "INSERT INTO scenes (session_id, scene_key, scene_index, scene_summary, observation_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, scene_key, scene_index, scene_summary, observation_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_scenes(self, session_id: int) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM scenes WHERE session_id = ? ORDER BY scene_index", (session_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_all_scenes(self, forest_id: int) -> List[Dict]:
        """Get all scenes across all sessions in a forest."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT sc.* FROM scenes sc
            JOIN sessions s ON sc.session_id = s.id
            WHERE s.forest_id = ?
            ORDER BY s.session_index, sc.scene_index
        """, (forest_id,))
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------
    def get_or_create_node(
        self,
        node_type: str,
        label: str,
        unique_key: str,
        forest_id: Optional[int] = None,
        session_id: Optional[int] = None,
        scene_id: Optional[int] = None,
        local_id: Optional[str] = None,
        properties: Optional[Dict] = None,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM nodes WHERE unique_key = ?", (unique_key,))
        row = cur.fetchone()
        if row:
            return row["id"]
        props_json = json.dumps(properties, ensure_ascii=False) if properties else None
        cur.execute(
            "INSERT INTO nodes (node_type, label, unique_key, forest_id, session_id, "
            "scene_id, local_id, properties_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (node_type, label, unique_key, forest_id, session_id, scene_id,
             local_id, props_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_nodes(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes ORDER BY id")
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------
    def add_edge(
        self,
        subject_node_id: int,
        predicate: str,
        object_node_id: int,
        forest_id: Optional[int] = None,
        session_id: Optional[int] = None,
        scene_id: Optional[int] = None,
        source: str = "qwen_observation",
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO edges (subject_node_id, predicate, object_node_id, "
            "forest_id, session_id, scene_id, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (subject_node_id, predicate, object_node_id, forest_id, session_id,
             scene_id, source),
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Motif CRUD
    # ------------------------------------------------------------------
    def get_or_create_motif(
        self,
        motif_key: str,
        motif_type: str,
        parts: Dict[str, str],
        node_id: Optional[int] = None,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM motifs WHERE motif_key = ?", (motif_key,))
        row = cur.fetchone()
        if row:
            return row["id"]
        parts_json = json.dumps(parts, ensure_ascii=False)
        cur.execute(
            "INSERT INTO motifs (node_id, motif_key, motif_type, parts_json) "
            "VALUES (?, ?, ?, ?)",
            (node_id, motif_key, motif_type, parts_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def link_scene_motif(self, scene_id: int, motif_id: int):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO scene_motifs (scene_id, motif_id) VALUES (?, ?)",
            (scene_id, motif_id),
        )
        self.conn.commit()

    def get_motifs_for_scene(self, scene_id: int) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT m.* FROM motifs m
            JOIN scene_motifs sm ON m.id = sm.motif_id
            WHERE sm.scene_id = ?
        """, (scene_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_all_motifs(self, forest_id: Optional[int] = None) -> List[Dict]:
        cur = self.conn.cursor()
        if forest_id is not None:
            cur.execute("""
                SELECT DISTINCT m.* FROM motifs m
                JOIN scene_motifs sm ON m.id = sm.motif_id
                JOIN scenes sc ON sm.scene_id = sc.id
                JOIN sessions s ON sc.session_id = s.id
                WHERE s.forest_id = ?
            """, (forest_id,))
        else:
            cur.execute("SELECT * FROM motifs ORDER BY id")
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Triplet view
    # ------------------------------------------------------------------
    def get_triplet_view(self, limit: int = 200) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM triplet_view LIMIT {limit}")
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_graph_dict(self) -> Dict[str, Any]:
        """Export all nodes and edges as a JSON-serializable dict."""
        nodes = []
        for n in self.get_all_nodes():
            node = {
                "id": n["id"],
                "node_type": n["node_type"],
                "label": n["label"],
                "unique_key": n["unique_key"],
            }
            if n["properties_json"]:
                node["properties"] = json.loads(n["properties_json"])
            nodes.append(node)

        triplets = []
        for t in self.get_triplet_view(limit=10000):
            triplets.append({
                "subject_id": t["subject_id"],
                "subject_type": t["subject_type"],
                "subject_label": t["subject_label"],
                "predicate": t["predicate"],
                "object_id": t["object_id"],
                "object_type": t["object_type"],
                "object_label": t["object_label"],
            })

        return {"nodes": nodes, "triplets": triplets}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def count_nodes(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes")
        return cur.fetchone()[0]

    def count_edges(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM edges")
        return cur.fetchone()[0]

    def count_motifs(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM motifs")
        return cur.fetchone()[0]
