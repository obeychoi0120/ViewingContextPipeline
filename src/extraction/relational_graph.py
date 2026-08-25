from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from extraction.backends import VLMBackend
from extraction.evidence import (
    build_scene_evidence,
    load_images,
)
from extraction.json_repair import extract_json


GRAPH_SCENE_SCHEMA = "scene-relational-graph/v1"
GRAPH_SUMMARY_SCHEMA = "graph-video-summary/v1"
TRIPLE_FIELDS = {"subject_id", "subject", "relation", "object_id", "object"}
CONTEXT_RELATIONS = {"SETTING", "SCENE_FUNCTION", "MOOD", "MEDIA_FORM"}


class RelationalGraphError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ontology:
    ontology_id: str
    status: str
    relations: frozenset[str]
    entity_reference_relations: frozenset[str]


def ontology_from_document(value: dict[str, Any]) -> Ontology:
    if value.get("ontology_id") != "relational-graph-ontology/v1":
        raise RelationalGraphError("ontology_id must be relational-graph-ontology/v1")
    if value.get("status") not in {"provisional", "final"}:
        raise RelationalGraphError("ontology status must be provisional or final")
    relations = value.get("relations")
    references = value.get("entity_reference_relations")
    if not isinstance(relations, list) or not relations or not all(isinstance(item, str) and item for item in relations):
        raise RelationalGraphError("ontology relations must be a non-empty string list")
    if not isinstance(references, list) or not set(references).issubset(set(relations)):
        raise RelationalGraphError("invalid entity_reference_relations")
    return Ontology(
        ontology_id=value["ontology_id"],
        status=value["status"],
        relations=frozenset(relations),
        entity_reference_relations=frozenset(references),
    )


def parse_graph_output(text: str, ontology: Ontology) -> list[dict[str, Any]]:
    payload = extract_json(text)
    if payload is None:
        raise RelationalGraphError("VLM output does not contain a JSON object")
    if set(payload) != {"triples"} or not isinstance(payload["triples"], list) or not payload["triples"]:
        raise RelationalGraphError("VLM output must contain one non-empty triples list")
    triples: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["triples"]):
        if not isinstance(raw, dict) or set(raw) != TRIPLE_FIELDS:
            raise RelationalGraphError(f"triple {index} must contain exactly {sorted(TRIPLE_FIELDS)}")
        subject_id = _text(raw["subject_id"], f"triple {index} subject_id")
        subject = _text(raw["subject"], f"triple {index} subject")
        relation = _text(raw["relation"], f"triple {index} relation").upper()
        object_text = _text(raw["object"], f"triple {index} object")
        object_id = raw["object_id"]
        if object_id is not None:
            object_id = _text(object_id, f"triple {index} object_id")
        if relation not in ontology.relations:
            raise RelationalGraphError(f"triple {index} uses unknown relation {relation!r}")
        if relation in ontology.entity_reference_relations:
            if object_id is None:
                raise RelationalGraphError(f"triple {index} {relation} requires object_id")
        elif object_id is not None:
            raise RelationalGraphError(f"triple {index} {relation} requires null object_id")
        if relation in CONTEXT_RELATIONS and subject_id != "scene":
            raise RelationalGraphError(f"triple {index} {relation} must use subject_id='scene'")
        triples.append({
            "subject_id": subject_id,
            "subject": subject,
            "relation": relation,
            "object_id": object_id,
            "object": object_text,
        })
    entity_labels: dict[str, str] = {}
    for row in triples:
        subject_id = row["subject_id"]
        if subject_id == "scene":
            continue
        previous = entity_labels.setdefault(subject_id, row["subject"])
        if previous != row["subject"]:
            raise RelationalGraphError(
                f"entity id {subject_id!r} uses conflicting labels {previous!r} and {row['subject']!r}"
            )
    declared = set(entity_labels)
    for index, row in enumerate(triples):
        if row["relation"] in ontology.entity_reference_relations and row["object_id"] not in declared:
            raise RelationalGraphError(
                f"triple {index} has dangling {row['relation']} target {row['object_id']!r}"
            )
    return triples


def extract_scene_graphs(
    *,
    content_id: str,
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
    timestamp_json_path: str | Path,
    backend: VLMBackend,
    prompt: str,
    ontology: Ontology,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in build_scene_evidence(scenes, frames_dir, timestamp_json_path):
        scene_idx = scene["scene_idx"]
        keyframes = scene["keyframes"]
        image_paths = scene["image_paths"]
        if not keyframes or len(image_paths) != len(keyframes):
            raise RelationalGraphError(
                f"scene {scene_idx} has {len(image_paths)} of {len(keyframes)} keyframes"
            )
        raw = backend.generate(load_images(image_paths), prompt, max_new_tokens)
        triples = parse_graph_output(raw, ontology)
        records.append({
            "schema_version": GRAPH_SCENE_SCHEMA,
            "content_id": content_id,
            "scene_idx": scene_idx,
            "keyframes": keyframes,
            "image_paths": image_paths,
            "triples": triples,
        })
    if not records:
        raise RelationalGraphError("video has no scenes")
    return records


def graph_summary_prompt(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise RelationalGraphError("graph summary requires scene records")
    parts: list[str] = []
    for record in records:
        triples = record.get("triples")
        if record.get("schema_version") != GRAPH_SCENE_SCHEMA or not isinstance(triples, list) or not triples:
            raise RelationalGraphError("graph summary received an invalid scene record")
        lines = [f"Scene {record['scene_idx']}:"]
        for triple in triples:
            lines.append(f"- {triple['subject']} - {triple['relation']} - {triple['object']}")
        parts.append("\n".join(lines))
    return template.format(scenes="\n\n".join(parts))


def validate_summary(text: str) -> str:
    summary = str(text or "").strip()
    words = len(summary.split())
    if not 150 <= words <= 300:
        raise RelationalGraphError(f"video summary must contain 150-300 words; got {words}")
    return summary


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RelationalGraphError(f"{label} must be non-empty")
    return text
