"""Reproduce a random content sample and descriptive output checks (no model calls)."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from extraction.recovery import fingerprint
from extraction.scene_storage import read_scene_records

OUT = Path(__file__).resolve().parent
RUN = ROOT / "artifacts/runs/Full_v3_260918"
SEED = 20260921
ARMS = [(rep, model) for rep in ("description", "graph") for model in ("qwen", "gemini")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": round(statistics.mean(values), 2), "max": max(values)} if values else {"n": 0}


def main():
    populations = [{p.stem for p in (RUN / "extraction" / rep / model / "scenes").glob("microlens_*.jsonl")}
                   for rep, model in ARMS]
    population = sorted(set.union(*populations))
    selected = random.Random(SEED).sample(population, 100)
    summary_hash = sha(ROOT / "prompts/graph_summary_v4.md")
    manifest = {"run": str(RUN.relative_to(ROOT)), "seed": SEED, "method": "Python random.Random(seed).sample(sorted(union of scene content IDs), 100), without replacement; missing summaries are retained, never replaced",
                "population_n": len(population), "population_hash": fingerprint(population),
                "scene_population_sizes": {f"{r}_{m}": len(p) for (r, m), p in zip(ARMS, populations)},
                "sample_n": len(selected), "draw_order": selected, "sorted_ids": sorted(selected),
                "summary_model": "gemini", "prompt_hashes": {p.name: sha(p) for p in sorted((ROOT / "prompts").glob("*.md"))},
                "files": []}
    packets = []
    all_metrics = {}
    for rep, model in ARMS:
        key = f"{rep}_{model}"
        counters = Counter()
        words, scene_words, entity_n, relation_n = [], [], [], []
        detail = []
        for cid in sorted(selected):
            base = RUN / "extraction" / rep / model
            scene_path = base / "scenes" / f"{cid}.jsonl"
            summary_path = base / "summaries/gemini" / f"{cid}.json"
            if not scene_path.exists():
                counters["missing_scene_files"] += 1
                continue
            rows = read_scene_records(scene_path)
            counters["scene_files"] += 1
            counters["scene_records"] += len(rows)
            counters["scene_records_with_provenance"] += sum("provenance" in x for x in rows)
            d = json.loads(summary_path.read_text()) if summary_path.exists() else None
            packet = {"id": cid, "arm": key, "scene_path": str(scene_path.relative_to(ROOT)),
                      "summary_path": str(summary_path.relative_to(ROOT)), "scenes": rows, "summary": d}
            packets.append(packet)
            entry = {"id": cid, "arm": key, "scene_path": str(scene_path.relative_to(ROOT)), "scene_sha256": sha(scene_path),
                     "scene_count": len(rows), "summary_path": str(summary_path.relative_to(ROOT)), "summary_exists": d is not None}
            if d is not None:
                prov = d.get("provenance", {})
                n = len(d.get("text", "").split())
                words.append(n)
                counters["summary_files"] += 1
                counters["summary_status_" + str(d.get("status"))] += 1
                counters["summary_prompt_hash_match"] += prov.get("prompt_hash") == summary_hash
                counters["summary_input_hash_match"] += prov.get("scene_input_hash") == fingerprint(rows)
                counters["summary_over_150_words"] += n > 150
                counters["summary_under_100_words"] += n < 100
                entry.update(summary_sha256=sha(summary_path), summary_words=n,
                             summary_input_hash_match=prov.get("scene_input_hash") == fingerprint(rows),
                             summary_prompt_hash_match=prov.get("prompt_hash") == summary_hash)
            else:
                counters["missing_summaries"] += 1
            for row in rows:
                if rep == "description":
                    n = len(row.get("description", "").split())
                    scene_words.append(n)
                    counters["description_over_350_words"] += n > 350
                    counters["description_under_200_words"] += n < 200
                    continue
                graph = row.get("graph")
                if not isinstance(graph, dict):
                    counters["raw_graph_records"] += 1
                    detail.append({"id": cid, "scene_idx": row["scene_idx"], "issue": "raw_graph"})
                    continue
                ents, rels = graph.get("entities", []), graph.get("relations", [])
                entity_n.append(len(ents)); relation_n.append(len(rels))
                counters["graph_records_with_context"] += "context" in graph
                counters["graph_over_4_entities"] += len(ents) > 4
                counters["graph_exactly_4_entities"] += len(ents) == 4
                counters["graph_over_4_relations"] += len(rels) > 4
                ids = [e["id"] for e in ents]
                if len(ids) != len(set(ids)):
                    counters["graph_duplicate_id_records"] += 1
                    detail.append({"id": cid, "scene_idx": row["scene_idx"], "issue": "duplicate_id"})
                bad = [r for r in rels if r["subject_id"] not in ids or r["object_id"] not in ids]
                self_rels = [r for r in rels if r["subject_id"] == r["object_id"]]
                if bad:
                    counters["graph_unresolved_reference_records"] += 1
                    detail.append({"id": cid, "scene_idx": row["scene_idx"], "issue": "unresolved_reference", "relations": bad})
                if self_rels:
                    counters["graph_self_relation_records"] += 1
                    detail.append({"id": cid, "scene_idx": row["scene_idx"], "issue": "self_relation", "relations": self_rels})
                counters["graph_entities"] += len(ents)
                counters["graph_relations"] += len(rels)
                counters["graph_entities_over_3_attributes"] += sum(len(e.get("attributes", [])) > 3 for e in ents)
            manifest["files"].append(entry)
        all_metrics[key] = {"counts": dict(counters), "summary_words": describe(words), "scene_words": describe(scene_words),
                            "entities_per_graph": describe(entity_n), "relations_per_graph": describe(relation_n), "structural_details": detail}
    assert len(selected) == len(set(selected)) == 100
    assert len(packets) == 400
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n")
    packets.sort(key=lambda x: (x["id"], x["arm"]))
    tmp = Path("/tmp/random100_prompt_review_260921_packets.jsonl")
    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets))
    print(json.dumps({"population_n": len(population), "seed": SEED, "sample": sorted(selected), "packet_path": str(tmp), "metrics": all_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
