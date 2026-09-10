"""Report recovery provenance separately from structural scene coverage."""
from collections import Counter
import json

from extraction.recovery import fingerprint
from pipeline_runtime import read_jsonl


def recovery_report(run_root, *, branches=None):
    report = {}
    root = run_root / "extraction"
    sources = {"graph_qwen": "graph/qwen", "graph_gemini": "graph/gemini", "desc": "description"}
    selected = [path for branch, path in sources.items() if branches is None or branch in branches]
    directories = [root / branch / step for branch in selected
                   for step in ("scenes", "summaries") if (root / branch / step).is_dir()]
    for output_dir in directories:
        directory = output_dir / ".recovery"
        counts = Counter()
        attempts = 0
        origins = {}
        for path in directory.glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            cycle = document["cycles"][-1]
            for previous in document["cycles"]:
                if "output" in previous:
                    output = previous["output"]
                    mode = previous["attempts"][-1]["repair_mode"]
                    origins[fingerprint(output)] = ("raw_fallback" if output.get("status") == "raw_fallback"
                                                    else "repaired" if mode == "repaired" else "native")
            attempts += len(cycle["attempts"])
            status = cycle["status"]
            if status == "complete":
                mode = cycle["attempts"][-1]["repair_mode"]
                counts["repaired" if mode == "repaired" else "native"] += 1
            elif status in {"raw_fallback", "failed"}:
                counts[status] += 1
            else:
                counts["interrupted"] += 1
        artifact_counts = Counter()
        scene_counts = Counter()
        for path in output_dir.glob("*.jsonl" if output_dir.name == "scenes" else "*.json"):
            rows = read_jsonl(path) if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
            for row in rows:
                mode = ("raw_fallback" if row.get("status") == "raw_fallback"
                        else origins.get(fingerprint(row), "legacy_unknown"))
                artifact_counts[mode] += 1
            if output_dir.name == "summaries":
                state_path = output_dir / ".inputs" / path.name
                if state_path.exists():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    scene_counts.update({key: state.get(key, 0) for key in
                                         ("scene_count", "normal_scene_count", "raw_scene_count")})
                    scene_counts["summaries_with_raw_input"] += int(state["raw_scene_count"] > 0)
                else:
                    scene_counts["legacy_unknown_summaries"] += 1
        report[output_dir.relative_to(root).as_posix()] = {
            "counts": dict(counts), "current_cycle_attempts": attempts,
            "artifact_counts": dict(artifact_counts), "summary_inputs": dict(scene_counts),
        }
    return report
