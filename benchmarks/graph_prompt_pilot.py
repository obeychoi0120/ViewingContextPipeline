"""Opt-in first-100 Graph pilot; never promotes a prompt or regenerates the full catalog."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from extraction import steps
from extraction.recovery import file_fingerprint
from extraction.scene_storage import read_scene_records
from extraction.step_support import scene_generation_rows, visual_rows
from extraction.summary_executor import summary_model_from_document
from extraction.structured_output import OutputValidationError, validate_graph_structure
from model_provenance import local_model_identity
from pipeline_runtime import RunContext, read_json, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class PilotContext(RunContext):
    selected_cohort: dict = field(default_factory=dict)

    def require_ready_cohort(self):
        return self.selected_cohort


def select_cohort(cohort, count=100):
    catalog = sorted(cohort["catalog"], key=lambda row: str(row["content_id"]))[:count]
    if len(catalog) != count:
        raise ValueError(f"pilot requires {count} videos")
    ids = {str(row["content_id"]) for row in catalog}
    if len(ids) != count:
        raise ValueError("duplicate pilot content IDs")
    return {**cohort, "catalog": catalog,
            "metadata_titles": [r for r in cohort.get("metadata_titles", [])
                                if str(r["content_id"]) in ids]}


def prepare(context, baseline):
    if context.run_root == baseline.run_root:
        raise ValueError("pilot must use a separate run from its baseline")
    cohort = select_cohort(context.require_ready_cohort())
    pilot = PilotContext(context.root, context.run_id, context.config, context.run_root, cohort)
    model = local_model_identity(context.path("models", "qwen"))
    blockers, evidence, baselines = set(), [], []
    if not context.path("models", "qwen").is_dir():
        blockers.add("configured Qwen checkpoint directory is missing")
    try:
        if version("vllm") != "0.28.0":
            blockers.add("vllm must be exactly 0.28.0")
    except PackageNotFoundError:
        blockers.add("vllm 0.28.0 is not installed")
    settings = context.config["extraction"]
    if settings["graph"] != {"scene_max_new_tokens": 1024, "summary_max_new_tokens": 2048}:
        blockers.add("Graph token budgets must remain 1024/2048")
    for visual in visual_rows(pilot):
        cid = visual["content_id"]
        path = baseline.scene_arm_dir("graph_qwen") / f"{cid}.jsonl"
        if not path.is_file():
            blockers.add(f"missing baseline: {cid}")
            continue
        baselines.append({"path": str(path), "sha256": file_fingerprint(path)})
        records = read_scene_records(path)
        for record in records:
            prov = record.get("provenance", {})
            if prov.get("model", {}).get("files_signature") != model["files_signature"]:
                blockers.add("baseline checkpoint signature differs; same weights are not verified")
            saved = prov.get("settings", {})
            expected = {"backend": "vllm-0.28.0", "qwen": settings["qwen"],
                        "visual_evidence": settings["visual_evidence"], "max_new_tokens": 1024,
                        "repetition_penalty": settings["graph_repetition_penalty"]}
            if any(saved.get(key) != value for key, value in expected.items()):
                blockers.add("baseline scene decoding/evidence settings differ")
        try:
            rows = scene_generation_rows(visual, prompt="pilot evidence check", max_new_tokens=1024)
            for row in rows:
                evidence.append({"content_id": cid, "scene_idx": row["scene_idx"],
                                 "keyframes": row["keyframes"],
                                 "images": [{"path": str(p), "sha256": file_fingerprint(Path(p))}
                                            for p in row["image_paths"]]})
        except (OSError, ValueError, RuntimeError) as exc:
            blockers.add(f"unavailable evidence for {cid}: {exc}")
    summary_models = set()
    for cid in [str(row["content_id"]) for row in cohort["catalog"]]:
        path = baseline.summary_arm_dir("graph_qwen") / f"{cid}.json"
        if not path.is_file():
            blockers.add(f"missing baseline summary: {cid}")
            continue
        baselines.append({"path": str(path), "sha256": file_fingerprint(path)})
        document = read_json(path)
        summary_model = summary_model_from_document(document)
        summary_models.add(summary_model)
        provenance = document.get("provenance", {})
        saved = provenance.get("settings", {})
        expected = {"max_new_tokens": 2048}
        if summary_model == "qwen":
            expected.update(backend="vllm-0.28.0", qwen=settings["qwen"],
                            greedy_decoding=settings["greedy_decoding"],
                            repetition_penalty=settings["summary_repetition_penalty"])
            if provenance.get("model", {}).get("files_signature") != model["files_signature"]:
                blockers.add("baseline summary checkpoint signature differs")
            if not settings["greedy_decoding"]:
                expected["sampling"] = settings["summary_sampling"]
        elif summary_model == "gemini":
            expected["backend"] = "gemini"
            if provenance.get("model") != context.config["models"]["gemini"]:
                blockers.add("baseline Gemini summary model/settings differ")
        else:
            blockers.add("unsupported baseline summary model")
        if any(saved.get(key) != value for key, value in expected.items()):
            blockers.add("baseline summary decoding settings differ")
    if len(summary_models) != 1:
        blockers.add("baseline must use one consistent summary model")
    summary_model = next(iter(summary_models)) if len(summary_models) == 1 else None
    manifest = {"baseline_run": baseline.run_id, "pilot_run": context.run_id,
                "summary_model": summary_model,
                "content_ids": [str(row["content_id"]) for row in cohort["catalog"]],
                "model": model, "extraction_settings": settings,
                "prompts": {name: file_fingerprint(context.root / "prompts" / name)
                            for name in ("scene_graph_v5.md", "summary_graph_v6.md")},
                "baseline_files": baselines, "evidence": evidence}
    return pilot, manifest, sorted(blockers)


def compare(pilot, baseline, manifest):
    """Keep missing/raw scenes visible; human quality scores start unreviewed."""
    rows = []
    for cid in manifest["content_ids"]:
        def scenes(context):
            path = context.scene_arm_dir("graph_qwen") / f"{cid}.jsonl"
            return {r["scene_idx"]: r for r in read_scene_records(path)} if path.is_file() else {}
        old, new = scenes(baseline), scenes(pilot)
        evidence = {r["scene_idx"]: r for r in manifest["evidence"] if r["content_id"] == cid}
        for index in sorted(old.keys() | new.keys() | evidence.keys()):
            rows.append({"content_id": cid, "scene_idx": index,
                         "baseline": old.get(index), "candidate": new.get(index),
                         "evidence": evidence.get(index),
                         "review": {key: None for key in (
                             "topic_format", "action_target_tool", "duplicate_identity",
                             "false_interaction", "verbatim_expression", "preserved_features")},
                         "review_notes": ""})
    counts = {"expected_scenes": len(rows)}
    for label in ("baseline", "candidate"):
        present = [r[label] for r in rows if r[label] is not None]
        structured = 0
        for record in present:
            try:
                validate_graph_structure(record.get("graph"))
            except OutputValidationError:
                continue
            structured += 1
        counts[label] = {"present": len(present), "structured": structured,
                         "raw": len(present) - structured, "missing": len(rows) - len(present),
                         "structured_rate": structured / len(rows) if rows else None}
    counts["quality_status"] = "unreviewed; keyframe review and recommendation evaluation required"
    return rows, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run", default="v4_260922")
    parser.add_argument("--model-path", type=Path, help="Relocated identical checkpoint, verified by signature")
    parser.add_argument("--execute", action="store_true", help="Generate only the selected first 100")
    args = parser.parse_args(argv)
    context, baseline = RunContext.load(args.run_id), RunContext.load(args.baseline_run)
    if args.model_path:
        context.config["models"]["qwen"] = str(args.model_path.resolve())
    pilot, manifest, blockers = prepare(context, baseline)
    directory = pilot.run_root / "graph_prompt_pilot"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("pilot inputs changed; use a new run ID")
    else:
        if pilot.run_root.exists():
            raise ValueError("use a fresh run ID to preserve existing outputs")
        write_json(manifest_path, manifest)
    write_json(directory / "preflight.json", {"blockers": blockers})
    if args.execute and not blockers:
        original = steps.qwen_generator

        @contextmanager
        def measured_generator(**kwargs):
            with original(**kwargs) as generate:
                def measured(tasks, on_task_complete=None):
                    def complete(task_id, text):
                        event = kwargs["runtime"].current_result or {}
                        with (directory / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"task_id": task_id, **{k: event.get(k) for k in
                                         ("output_tokens", "prompt_tokens", "finish_reason")}}) + "\n")
                        if on_task_complete:
                            on_task_complete(task_id, text)
                    return generate(tasks, on_task_complete=complete)
                yield measured
        steps.qwen_generator = measured_generator
        try:
            steps.extract_graph_scenes(pilot, model="qwen", arm="graph_qwen",
                                       schema="prompts/scene_graph_v5.md")
            steps.summarize_graph(pilot, model=manifest["summary_model"], arm="graph_qwen",
                                  schema="prompts/summary_graph_v6.md")
        finally:
            steps.qwen_generator = original
    rows, counts = compare(pilot, baseline, manifest)
    write_jsonl(directory / "scene_pairs.jsonl", rows)
    attempts_path = directory / "attempts.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    for stage, selected in (("scene", [a for a in attempts if ":" in a["task_id"]]),
                            ("summary", [a for a in attempts if ":" not in a["task_id"]])):
        tokens = [a["output_tokens"] for a in selected if a["output_tokens"] is not None]
        counts[stage + "_generation"] = {
            "attempts": len(selected), "mean_output_tokens": sum(tokens) / len(tokens) if tokens else None,
            "length_finish_rate": sum(a["finish_reason"] == "length" for a in selected) / len(selected)
            if selected else None}
    write_json(directory / "metrics.json", counts)
    print(json.dumps({"report": str(directory), "blockers": blockers}, ensure_ascii=False, indent=2))
    return 1 if args.execute and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
