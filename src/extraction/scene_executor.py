from __future__ import annotations

from itertools import chain

from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.structured_output import OutputValidationError, validate_graph_structure
from extraction.generation import generate_penalty_passes
from extraction.step_support import (
    write_progress,
    write_scene_results,
)


def graph_scene_result(row, text, *, error=None, diagnostics=None, strict=True):
    """Convert one response without changing artifacts or the raw graph."""
    parsed = parse_or_repair_graph(text) if error is None else None
    common = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"]}
    parse_warning = parsed.error if parsed is not None else None
    if strict and parsed is not None and parsed.graph is not None:
        try:
            validate_graph_structure(parsed.graph)
        except OutputValidationError as exc:
            parse_warning = str(exc)
    if error is None and parsed is not None:
        structured = parsed.graph is not None and parse_warning is None
        return {
            **common,
            "graph": parsed.graph if structured else text,
            "parse_mode": parsed.parse_mode if structured else "text",
            "semantic_warnings": graph_semantic_warnings(parsed.graph) if structured else [parse_warning],
        }, None
    failure = {
        **common,
        "failure_kind": "generation" if error else "graph_parse",
        "error": error or (parsed.error if parsed is not None else None) or "Graph parsing failed",
        "raw_response": text,
    }
    if diagnostics is not None:
        failure["response_diagnostics"] = diagnostics
    return None, failure


def description_scene_result(row, text, *, content_id, error=None):
    common = {
        "content_id": content_id,
        "scene_idx": row["scene_idx"],
        "keyframes": row["keyframes"],
    }
    description = text.strip()
    if description and error is None:
        return {"schema_version": SCENE_SCHEMA_VERSION, **common, "description": description}, None
    return None, {
        "schema_version": "description-generation-failure/v1",
        **common,
        "failure_kind": "generation" if error else "empty_response",
        "error": error or "model produced an empty description",
    }


class SceneResults:
    def __init__(self, pending, *, scene_dir, failures, records, progress, arm, source, names, provenance=None):
        self.pending = pending
        self.scene_dir = scene_dir
        self.failures = failures
        self.records = records
        self.progress = progress
        self.arm = arm
        self.source = source
        self.names = names
        self.provenance = provenance
        self.penalty = None
        self.rows = {}
        self.contents = {}
        self.completed = set()

    def tasks(self):
        for visual, scene_rows in self.pending:
            cid = str(visual["content_id"])
            self.contents[cid] = {int(row["scene_idx"]): row for row in self.records.get(cid, [])}
            for row in scene_rows:
                task_id = row["task"].task_id
                if task_id in self.rows:
                    raise ValueError(f"duplicate scene task: {task_id}")
                self.rows[task_id] = (cid, row)
                yield row["task"]

    def receive(self, task_id, text, *, error=None, diagnostics=None, truncated=False, final=True):
        if task_id not in self.rows:
            raise RuntimeError(f"unexpected scene result: {task_id}")
        if task_id in self.completed:
            return
        cid, row = self.rows[task_id]
        if self.arm == "graph":
            record, failure = graph_scene_result(
                row, text, error=error or ("graph: output truncated at token limit" if truncated else None),
                diagnostics=diagnostics,
            )
        else:
            record, failure = description_scene_result(
                row, text, content_id=cid,
                error=error or ("description: output truncated at token limit" if truncated else None),
            )
        provenance = self.provenance
        if provenance and self.penalty is not None:
            provenance = {**provenance, "settings": {**provenance["settings"],
                          "actual_repetition_penalty": self.penalty}}
        if record is not None:
            if provenance is not None:
                record["provenance"] = provenance
            self.contents[cid][int(row["scene_idx"])] = record
        else:
            self.contents[cid].pop(int(row["scene_idx"]), None)
        records = list(self.contents[cid].values())
        write_scene_results(self.scene_dir / f"{cid}.jsonl", records)
        self.records[cid] = records
        if failure is not None:
            self.failures.record(cid, int(row["scene_idx"]), failure["error"], "", provenance=provenance)
        else:
            self.failures.remove(cid, int(row["scene_idx"]))
        self.completed.add(task_id)
        self.progress.complete(task_id=task_id, failed=failure is not None,
                               raw=record is not None and record.get("status") == "raw_fallback")
        return failure is not None


def run_qwen_scenes(
    pending, *, scene_dir, failures, model_path, generator_factory,
    names, progress, arm, existing_records, source=None,
    qwen_options=None, image_limit=6, runtime=None, penalties=(1.0,), provenance=None,
):
    results = SceneResults(pending, scene_dir=scene_dir, failures=failures,
                           records=existing_records, progress=progress,
                           arm=arm, source=source, names=names, provenance=provenance)

    def receive(task_id, text, *, final):
        event = runtime.current_result if runtime and runtime.current_result else {}
        results.completed.discard(task_id)
        return results.receive(task_id, text, truncated=event.get("finish_reason") == "length", final=final)

    def begin_pass(index, count, total):
        results.penalty = penalties[index - 1]
        progress.begin_pass(progress.total if total is None else total, index=index, count=count)

    tasks = results.tasks()
    first = next(tasks, None)
    if first is None:
        return
    with generator_factory(
        model_path=model_path, settings=qwen_options, image_limit=image_limit,
        runtime=runtime, on_progress=progress.update_stats,
        log=lambda message: write_progress(progress, message),
    ) as generate:
        generate_penalty_passes(generate, chain((first,), tasks), list(penalties), receive,
                                log=lambda message: write_progress(progress, message),
                                on_pass=begin_pass)


def run_gemini_scenes(
    pending, *, pool, records_by_content, scene_dir, failures, names, progress, arm="graph", provenance=None,
):
    results = SceneResults(pending, scene_dir=scene_dir, failures=failures,
                           records=records_by_content, progress=progress,
                           arm=arm, source="gemini", names=names, provenance=provenance)

    def receive(outcome):
        candidates = (outcome.response_diagnostics or {}).get("candidates") or []
        truncated = any(
            str(getattr(candidate.get("finish_reason"), "value", candidate.get("finish_reason")))
            .rsplit(".", 1)[-1].upper() == "MAX_TOKENS"
            for candidate in candidates
        )
        results.receive(outcome.task_id, outcome.text, error=outcome.error,
                        diagnostics=outcome.response_diagnostics, truncated=truncated)

    tasks = results.tasks()
    first = next(tasks, None)
    if first is not None:
        write_progress(progress, "[Gemini] streaming scenes across contents; "
                       "idle workers immediately take the next scene")
        pool.generate(chain((first,), tasks), receive, on_progress=progress.update_stats)
    missing = results.rows.keys() - results.completed
    if missing:
        raise RuntimeError(f"missing Gemini results: {sorted(missing)}")
