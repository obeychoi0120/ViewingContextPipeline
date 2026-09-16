from __future__ import annotations

from itertools import chain

from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.monitoring import graph_skip_message, scene_messages
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.structured_output import OutputValidationError, validate_graph_structure
from extraction.raw_output import raw_graph_record, is_raw_graph
from extraction.generation import generate_once
from extraction.step_support import (
    write_progress,
    write_scene_results,
)


def graph_scene_result(row, text, *, error=None, diagnostics=None, strict=True):
    """Convert one response without changing artifacts or the raw graph."""
    parsed = parse_or_repair_graph(text) if error is None else None
    common = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"]}
    if strict and parsed is not None and parsed.graph is not None:
        try:
            validate_graph_structure(parsed.graph)
        except OutputValidationError as exc:
            error = str(exc)
    if error is None and parsed is not None and parsed.graph is not None:
        return {
            **common,
            "graph": parsed.graph,
            "parse_mode": parsed.parse_mode,
            "semantic_warnings": graph_semantic_warnings(parsed.graph),
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


def _report_scene(progress, name, record, failure, *, arm, source):
    if record is not None:
        write_progress(progress, scene_messages(name, [record], arm=arm, source=source)[0])
    elif arm == "graph":
        write_progress(progress, graph_skip_message(name, failure, source=source))
    else:
        write_progress(
            progress,
            f"[SKIPPED] {name} | description scene "
            f"#{int(failure['scene_idx']):03d} | {failure['error']}",
        )


class SceneResults:
    def __init__(self, pending, *, scene_dir, failures, records, progress, arm, source, names):
        self.pending = pending
        self.scene_dir = scene_dir
        self.failures = failures
        self.records = records
        self.progress = progress
        self.arm = arm
        self.source = source
        self.names = names
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

    def receive(self, task_id, text, *, error=None, diagnostics=None, truncated=False):
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
            # Keep nonempty failed observations usable by the downstream E2E run.
            if failure is not None and error is None and text.strip():
                record = raw_graph_record(row, text)
        else:
            record, failure = description_scene_result(row, text, content_id=cid, error=error)
        if record is not None:
            record["provenance"] = row.get("provenance", {})
            record["generation"] = {"input_key": row.get("input_key")}
            self.contents[cid][int(row["scene_idx"])] = record
        else:
            self.contents[cid].pop(int(row["scene_idx"]), None)
        records = list(self.contents[cid].values())
        write_scene_results(self.scene_dir / f"{cid}.jsonl", records)
        self.records[cid] = records
        if failure is not None:
            self.failures.record(cid, int(row["scene_idx"]), failure["error"], text)
            if record is None:
                _report_scene(self.progress, self.names.get(cid, f"{cid}.mp4"),
                              record, failure, arm=self.arm, source=self.source)
        self.completed.add(task_id)
        self.progress.complete(task_id=task_id, failed=failure is not None,
                               raw=record is not None and is_raw_graph(record))


def run_qwen_scenes(
    pending, *, scene_dir, failures, model_path, generator_factory,
    names, progress, arm, existing_records, source=None,
    qwen_options=None, image_limit=6, runtime=None,
):
    results = SceneResults(pending, scene_dir=scene_dir, failures=failures,
                           records=existing_records, progress=progress,
                           arm=arm, source=source, names=names)

    def receive(task_id, text):
        event = runtime.current_result if runtime and runtime.current_result else {}
        results.receive(task_id, text, truncated=event.get("finish_reason") == "length")

    tasks = results.tasks()
    first = next(tasks, None)
    if first is None:
        return
    with generator_factory(
        model_path=model_path, settings=qwen_options, image_limit=image_limit,
        runtime=runtime, on_progress=progress.update_stats,
        log=lambda message: write_progress(progress, message),
    ) as generate:
        generate_once(generate, chain((first,), tasks), receive)


def run_gemini_scenes(
    pending, *, pool, records_by_content, scene_dir, failures, names, progress, arm="graph",
):
    results = SceneResults(pending, scene_dir=scene_dir, failures=failures,
                           records=records_by_content, progress=progress,
                           arm=arm, source="gemini", names=names)

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
