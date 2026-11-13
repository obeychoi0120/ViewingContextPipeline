from __future__ import annotations

from itertools import chain

from extraction.token_usage import qwen_output_tokens, gemini_output_tokens

from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.graph_warnings import REPEAT_OCCURRED, TOKEN_LIMIT, USABLE_OUTCOMES, outcome
from extraction.semantic_graph import parse_or_repair_graph
from extraction.generation import generate_penalty_passes
from extraction.step_support import (
    write_progress,
    write_scene_results,
)


def graph_scene_result(row, text, *, error=None, truncated=False, diagnostics=None, repeated=False):
    """Convert one response; preserve unresolved or cut graphs as raw text.

    ``repeated`` marks an output stopped before repeating a line: a graph that parses is
    used as usual and only tagged REPEAT_OCCURRED for counting.
    """
    if error is None and not text.strip():
        error = "model produced an empty graph"
    common = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"]}
    if error is None and truncated:
        # Output cut at the token limit is kept as text and never parsed.
        return {**common, "graph": text, "parse_mode": "text", "semantic_warnings": [TOKEN_LIMIT]}, None
    parsed = parse_or_repair_graph(text) if error is None else None
    parse_warning = parsed.error if parsed is not None else None
    warnings = list(parsed.warning) if parsed is not None else []
    if error is None and parsed is not None:
        structured = parsed.graph is not None and parse_warning is None
        if repeated and not warnings:
            warnings = [REPEAT_OCCURRED]
        return {
            **common,
            "graph": parsed.graph if structured else text,
            "parse_mode": parsed.parse_mode if structured else "text",
            "semantic_warnings": warnings,
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


def description_scene_result(row, text, *, content_id, error=None, truncated=False):
    common = {
        "content_id": content_id,
        "scene_idx": row["scene_idx"],
        "keyframes": row["keyframes"],
    }
    description = text.strip()
    if description and error is None:
        return {
            "schema_version": SCENE_SCHEMA_VERSION, **common, "description": description,
            "semantic_warnings": [TOKEN_LIMIT] if truncated else [],
        }, None
    return None, {
        "schema_version": "description-generation-failure/v1",
        **common,
        "failure_kind": "generation" if error else "empty_response",
        "error": error or "model produced an empty description",
    }


class SceneResults:
    def __init__(
        self,
        pending,
        *,
        scene_dir,
        records,
        progress,
        arm,
        names,
        provenance=None,
    ):
        self.pending = pending
        self.scene_dir = scene_dir
        self.records = records
        self.progress = progress
        self.arm = arm
        self.names = names
        self.provenance = provenance
        self.penalty = None
        self.rows = {}
        self.contents = {}
        self.completed = set()
        # REPEAT_OCCURRED records from an earlier pass, kept if a retry does no better.
        self.repeat_fallback = {}

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

    def receive(
        self,
        task_id,
        text,
        *,
        error=None,
        diagnostics=None,
        truncated=False,
        final=True,
        tokens=None,
        repeated=False,
    ):
        if task_id not in self.rows:
            raise RuntimeError(f"unexpected scene result: {task_id}")
        if task_id in self.completed:
            return
        cid, row = self.rows[task_id]
        if self.arm == "graph":
            record, failure = graph_scene_result(
                row, text, error=error, truncated=truncated, diagnostics=diagnostics, repeated=repeated,
            )
        else:
            record, failure = description_scene_result(
                row, text, content_id=cid, error=error, truncated=truncated,
            )
        provenance = self.provenance
        if provenance and self.penalty is not None:
            provenance = {
                **provenance,
                "settings": {**provenance["settings"], "actual_repetition_penalty": self.penalty},
            }
        fallback = self.repeat_fallback.pop(task_id, None)
        if fallback is not None and outcome(record) not in USABLE_OUTCOMES:
            # A retry that is no longer usable does not replace the usable repeated graph.
            record = fallback
        elif record is not None:
            record["tokens"] = tokens
            if provenance is not None:
                record["provenance"] = provenance
            self.contents[cid][int(row["scene_idx"])] = record
        else:
            self.contents[cid][int(row["scene_idx"])] = {
                "content_id": cid, "scene_idx": int(row["scene_idx"]),
                "tokens": tokens, "error": failure["error"], "raw_output": text,
            }
        records = list(self.contents[cid].values())
        write_scene_results(self.scene_dir / f"{cid}.jsonl", records)
        self.records[cid] = records
        self.completed.add(task_id)
        result = outcome(record)
        self.progress.complete(task_id=task_id, outcome=result)
        if result == "repeat_occurred":
            self.repeat_fallback[task_id] = record
        # Everything but success is generated again by a later penalty pass. A retry that
        # succeeds is stored as a plain success, without the earlier warning.
        return result != "success"


def run_qwen_scenes(
    pending,
    *,
    scene_dir,
    model_path,
    generator_factory,
    names,
    progress,
    arm,
    existing_records,
    qwen_options=None,
    image_limit=6,
    runtime=None,
    penalties=(1.0,),
    provenance=None,
):
    results = SceneResults(
        pending,
        scene_dir=scene_dir,
        records=existing_records,
        progress=progress,
        arm=arm,
        names=names,
        provenance=provenance,
    )

    def receive(task_id, text, *, final):
        event = runtime.current_result if runtime and runtime.current_result else {}
        results.completed.discard(task_id)
        return results.receive(
            task_id,
            text,
            truncated=event.get("finish_reason") == "length",
            repeated=bool(event.get("repeat_stopped")),
            final=final,
            tokens=qwen_output_tokens(event),
        )

    def begin_pass(index, count, total):
        results.penalty = penalties[index - 1]
        progress.begin_pass(progress.total if total is None else total, index=index, count=count)

    tasks = results.tasks()
    first = next(tasks, None)
    if first is None:
        return
    with generator_factory(
        model_path=model_path,
        settings=qwen_options,
        image_limit=image_limit,
        runtime=runtime,
        on_progress=progress.update_stats,
        log=lambda message: write_progress(progress, message),
    ) as generate:
        generate_penalty_passes(
            generate,
            chain((first,), tasks),
            list(penalties),
            receive,
            log=lambda message: write_progress(progress, message),
            on_pass=begin_pass,
        )


def run_gemini_scenes(
    pending,
    *,
    pool,
    records_by_content,
    scene_dir,
    names,
    progress,
    arm="graph",
    provenance=None,
):
    results = SceneResults(
        pending,
        scene_dir=scene_dir,
        records=records_by_content,
        progress=progress,
        arm=arm,
        names=names,
        provenance=provenance,
    )

    def receive(outcome):
        candidates = (outcome.response_diagnostics or {}).get("candidates") or []
        truncated = any(
            str(getattr(candidate.get("finish_reason"), "value", candidate.get("finish_reason")))
            .rsplit(".", 1)[-1]
            .upper()
            == "MAX_TOKENS"
            for candidate in candidates
        )
        results.receive(
            outcome.task_id,
            outcome.text,
            error=outcome.error,
            diagnostics=outcome.response_diagnostics,
            truncated=truncated,
            tokens=gemini_output_tokens(outcome.response_diagnostics),
        )

    tasks = results.tasks()
    first = next(tasks, None)
    if first is not None:
        write_progress(
            progress,
            "[Gemini] streaming scenes across contents; "
            "idle workers immediately take the next scene",
        )
        pool.generate(chain((first,), tasks), receive, on_progress=progress.update_stats)
    missing = results.rows.keys() - results.completed
    if missing:
        raise RuntimeError(f"missing Gemini results: {sorted(missing)}")
