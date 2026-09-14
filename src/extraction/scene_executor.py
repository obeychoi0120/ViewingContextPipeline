from __future__ import annotations


from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.monitoring import graph_skip_message, scene_messages
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.semantic_graph.json_repair import repair_graph_json_once
from extraction.structured_output import OutputValidationError, validate_graph_structure
from extraction.raw_output import raw_graph_record
from extraction.recovery import generate_with_recovery
from pipeline_runtime import write_json
from extraction.step_support import (
    write_progress,
    write_scene_checkpoint,
)


QWEN_GRAPH_SCENE_CONCURRENCY = 8


def graph_scene_result(row, text, *, error=None, diagnostics=None, strict=False):
    """Convert one response without changing artifacts or the raw graph."""
    parsed = parse_or_repair_graph(text) if error is None else None
    common = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"]}
    if strict and parsed is not None and parsed.graph is not None:
        try:
            validate_graph_structure(parsed.graph)
        except OutputValidationError as exc:
            repaired = repair_graph_json_once(text) if parsed.parse_mode == "native" else None
            try:
                validate_graph_structure(repaired)
            except OutputValidationError:
                error = str(exc)
            else:
                from extraction.semantic_graph.json_repair import GraphParseResult
                parsed = GraphParseResult(graph=repaired, parse_mode="repaired")
    if error is None and parsed is not None and parsed.graph is not None:
        return {
            **common,
            "graph": parsed.graph,
            "parse_mode": parsed.parse_mode,
            "semantic_warnings": graph_semantic_warnings(parsed.graph),
        }, None
    failure = {
        **common,
        "failure_kind": "generation" if error else "json_repair",
        "error": error or (parsed.error if parsed is not None else None) or "JSON repair failed",
        "raw_response": text,
    }
    if diagnostics is not None:
        failure["response_diagnostics"] = diagnostics
    return None, failure


def description_scene_result(row, text, *, content_id):
    common = {
        "content_id": content_id,
        "scene_idx": row["scene_idx"],
        "keyframes": row["keyframes"],
    }
    description = text.strip()
    if description:
        return {"schema_version": SCENE_SCHEMA_VERSION, **common, "description": description}, None
    return None, {
        "schema_version": "description-generation-failure/v1",
        **common,
        "failure_kind": "empty_response",
        "error": "model produced an empty description",
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


def run_qwen_scenes(
    pending, *, scene_dir, failure_dir, model_path, gpus, generator_factory,
    names, progress, arm, existing_records, existing_failures, source=None,
    qwen_options=None, image_limit=6, runtime=None, penalties=1.0, force=False,
):
    records_by_content, failures_by_content = {}, {}
    rows_by_task, states = {}, {}
    reused = sum(len(rows) for rows in existing_records.values())
    for visual, scene_rows in pending:
        content_id = str(visual["content_id"])
        states[content_id] = {
            "records": {}, "failures": {}, "handled": set(), "remaining": len(scene_rows),
            "cached": existing_records.get(content_id, []),
            "cached_failures": existing_failures.get(content_id, []),
        }
        for row in scene_rows:
            task_id = row["task"].task_id
            if task_id in rows_by_task:
                raise ValueError(f"duplicate scene task: {task_id}")
            rows_by_task[task_id] = (content_id, row)
    write_progress(progress, f"[Qwen] reused={reused} "
                             f"pending={len(rows_by_task)} contents={len(pending)}")
    for content_id, state in states.items():
        if state["remaining"] == 0:
            write_scene_checkpoint(scene_dir / f"{content_id}.jsonl",
                                   failure_dir / f"{content_id}.jsonl",
                                   state["cached"], state["cached_failures"])
            records_by_content[content_id] = state["cached"]
            failures_by_content[content_id] = state["cached_failures"]
    if not rows_by_task:
        return records_by_content, failures_by_content
    write_progress(progress, (
        "[Qwen] processing contents sequentially with up to 8 concurrent scenes; "
        "each completed scene is checkpointed immediately"
        if arm == "graph" else
        "[Qwen] streaming scenes continuously; each completed scene is checkpointed immediately"
    ))

    def validate(task_id, text):
        content_id, row = rows_by_task[task_id]
        record, failure = (
            graph_scene_result(row, text, strict=True) if arm == "graph"
            else description_scene_result(row, text, content_id=content_id)
        )
        if failure is not None:
            raise OutputValidationError(failure["error"])
        return record, record.get("parse_mode", "native")

    def publish(task_id, record, failure):
        content_id, row = rows_by_task[task_id]
        state = states[content_id]
        if task_id in state["records"] or task_id in state["failures"]:
            return
        if record is not None:
            state["records"][task_id] = record
        else:
            state["failures"][task_id] = failure
            _report_scene(progress, names.get(content_id, f"{content_id}.mp4"),
                          record, failure, arm=arm, source=source)
        state["handled"].add(int(row["scene_idx"]))
        completed = [*state["cached"], *state["records"].values()]
        failed = [
            *[r for r in state["cached_failures"] if int(r["scene_idx"]) not in state["handled"]],
            *state["failures"].values(),
        ]
        write_scene_checkpoint(scene_dir / f"{content_id}.jsonl",
                               failure_dir / f"{content_id}.jsonl", completed, failed)
        state["remaining"] -= 1
        if state["remaining"] == 0:
            records_by_content[content_id] = completed
            failures_by_content[content_id] = failed
        progress.complete(failed=failure is not None)

    def failed(task_id, attempts):
        content_id, row = rows_by_task[task_id]
        last = attempts[-1]
        failure = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"],
                   "failure_kind": "json_repair" if arm == "graph" else "empty_response",
                   "error": last["error"], "raw_response": last["raw_response"]}
        if arm == "description":
            failure.update(schema_version="description-generation-failure/v1", content_id=content_id)
        publish(task_id, None, failure)

    from extraction.summary_executor import qwen_progress
    with generator_factory(
        model_path=model_path, gpus=gpus, settings=qwen_options,
        image_limit=image_limit, runtime=runtime, log=lambda message: write_progress(progress, message),
        on_progress=lambda stats: qwen_progress(progress, stats),
    ) as generate:
        generate_with_recovery(
            generate, [row["task"] for _, row in rows_by_task.values()],
            penalties=penalties, directory=scene_dir / ".recovery",
            identity={"model": str(model_path), "settings": qwen_options, "backend": "vllm-0.28.0"},
            validate=validate, complete=lambda task_id, record: publish(task_id, record, None),
            failed=failed,
            raw_fallback=(lambda task_id, raw: raw_graph_record(rows_by_task[task_id][1], raw))
            if arm == "graph" else None,
            force=force, runtime=runtime,
            log=lambda message: write_progress(progress, message) if arm != "graph" else None,
            rounds_across_batches=arm != "graph",
            batch_size=QWEN_GRAPH_SCENE_CONCURRENCY if arm == "graph" else 256,
            batch_key=(lambda task: rows_by_task[task.task_id][0]) if arm == "graph" else None,
        )
    return records_by_content, failures_by_content


def _complete_gemini_content(
    visual,
    scene_rows,
    generated,
    *,
    records_by_content,
    failures_by_content,
    scene_dir,
    failure_dir,
    force,
):
    content_id = str(visual["content_id"])
    records = list(records_by_content.get(content_id, [])) if not force else []
    new_records = []
    handled_indices = {int(row["scene_idx"]) for row in scene_rows}
    records = [row for row in records if int(row["scene_idx"]) not in handled_indices]
    failures = ([row for row in failures_by_content.get(content_id, [])
                 if int(row["scene_idx"]) not in handled_indices] if not force else [])
    new_failures = []
    for row in scene_rows:
        record, failure = generated[row["task"].task_id]
        (new_records if record is not None else new_failures).append(
            record if record is not None else failure
        )
    records.extend(new_records)
    failures.extend(new_failures)
    write_scene_checkpoint(
        scene_dir / f"{content_id}.jsonl", failure_dir / f"{content_id}.jsonl", records, failures
    )
    records_by_content[content_id] = records
    failures_by_content[content_id] = failures


def run_gemini_scenes(
    pending,
    *,
    pool,
    records_by_content,
    failures_by_content,
    scene_dir,
    failure_dir,
    force,
    names,
    progress,
):
    # This marker contains only content IDs, never generated Scene responses.
    # It distinguishes an interrupted refresh from an older completed output.
    marker = scene_dir / ".pending-contents.json"
    cursor = scene_dir / ".completed-contents.json"
    unfinished = [str(visual["content_id"]) for visual, _ in pending]
    if not unfinished:
        return
    write_json(cursor, {"count": 0})
    write_json(marker, {"content_ids": unfinished})
    for content_index, (visual, scene_rows) in enumerate(pending):
        content_id = str(visual["content_id"])
        rows_by_task = {row["task"].task_id: row for row in scene_rows}
        if len(rows_by_task) != len(scene_rows):
            raise ValueError(f"duplicate scene task in content: {content_id}")
        generated = {}

        def receive(outcome):
            if outcome.task_id not in rows_by_task:
                raise RuntimeError(f"unexpected Gemini result: {outcome.task_id}")
            if outcome.task_id in generated:
                return
            row = rows_by_task[outcome.task_id]
            record, failure = graph_scene_result(
                row, outcome.text, error=outcome.error,
                diagnostics=outcome.response_diagnostics,
            )
            if failure is not None and outcome.error is None and outcome.text.strip():
                record, failure = raw_graph_record(row, outcome.text), None
            generated[outcome.task_id] = (record, failure)
            if failure is not None:
                write_progress(progress, graph_skip_message(
                    names.get(content_id, f"{content_id}.mp4"), failure, source="gemini"))
            progress.complete(failed=failure is not None)

        # The pool bounds concurrency and each worker immediately takes the next
        # Scene from this content's queue, even while other workers are still busy.
        pool.generate([row["task"] for row in scene_rows], receive,
                      on_progress=progress.update_stats)
        missing = rows_by_task.keys() - generated.keys()
        if missing:
            raise RuntimeError(f"missing Gemini results: {sorted(missing)}")
        _complete_gemini_content(
            visual, scene_rows, generated,
            records_by_content=records_by_content, failures_by_content=failures_by_content,
            scene_dir=scene_dir, failure_dir=failure_dir, force=force,
        )
        write_json(cursor, {"count": content_index + 1})
    marker.unlink()
    cursor.unlink()
