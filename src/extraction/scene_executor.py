from __future__ import annotations

from itertools import chain

from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.monitoring import graph_skip_message, scene_messages
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.semantic_graph.json_repair import repair_graph_json_once
from extraction.structured_output import OutputValidationError, validate_graph_structure
from extraction.raw_output import raw_graph_record
from extraction.recovery import generate_with_recovery, start_force_run
from pipeline_runtime import read_json, write_json
from extraction.step_support import (
    write_progress,
    write_scene_checkpoint,
)


def graph_scene_result(row, text, *, error=None, diagnostics=None, strict=True):
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
    pending, *, scene_dir, failure_dir, model_path, generator_factory,
    names, progress, arm, existing_records, existing_failures, source=None,
    qwen_options=None, image_limit=6, runtime=None, penalties=1.0, force=False, identity=None,
):
    records_by_content, failures_by_content = {}, {}
    rows_by_task, states = {}, {}
    if force:
        start_force_run(scene_dir / ".recovery")

    def register_content(visual, scene_rows):
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
        return [row["task"] for row in scene_rows]

    def stream_tasks():
        for visual, scene_rows in pending:
            yield from register_content(visual, scene_rows)

    write_progress(progress, "[Qwen] streaming scenes across contents at one repetition penalty; "
                   "retry failed scenes after the full pass; checkpoint each completed scene")

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
            record["provenance"] = row.get("provenance", {})
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
        model_path=model_path, settings=qwen_options,
        image_limit=image_limit, runtime=runtime, log=lambda message: write_progress(progress, message),
        on_progress=lambda stats: qwen_progress(progress, stats),
    ) as generate:
        generate_with_recovery(
            generate, stream_tasks(),
            penalties=penalties, directory=scene_dir / ".recovery",
            identity=identity or {
                "model": str(model_path), "settings": qwen_options, "backend": "vllm-0.28.0",
            },
            validate=validate, complete=lambda task_id, record: publish(task_id, record, None),
            failed=failed,
            raw_fallback=(lambda task_id, raw: raw_graph_record(rows_by_task[task_id][1], raw))
            if arm == "graph" else None,
            runtime=runtime,
            log=lambda message: write_progress(progress, message),
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
    content_ids,
    arm="graph",
    resume_force=False,
):
    # This marker contains only content IDs, never generated Scene responses.
    # It distinguishes an interrupted refresh from an older completed output.
    marker = scene_dir / ".pending-contents.json"
    cursor = scene_dir / ".completed-contents.json"
    if not content_ids:
        return
    positions = {str(cid): index for index, cid in enumerate(content_ids)}
    states = {}
    rows_by_task = {}
    completed_positions = set()
    frontier = 0
    discovered_position = -1
    if resume_force and not force and cursor.is_file():
        previous = read_json(cursor)
        frontier = previous["count"]
        completed_positions.update(positions[cid]
                                   for cid in previous.get("completed_content_ids", []))

    def checkpoint_cursor():
        nonlocal frontier
        while frontier in completed_positions:
            completed_positions.remove(frontier)
            frontier += 1
        write_json(cursor, {
            "count": frontier,
            "completed_content_ids": [content_ids[i] for i in sorted(completed_positions)],
            "active_content_ids": list(states),
        })

    checkpoint_cursor()
    write_json(marker, {"content_ids": content_ids, "force": force or resume_force})

    def stream_tasks():
        nonlocal discovered_position
        for visual, scene_rows in pending:
            content_id = str(visual["content_id"])
            content_index = positions[content_id]
            # Contents omitted by pending are already cached, including on resume.
            completed_positions.update(range(max(frontier, discovered_position + 1), content_index))
            discovered_position = content_index
            states[content_id] = (visual, scene_rows, {})
            for row in scene_rows:
                task_id = row["task"].task_id
                if task_id in rows_by_task:
                    raise ValueError(f"duplicate scene task: {task_id}")
                rows_by_task[task_id] = (content_id, row)
            checkpoint_cursor()
            yield from (row["task"] for row in scene_rows)

    def receive(outcome):
        if outcome.task_id not in rows_by_task:
            raise RuntimeError(f"unexpected Gemini result: {outcome.task_id}")
        content_id, row = rows_by_task[outcome.task_id]
        visual, scene_rows, generated = states[content_id]
        if outcome.task_id in generated:
            return
        if arm == "graph" or outcome.error:
            record, failure = graph_scene_result(
                row, outcome.text, error=outcome.error, diagnostics=outcome.response_diagnostics,
            )
            if failure is not None and outcome.error is None and outcome.text.strip():
                record, failure = raw_graph_record(row, outcome.text), None
        else:
            record, failure = description_scene_result(row, outcome.text, content_id=content_id)
        if record is not None:
            record["provenance"] = row.get("provenance", {})
            record["generation"] = {"input_key": row.get("input_key"), "attempt_count": 1,
                                    "repair_mode": record.get("parse_mode", "native")}
        generated[outcome.task_id] = (record, failure)
        if failure is not None:
            write_progress(progress, graph_skip_message(
                names.get(content_id, f"{content_id}.mp4"), failure, source="gemini"))
        if len(generated) == len(scene_rows):
            _complete_gemini_content(
                visual, scene_rows, generated,
                records_by_content=records_by_content, failures_by_content=failures_by_content,
                scene_dir=scene_dir, failure_dir=failure_dir, force=force,
            )
            for completed_row in scene_rows:
                del rows_by_task[completed_row["task"].task_id]
            del states[content_id]
            completed_positions.add(positions[content_id])
            checkpoint_cursor()
        progress.complete(failed=failure is not None)

    tasks = stream_tasks()
    first = next(tasks, None)
    if first is not None:
        write_progress(progress, "[Gemini] streaming scenes across contents; "
                       "idle workers immediately take the next scene")
        pool.generate(chain((first,), tasks), receive, on_progress=progress.update_stats)
    if rows_by_task:
        raise RuntimeError(f"missing Gemini results: {sorted(rows_by_task)}")
    marker.unlink()
    cursor.unlink()
