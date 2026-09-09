from __future__ import annotations


from extraction.backends import GeminiGenerationOutcome
from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.monitoring import graph_skip_message, scene_messages
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.step_support import (
    complete_content_progress,
    write_progress,
    write_scene_checkpoint,
)


def graph_scene_result(row, text, *, error=None, diagnostics=None):
    """Convert one response without changing artifacts or the raw graph."""
    parsed = parse_or_repair_graph(text) if error is None else None
    common = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"]}
    if parsed is not None and parsed.graph is not None:
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
    qwen_options=None, image_limit=6, runtime=None,
):
    records_by_content, failures_by_content = {}, {}
    rows_by_task, states = {}, {}
    reused = sum(len(rows) for rows in existing_records.values())
    unknown = sum(
        runtime.classification(f"{content_id}:{row['scene_idx']}", row) == "legacy_unknown"
        for content_id, rows in existing_records.items() for row in rows
    ) if runtime else reused
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
    write_progress(progress, f"[Qwen] reused={reused} legacy_unknown={unknown} "
                             f"pending={len(rows_by_task)} contents={len(pending)}")
    for content_id, state in states.items():
        if state["remaining"] == 0:
            write_scene_checkpoint(scene_dir / f"{content_id}.jsonl",
                                   failure_dir / f"{content_id}.jsonl",
                                   state["cached"], state["cached_failures"])
            records_by_content[content_id] = state["cached"]
            failures_by_content[content_id] = state["cached_failures"]
            complete_content_progress(progress)
    if not rows_by_task:
        return records_by_content, failures_by_content
    write_progress(progress, "[Qwen] starting vLLM GPU workers; each completed scene is checkpointed immediately")

    def complete(task_id, text):
        content_id, row = rows_by_task[task_id]
        state = states[content_id]
        if task_id in state["records"] or task_id in state["failures"]:
            return
        record, failure = (
            graph_scene_result(row, text) if arm == "graph"
            else description_scene_result(row, text, content_id=content_id)
        )
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
        if runtime:
            runtime.record(task_id, record if record is not None else failure,
                           status="complete" if record is not None else "failed",
                           artifact_id=f"{content_id}:{row['scene_idx']}")
        state["remaining"] -= 1
        if state["remaining"] == 0:
            records_by_content[content_id] = completed
            failures_by_content[content_id] = failed
            progress.update(1)

    from extraction.summary_executor import qwen_progress
    with generator_factory(
        model_path=model_path, gpus=gpus, settings=qwen_options,
        image_limit=image_limit, runtime=runtime, log=lambda message: write_progress(progress, message),
        on_progress=lambda stats: qwen_progress(progress, stats),
    ) as generate:
        # All videos share one admission stream; results remain independently keyed.
        returned = generate([row["task"] for _, row in rows_by_task.values()], complete)
        for task_id, text in returned.items():
            complete(task_id, text)
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
    names,
    progress,
):
    content_id = str(visual["content_id"])
    records = list(records_by_content.get(content_id, [])) if not force else []
    new_records = []
    failures = []
    for row in scene_rows:
        outcome = generated[row["task"].task_id]
        record, failure = graph_scene_result(
            row,
            outcome.text,
            error=outcome.error,
            diagnostics=outcome.response_diagnostics,
        )
        (new_records if record is not None else failures).append(
            record if record is not None else failure
        )
    records.extend(new_records)
    write_scene_checkpoint(
        scene_dir / f"{content_id}.jsonl", failure_dir / f"{content_id}.jsonl", records, failures
    )
    records_by_content[content_id] = records
    failures_by_content[content_id] = failures
    name = names.get(content_id, f"{content_id}.mp4")
    for message in scene_messages(name, new_records, arm="graph", source="gemini"):
        write_progress(progress, message)
    for failure in failures:
        write_progress(progress, graph_skip_message(name, failure, source="gemini"))
    complete_content_progress(progress)


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
    task_context = {}
    generated_by_content: dict[str, dict[str, GeminiGenerationOutcome]] = {}
    tasks = []
    for visual, scene_rows in pending:
        content_id = str(visual["content_id"])
        generated_by_content[content_id] = {}
        for row in scene_rows:
            task = row["task"]
            tasks.append(task)
            task_context[task.task_id] = (content_id, visual, scene_rows)

    def complete(outcome):
        content_id, visual, scene_rows = task_context[outcome.task_id]
        responses = generated_by_content[content_id]
        responses[outcome.task_id] = outcome
        if len(responses) == len(scene_rows):
            _complete_gemini_content(
                visual,
                scene_rows,
                responses,
                records_by_content=records_by_content,
                failures_by_content=failures_by_content,
                scene_dir=scene_dir,
                failure_dir=failure_dir,
                force=force,
                names=names,
                progress=progress,
            )

    pool.generate(tasks, complete)
