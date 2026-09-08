from __future__ import annotations


from extraction.backends import GeminiGenerationOutcome
from extraction.descriptions import SCENE_SCHEMA_VERSION
from extraction.monitoring import graph_skip_message, scene_messages
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.step_support import (
    complete_content_progress, write_progress, write_scene_checkpoint,
)


def graph_scene_result(row, text, *, error=None, diagnostics=None):
    """Convert one response without changing artifacts or the raw graph."""
    parsed = parse_or_repair_graph(text) if error is None else None
    common = {"scene_idx": row["scene_idx"], "keyframes": row["keyframes"]}
    if parsed is not None and parsed.graph is not None:
        return {**common, "graph": parsed.graph, "parse_mode": parsed.parse_mode,
                "semantic_warnings": graph_semantic_warnings(parsed.graph)}, None
    failure = {
        **common, "failure_kind": "generation" if error else "json_repair",
        "error": error or (parsed.error if parsed is not None else None) or "JSON repair failed",
        "raw_response": text,
    }
    if diagnostics is not None:
        failure["response_diagnostics"] = diagnostics
    return None, failure


def description_scene_result(row, text, *, content_id):
    common = {"content_id": content_id, "scene_idx": row["scene_idx"],
              "keyframes": row["keyframes"]}
    description = text.strip()
    if description:
        return {"schema_version": SCENE_SCHEMA_VERSION, **common,
                "description": description}, None
    return None, {"schema_version": "description-generation-failure/v1", **common,
                  "failure_kind": "empty_response", "error": "model produced an empty description"}


def _report_scene(progress, name, record, failure, *, arm, source):
    if record is not None:
        write_progress(progress, scene_messages(name, [record], arm=arm, source=source)[0])
    elif arm == "graph":
        write_progress(progress, graph_skip_message(name, failure, source=source))
    else:
        write_progress(progress, f"[SKIPPED] {name} | description scene "
                       f"#{int(failure['scene_idx']):03d} | {failure['error']}")


def run_qwen_scenes(
    pending, *, scene_dir, failure_dir, model_path, gpus, generator_factory,
    names, progress, arm, source=None,
):
    records_by_content = {}
    failures_by_content = {}
    write_progress(progress,
        "[Qwen] starting GPU workers; each completed scene is checkpointed immediately")
    with generator_factory(model_path=model_path, gpus=gpus) as generate:
        for visual, scene_rows in pending:
            content_id = str(visual["content_id"])
            name = names.get(content_id, f"{content_id}.mp4")
            path = scene_dir / f"{content_id}.jsonl"
            failure_path = failure_dir / f"{content_id}.jsonl"
            rows_by_task = {row["task"].task_id: row for row in scene_rows}
            records = {}
            failures = {}
            label = "graph" if arm == "graph" else "desc"
            write_progress(progress, f"[Qwen_{label}] {name} | submitted {len(scene_rows)} scenes")

            def complete(task_id, text):
                row = rows_by_task[task_id]
                record, failure = (
                    graph_scene_result(row, text) if arm == "graph" else
                    description_scene_result(row, text, content_id=visual["content_id"])
                )
                if record is not None:
                    records[task_id] = record
                else:
                    failures[task_id] = failure
                _report_scene(progress, name, record, failure, arm=arm, source=source)
                completed = list(records.values())
                failed = list(failures.values())
                write_scene_checkpoint(path, failure_path, completed, failed)
                if len(records) + len(failures) == len(scene_rows):
                    records_by_content[content_id] = completed
                    failures_by_content[content_id] = failed
                    complete_content_progress(progress)

            returned = generate([row["task"] for row in scene_rows], complete)
            for task_id, text in returned.items():
                if task_id not in records and task_id not in failures:
                    complete(task_id, text)
            if not scene_rows:
                write_scene_checkpoint(path, failure_path, [], [])
                records_by_content[content_id] = []
                failures_by_content[content_id] = []
                if arm == "description":
                    write_progress(progress, f"[SKIPPED] {name} | no scenes to extract")
                complete_content_progress(progress)
    return records_by_content, failures_by_content


def _complete_gemini_content(
    visual, scene_rows, generated, *, records_by_content, failures_by_content,
    scene_dir, failure_dir, force, names, progress,
):
    content_id = str(visual["content_id"])
    records = list(records_by_content.get(content_id, [])) if not force else []
    new_records = []
    failures = []
    for row in scene_rows:
        outcome = generated[row["task"].task_id]
        record, failure = graph_scene_result(
            row, outcome.text, error=outcome.error, diagnostics=outcome.response_diagnostics,
        )
        (new_records if record is not None else failures).append(
            record if record is not None else failure)
    records.extend(new_records)
    write_scene_checkpoint(scene_dir / f"{content_id}.jsonl",
                           failure_dir / f"{content_id}.jsonl", records, failures)
    records_by_content[content_id] = records
    failures_by_content[content_id] = failures
    name = names.get(content_id, f"{content_id}.mp4")
    for message in scene_messages(name, new_records, arm="graph", source="gemini"):
        write_progress(progress, message)
    for failure in failures:
        write_progress(progress, graph_skip_message(name, failure, source="gemini"))
    complete_content_progress(progress)


def run_gemini_scenes(
    pending, *, pool, records_by_content, failures_by_content, scene_dir, failure_dir,
    force, names, progress,
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
                visual, scene_rows, responses, records_by_content=records_by_content,
                failures_by_content=failures_by_content, scene_dir=scene_dir,
                failure_dir=failure_dir, force=force, names=names, progress=progress,
            )
    pool.generate(tasks, complete)
