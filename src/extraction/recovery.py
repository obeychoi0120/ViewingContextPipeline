"""Per-task generation budgets, durable responses, and replayable artifact commits."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from artifact_io import atomic_write_json
from extraction.structured_output import OutputValidationError


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def penalty_schedule(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(type(x) not in (int, float) or not math.isfinite(x)
                         or not 1 <= x <= 2 for x in values):
        raise ValueError("repetition penalty must be a number or non-empty list in [1, 2]")
    return [float(x) for x in values]


def _save(path, document):
    atomic_write_json(path, document, durable=True)


def has_pending_recovery(directory, task_id):
    path = directory / f"{fingerprint(task_id)}.json"
    if not path.exists():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    return bool(document["cycles"] and document["cycles"][-1]["status"]
                not in {"complete", "raw_fallback"})


def generate_with_recovery(generate, tasks, *, penalties, directory, identity, validate,
                           complete, failed, raw_fallback=None, force=False,
                           runtime=None, attempt_metadata=None, log=lambda message: None):
    """Only validation failures retry. Callbacks and engine errors propagate unchanged."""
    penalties = penalty_schedule(penalties)
    states = {}
    for task in tasks:
        if task.task_id in states:
            raise ValueError(f"duplicate recovery task: {task.task_id}")
        images = [file_fingerprint(path) for path in task.image_paths]
        key = fingerprint({"task": asdict(task), "images": images, "identity": identity,
                           "penalties": penalties, "recovery_version": 1})
        path = directory / f"{fingerprint(task.task_id)}.json"
        document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "schema_version": "generation-recovery/v1", "task_id": task.task_id, "cycles": [],
        }
        if (document.get("schema_version") != "generation-recovery/v1"
                or document.get("task_id") != task.task_id or not isinstance(document.get("cycles"), list)):
            raise ValueError(f"invalid recovery checkpoint: {path}")
        cycles = document["cycles"]
        if (force or not cycles or cycles[-1]["identity"] != key
                or cycles[-1]["status"] == "failed"):
            cycles.append({"cycle_id": str(uuid4()), "identity": key, "status": "running",
                           "attempts": []})
            _save(path, document)
        states[task.task_id] = (task, path, document, cycles[-1])

    done = set()

    def commit(task_id, *, live=False):
        task, path, document, cycle = states[task_id]
        if cycle["status"] in {"prepared", "complete", "raw_fallback"}:
            replay = not live or cycle.get("selected_attempt") != len(cycle["attempts"])
            if runtime and replay:
                runtime.current_result = None
            complete(task_id, cycle["output"])
            if runtime and replay:
                runtime.record_replayed(task_id, cycle["output"], cycle.get("origin"))
            cycle["status"] = cycle["output"].get("status", "complete")
        elif cycle["status"] == "prepared_failure":
            failed(task_id, cycle["attempts"])
            cycle["status"] = "failed"
        else:
            return
        _save(path, document)
        done.add(task_id)

    for task_id in states:
        commit(task_id)

    while len(done) < len(states):
        index = min(len(state[3]["attempts"]) for key, state in states.items() if key not in done)
        if index >= len(penalties):
            raise ValueError("recovery checkpoint exceeds configured generation budget")
        batch = [replace(state[0], repetition_penalty=penalties[index])
                 for key, state in states.items()
                 if key not in done and len(state[3]["attempts"]) == index]
        log(f"[RETRY] attempt={index + 1}/{len(penalties)} "
            f"repetition_penalty={penalties[index]} pending={len(batch)}")
        handled = set()
        batch_ids = {task.task_id for task in batch}

        def handle(task_id, text):
            if task_id not in batch_ids:
                raise RuntimeError(f"unexpected recovery result: {task_id}")
            if task_id in handled:
                return
            handled.add(task_id)
            task, path, document, cycle = states[task_id]
            event = runtime.current_result if runtime and runtime.current_result else {}
            attempt = {"attempt": index + 1, "penalty": penalties[index], "seed": task.seed,
                       "raw_response": text, "error": None, "repair_mode": None,
                       "origin": runtime.checkpoint_origin() if runtime else None,
                       **{key: event.get(key) for key in
                          ("output_tokens", "finish_reason", "stop_reason")}}
            if attempt_metadata is not None:
                attempt.update(attempt_metadata(task_id))
            try:
                output, mode = validate(task_id, text)
                attempt["repair_mode"] = mode
            except OutputValidationError as exc:
                output = None
                attempt["error"] = str(exc)
                attempt["repair_mode"] = "failed"
            cycle["attempts"].append(attempt)
            selected = attempt
            if output is None and index + 1 == len(penalties):
                selected = next((row for row in reversed(cycle["attempts"])
                                 if row["raw_response"].strip()), None)
                if selected is not None and raw_fallback is not None:
                    output = raw_fallback(task_id, selected["raw_response"])
                if output is None:
                    cycle["status"] = "prepared_failure"
            if output is not None:
                cycle.update(status="prepared", output=output,
                             selected_attempt=selected["attempt"], origin=selected["origin"])
            # A response is durable before publishing it; replay never needs another generation.
            _save(path, document)
            commit(task_id, live=True)

        returned = generate(batch, handle)
        for task in batch:
            if task.task_id not in handled:
                if task.task_id not in returned:
                    raise RuntimeError(f"missing generation result: {task.task_id}")
                handle(task.task_id, returned[task.task_id])
