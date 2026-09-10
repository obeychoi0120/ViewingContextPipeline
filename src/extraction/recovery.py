"""Per-task generation budgets, durable responses, and replayable artifact commits."""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import chain
import hashlib
import json
import math
import shutil
from pathlib import Path
from uuid import uuid4

from artifact_io import atomic_write_json
from extraction.structured_output import OutputValidationError


def clear_recovery(output_dir):
    """Remove a successful step's recovery journal within its output directory."""
    output_dir = Path(output_dir).resolve()
    directory = output_dir / ".recovery"
    if directory.exists():
        if directory.resolve().parent != output_dir:
            raise ValueError(f"recovery directory escapes step output: {directory}")
        shutil.rmtree(directory)


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


def active_force_run(directory):
    path = directory / ".force-run"
    return json.loads(path.read_text(encoding="utf-8"))["force_run_id"] if path.exists() else None


def has_pending_recovery(directory, task_id, force_run_id=None):
    path = directory / f"{fingerprint(task_id)}.json"
    if not path.exists():
        return force_run_id is not None
    document = json.loads(path.read_text(encoding="utf-8"))
    if force_run_id and (not document["cycles"]
                         or document["cycles"][-1].get("force_run_id") != force_run_id):
        return True
    return bool(document["cycles"] and document["cycles"][-1]["status"]
                not in {"complete", "raw_fallback"})


def generate_with_recovery(generate, tasks, *, penalties, directory, identity, validate,
                           complete, failed, raw_fallback=None, force=False,
                           runtime=None, attempt_metadata=None, log=lambda message: None,
                           batch_size=256, rounds_across_batches=False):
    """Bound input hashing/checkpoint IO before inference; keep one caller-owned model pool."""
    tasks = list(tasks)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate recovery task")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("recovery batch size must be a positive integer")
    penalties = penalty_schedule(penalties)
    force_run_id = active_force_run(directory)
    if force:
        force_run_id = str(uuid4())
        # One intent marker covers tasks in later batches, including existing good outputs.
        _save(directory / ".force-run", {"force_run_id": force_run_id})
    for attempt_index in (range(len(penalties)) if rounds_across_batches else [None]):
        remaining = []
        if rounds_across_batches:
            log(f"[RECOVERY] scene pass={attempt_index + 1}/{len(penalties)} "
                f"repetition_penalty={penalties[attempt_index]}")
        submission_size = (len(tasks) or 1) if rounds_across_batches else batch_size
        for offset in range(0, len(tasks), submission_size):
            batch = tasks[offset:offset + submission_size]
            if not rounds_across_batches:
                log(f"[RECOVERY] preparing tasks {offset + 1}-{offset + len(batch)}/{len(tasks)} "
                    "(input hashes and checkpoints)")
            remaining.extend(_generate_recovery_batch(
                generate, batch, penalties=penalties, directory=directory, identity=identity,
                validate=validate, complete=complete, failed=failed, raw_fallback=raw_fallback,
                force=force and attempt_index in (None, 0), force_run_id=force_run_id,
                runtime=runtime, attempt_metadata=attempt_metadata, log=log,
                attempt_index=attempt_index,
            ))
        tasks = remaining
        if not tasks:
            break
    if force_run_id:
        (directory / ".force-run").unlink(missing_ok=True)


def _generate_recovery_batch(generate, tasks, *, penalties, directory, identity, validate,
                             complete, failed, raw_fallback, force, force_run_id,
                             runtime, attempt_metadata, log, attempt_index):
    """Only validation failures retry. Callbacks and engine errors propagate unchanged."""
    penalties = penalty_schedule(penalties)
    states = {}
    streaming = attempt_index is not None
    remaining = []

    def prepare(task):
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
                or force_run_id and cycles[-1].get("force_run_id") != force_run_id
                or cycles[-1]["status"] == "failed"):
            cycles.append({"cycle_id": str(uuid4()), "identity": key, "status": "running",
                           "attempts": []})
            if force_run_id:
                cycles[-1]["force_run_id"] = force_run_id
        states[task.task_id] = (task, path, document, cycles[-1])

    if not streaming:
        for task in tasks:
            prepare(task)

    done = set()

    def commit(task_id, *, live=False):
        task, path, document, cycle = states[task_id]
        previous_status = cycle["status"]
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
        if cycle["status"] != previous_status:
            _save(path, document)
        done.add(task_id)

    for task_id in states:
        commit(task_id)

    while streaming or len(done) < len(states):
        index = attempt_index if streaming else min(
            len(state[3]["attempts"]) for key, state in states.items() if key not in done)
        if index >= len(penalties):
            raise ValueError("recovery checkpoint exceeds configured generation budget")
        handled = set()
        batch_ids = set()

        def admit():
            for task in tasks:
                prepare(task)
                commit(task.task_id)
                count = len(states[task.task_id][3]["attempts"])
                if task.task_id in done:
                    del states[task.task_id]
                elif count == index:
                    batch_ids.add(task.task_id)
                    yield replace(task, repetition_penalty=penalties[index])
                else:
                    if count >= len(penalties) or count < index:
                        raise ValueError("recovery checkpoint has an invalid generation budget")
                    remaining.append(task)
                    del states[task.task_id]

        if streaming:
            iterator = admit()
            first = next(iterator, None)
            if first is None:
                break
            batch = chain((first,), iterator)
        else:
            batch = [replace(state[0], repetition_penalty=penalties[index])
                     for key, state in states.items()
                     if key not in done and len(state[3]["attempts"]) == index]
            batch_ids = {task.task_id for task in batch}
            log(f"[RETRY] attempt={index + 1}/{len(penalties)} "
                f"repetition_penalty={penalties[index]} pending={len(batch)}")

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
            if streaming:
                if task_id not in done:
                    remaining.append(task)
                del states[task_id]

        returned = generate(batch, handle)
        for task_id in batch_ids - handled:
            if task_id not in returned:
                raise RuntimeError(f"missing generation result: {task_id}")
            handle(task_id, returned[task_id])
        if streaming:
            break
    if streaming:
        return remaining
    return [state[0] for key, state in states.items() if key not in done]
