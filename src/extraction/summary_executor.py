"""Shared summary storage with Qwen and Gemini generation policies."""

from __future__ import annotations
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from arm_registry import Arm
from typing import Callable, Iterator
from tqdm import tqdm
from extraction.progress import InferenceProgress
from extraction.qwen_runtime import QwenRuntime
from extraction.token_usage import qwen_output_tokens, gemini_output_tokens, validate_tokens
from artifact_io import atomic_write_json
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.descriptions import description_summary_prompt
from extraction.errors import ExtractionStepError
from artifact_io import fingerprint
from extraction.qwen_config import penalty_schedule
from model_provenance import canonical, without_provenance_arm
from extraction.failures import FailureLog
from extraction.generation import generate_once
from extraction.scene_storage import read_scene_records
from extraction.semantic_graph import graph_summary_prompt
from extraction.step_support import minimal_description_records, minimal_graph_records, result
from extraction.summary_storage import (
    SUMMARY_SCHEMA_VERSION,
    inspect_summary,
    reuse_summary_document,
    check_summary_model,
    validate_summary_template,
)
from artifact_io import read_json, read_jsonl

GenerationFunction = Callable


@dataclass
class SummaryBatch:
    """Pending attempts and published results for one source/model execution."""

    arm: Arm
    output_dir: Path
    failures: FailureLog
    pending: dict = field(default_factory=dict)
    documents: dict = field(default_factory=dict)
    next_penalties: dict = field(default_factory=dict)

    def publish(self, cid, text, violations, *, raw=False, tokens=None):
        records, prov, _ = self.pending[cid]
        if raw:
            text = ""
        doc = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "content_id": cid,
            "tokens": validate_tokens(tokens),
            "arm": self.arm.name,
            "status": "failed" if raw else "complete",
            "text": text,
            "scene_count": len(records),
            "word_count": len(text.split()),
            "violations": violations,
            "correction_count": 0,
            "provenance": without_provenance_arm(prov),
        }
        output = self.output_dir / f"{cid}.json"
        try:
            unchanged = output.is_file() and read_json(output) == doc
        except (ValueError, OSError):
            unchanged = False
        if not unchanged:
            atomic_write_json(output, doc, durable=True, sort_keys=False)
        self.documents[cid] = doc


def summary_scene_input(source, normalize, graph, provenance):
    try:
        all_records = normalize(read_scene_records(source), source) if source.is_file() else []
        scene_failure_path = source.parent / "failures" / source.name
        scene_failures = read_jsonl(scene_failure_path) if scene_failure_path.is_file() else []
        failed_indices = {r["scene_idx"] for r in scene_failures}
        records = [
            r
            for r in all_records
            if r.get("status") not in {"failed", "raw_fallback"}
            and r["scene_idx"] not in failed_indices
        ]
    except (ValueError, KeyError, TypeError) as exc:
        raise ExtractionStepError(f"invalid scene input {source}: {exc}") from exc
    text_count = sum((r.get("parse_mode") == "text" for r in records)) if graph else 0
    raw_count = len(all_records) - len(records) + text_count
    scene_provenance = [without_provenance_arm(r.get("provenance", {})) for r in records]
    if not records:
        scene_provenance = [without_provenance_arm(r.get("provenance", {})) for r in scene_failures]
    prov = {
        **provenance,
        "scene_provenance": scene_provenance,
        "scene_input_hash": fingerprint(
            canonical([{k: v for k, v in record.items() if k != "tokens"} for record in records])
        ),
        "scene_path": str(source),
        "normal_scene_count": len(records) - text_count,
        "raw_scene_count": raw_count,
        **({"text_scene_count": text_count} if graph else {}),
    }
    prov["input_hash"] = fingerprint(prov)
    return records, prov


def prepare_summary_batch(context, *, arm, schema, catalog, provenance, generation, model, force):
    output_dir = context.summary_arm_dir(arm.name)
    scene_dir = context.scene_arm_dir(arm.scene_arm)
    if not force:
        check_summary_model(output_dir, model)
    failures = FailureLog(output_dir)
    if force:
        for path in output_dir.glob("*.json"):
            path.unlink()
        failures.clear_contents(list(failures.by_content))
    template = schema.read_text(encoding="utf-8")
    validate_summary_template(template, arm.uses_title)
    graph = arm.representation == "graph"
    normalize = minimal_graph_records if graph else minimal_description_records
    build_prompt = graph_summary_prompt if graph else description_summary_prompt
    settings = context.config["extraction"]
    max_tokens = settings["graph" if graph else "description"][model]["summary_max_new_tokens"]
    schedule = (
        penalty_schedule(settings["summary_repetition_penalty"]) if model == "qwen" else [1.0]
    )
    if any((left >= right for left, right in zip(schedule, schedule[1:]))):
        raise ExtractionStepError("summary repetition penalties must be strictly increasing")
    batch = SummaryBatch(arm, output_dir, failures)
    pending, documents, next_penalties = batch.pending, batch.documents, batch.next_penalties
    publish = batch.publish

    for item in catalog:
        cid = str(item["content_id"])
        source = scene_dir / f"{cid}.jsonl"
        output = output_dir / f"{cid}.json"
        if output.is_file() and (not force) and (not failures.contains(cid, None)):
            try:
                existing = reuse_summary_document(output, content_id=cid, arm=arm.name)
            except ExtractionStepError:
                pass
            else:
                prov = existing["provenance"]
                if (
                    prov.get("uses_title") is not False
                    or "english_title" in prov
                    or prov.get("scene_arm") != arm.scene_arm
                ):
                    raise ExtractionStepError(
                        f"Summary source/title policy mismatch: {output}; use --force or a new run"
                    )
                if existing["status"] == "complete":
                    documents[cid] = existing
                    continue
        records, prov = summary_scene_input(source, normalize, graph, provenance)
        if not records:
            pending[cid] = (records, prov, None)
            failures.record(
                cid, None, "no successful scenes", "", summary_model=model, provenance=prov
            )
            publish(cid, "", ["no successful scenes"], raw=True)
            continue
        task = QwenGenerationTask(
            task_id=cid,
            image_paths=(),
            prompt=build_prompt(template, records, english_title=None),
            max_new_tokens=max_tokens,
            **generation,
        )
        pending[cid] = (records, prov, task)
        failed = failures.rows.get((cid, None))
        input_changed = (
            failed
            and failed.get("provenance", {}).get("scene_input_hash")
            and (failed["provenance"]["scene_input_hash"] != prov["scene_input_hash"])
        )
        if failed is None or model == "gemini" or input_changed:
            next_penalties[cid] = schedule[0]
        else:
            last = failed.get("repetition_penalty")
            if last is not None and (type(last) not in (int, float) or not 1 <= last <= 2):
                raise ExtractionStepError(f"invalid summary repetition_penalty for {cid}: {last}")
            next_penalty = next(
                (value for value in schedule if last is not None and value > last), None
            )
            if next_penalty is None:
                pending[cid] = (
                    records,
                    failed.get(
                        "provenance", {"representation": arm.representation, "summary_model": model}
                    ),
                    task,
                )
                publish(
                    cid,
                    failed["raw_output"],
                    failed["error"].split(", "),
                    raw=True,
                    tokens=failed.get("tokens"),
                )
            else:
                next_penalties[cid] = next_penalty
    return batch, schedule


def run_qwen_summaries(context, batch, schedule, progress, generator_factory):
    pending, next_penalties = batch.pending, batch.next_penalties
    failures, publish = batch.failures, batch.publish
    settings = context.config["extraction"]
    output_dir, model = batch.output_dir, "qwen"
    runtime = QwenRuntime()

    def receive(cid, text):
        normalized, violations = inspect_summary(text)
        records, prov, task = pending[cid]
        prov = {**prov, "settings": {**prov["settings"], "actual_repetition_penalty": penalty}}
        prov["input_hash"] = fingerprint({k: v for k, v in prov.items() if k != "input_hash"})
        pending[cid] = (records, prov, task)
        event = runtime.current_result or {}
        tokens = qwen_output_tokens(event)
        if event.get("finish_reason") == "length":
            violations.append("max_tokens")
        final = penalty == schedule[-1]
        if violations:
            failures.record(
                cid,
                None,
                ", ".join(violations),
                "",
                repetition_penalty=penalty,
                summary_model=model,
                provenance=pending[cid][1],
                tokens=tokens,
            )
            if final:
                publish(cid, text, violations, raw=True, tokens=tokens)
                del next_penalties[cid]
            else:
                (output_dir / f"{cid}.json").unlink(missing_ok=True)
                next_penalties[cid] = schedule[schedule.index(penalty) + 1]
        else:
            publish(cid, normalized, [], tokens=tokens)
            failures.remove(cid, None)
            del next_penalties[cid]
        progress.complete(task_id=cid, failed=bool(violations), raw=False)

    with generator_factory(
        model_path=context.path("models", "qwen"),
        settings=settings.get("qwen"),
        image_limit=settings["visual_evidence"]["num_keyframes"],
        runtime=runtime,
        on_progress=progress.update_stats,
        log=progress.write_log,
    ) as generate:
        for index, penalty in enumerate(schedule, start=1):
            tasks = [
                replace(pending[cid][2], repetition_penalty=penalty)
                for cid, value in next_penalties.items()
                if value == penalty
            ]
            if tasks:
                progress.begin_pass(len(tasks), index=index, count=len(schedule))
                progress.write_log(
                    f"[Qwen] repetition_penalty={penalty:.2f} pass={index}/{len(schedule)} tasks={len(tasks)}"
                )
                generate_once(generate, tasks, receive)


def run_gemini_summaries(context, batch, progress, gemini_pool_factory):
    pending, next_penalties = batch.pending, batch.next_penalties
    failures, publish = batch.failures, batch.publish
    settings = context.config["extraction"]
    model = "gemini"
    pool = gemini_pool_factory(settings["gemini"]["threads"], **context.config["models"]["gemini"])
    expected = set(next_penalties)
    completed = set()

    def receive_gemini(outcome):
        cid = outcome.task_id
        if cid not in expected or cid in completed:
            raise ExtractionStepError(f"unexpected Gemini summary result: {cid}")
        normalized, violations = inspect_summary(outcome.text)
        tokens = gemini_output_tokens(outcome.response_diagnostics)
        if outcome.error:
            violations.append(outcome.error)
        candidates = (outcome.response_diagnostics or {}).get("candidates") or []
        if any(
            (
                str(getattr(c.get("finish_reason"), "value", c.get("finish_reason")))
                .rsplit(".", 1)[-1]
                .upper()
                == "MAX_TOKENS"
                for c in candidates
            )
        ):
            violations.append("max_tokens")
        if violations:
            failures.record(
                cid,
                None,
                ", ".join(violations),
                "",
                summary_model=model,
                provenance=pending[cid][1],
                tokens=tokens,
            )
            publish(cid, "", violations, raw=True, tokens=tokens)
        else:
            publish(cid, normalized, [], tokens=tokens)
            failures.remove(cid, None)
        completed.add(cid)
        progress.complete(task_id=cid, failed=bool(violations), raw=False)

    pool.generate(
        (pending[cid][2] for cid in next_penalties),
        receive_gemini,
        on_progress=progress.update_stats,
    )
    if completed != expected:
        raise ExtractionStepError("Gemini workers did not complete all summaries")


def run_summary_stage(
    context,
    *,
    arm,
    schema,
    catalog,
    provenance,
    generation,
    model,
    gemini_pool_factory,
    force=False,
    generator_factory,
):
    batch, schedule = prepare_summary_batch(
        context,
        arm=arm,
        schema=schema,
        catalog=catalog,
        provenance=provenance,
        generation=generation,
        model=model,
        force=force,
    )
    content_ids = [str(item["content_id"]) for item in catalog]
    with InferenceProgress(
        total=len(batch.next_penalties),
        reused=len(batch.documents),
        empty=batch.failures.count(content_ids),
        desc=f"Summary {arm.name} ({model})",
        unit="summary",
        progress_factory=tqdm,
    ) as progress:
        if batch.next_penalties:
            if model == "gemini":
                run_gemini_summaries(context, batch, progress, gemini_pool_factory)
            else:
                run_qwen_summaries(context, batch, schedule, progress, generator_factory)
    failed_count = batch.failures.count(content_ids)
    print(f"[SUMMARY] {arm.name}: outputs={len(batch.documents)} failed={failed_count}", flush=True)
    if model == "gemini" and failed_count:
        raise ExtractionStepError(f"{failed_count} Gemini summaries failed; rerun to retry")
    return result(
        f"summarize-{arm.name}", content_count=len(batch.documents), failure_count=failed_count
    )


@contextmanager
def qwen_generator(
    *, model_path: Path, settings=None, image_limit=6, runtime=None, log=None, on_progress=None
) -> Iterator[GenerationFunction]:

    def ready(event):
        if runtime:
            runtime.engine_ready(event)
        if log:
            log(
                f"[Qwen] GPU {event['gpu_id']} ready | {event['gpu_name']} | settings={event['settings']}"
            )

    with ExitStack() as resources:
        pool = None

        def generate(tasks, on_task_complete=None):
            nonlocal pool
            if pool is None:
                if log:
                    log("[Qwen] starting vLLM GPU workers")
                pool = resources.enter_context(
                    QwenWorkerPool(
                        str(model_path),
                        settings=settings,
                        image_limit=image_limit,
                        on_runtime=ready,
                        on_progress=on_progress,
                    )
                )

            def complete(task_id, text):
                if runtime:
                    runtime.current_result = pool.last_result
                try:
                    on_task_complete(task_id, text)
                finally:
                    if runtime:
                        runtime.current_result = None

            return pool.generate(tasks, complete if on_task_complete else None)

        yield generate
