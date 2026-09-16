"""Summary generation with failure-only repetition penalty passes."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Callable, Iterator
from tqdm import tqdm
from extraction.progress import InferenceProgress
from extraction.qwen_runtime import QwenRuntime

from artifact_io import atomic_write_json
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.descriptions import description_summary_prompt
from extraction.errors import ExtractionStepError
from extraction.input_tracking import clear_dirty
from extraction.recovery import fingerprint
from extraction.failures import FailureLog
from extraction.generation import generate_penalty_passes
from extraction.scene_storage import read_scene_records
from extraction.semantic_graph import graph_summary_prompt
from extraction.step_support import minimal_description_records, minimal_graph_records, result
from extraction.summary_validation import SUMMARY_SCHEMA_VERSION, inspect_summary
from pipeline_runtime import read_json

GenerationFunction = Callable


def reuse_summary_document(
    output_path, *, schema_version=SUMMARY_SCHEMA_VERSION, content_id, arm, scene_count=None
):
    try:
        doc = read_json(output_path)
        if (
            doc.get("schema_version") != schema_version
            or doc.get("content_id") != content_id
            or doc.get("arm") != arm
            or type(doc.get("scene_count")) is not int
            or doc["scene_count"] <= 0
            or scene_count is not None
            and doc["scene_count"] != scene_count
            or doc.get("status") not in {"complete", "raw_fallback"}
            or not isinstance(doc.get("provenance"), dict)
        ):
            raise ValueError("summary identity, provenance, status, or scene count mismatch")
        text, violations = inspect_summary(doc.get("text"))
        if not text or doc.get("word_count") != len(text.split()):
            raise ValueError("empty summary or incorrect word count")
        if doc["status"] == "complete" and (violations or doc.get("violations") != []):
            raise ValueError("normal summary violates the paragraph contract")
        if doc["status"] == "raw_fallback" and not isinstance(doc.get("violations"), list):
            raise ValueError("raw summary requires recorded violations")
        return doc
    except (ValueError, KeyError, TypeError) as exc:
        raise ExtractionStepError(f"invalid summary {output_path}: {exc}") from exc


def run_summary_stage(
    context,
    *,
    arm,
    schema,
    catalog,
    provenance,
    generation,
    force=False,
    generator_factory,
):
    output_dir = context.extraction_dir(arm.representation, arm.model, "summaries")
    scene_dir = context.extraction_dir(arm.representation, arm.model, "scenes")
    failures = FailureLog(output_dir)
    content_ids = [str(item["content_id"]) for item in catalog]
    if force:
        failures.clear_contents(content_ids)
    template = schema.read_text(encoding="utf-8")
    graph = arm.representation == "graph"
    normalize = minimal_graph_records if graph else minimal_description_records
    build_prompt = graph_summary_prompt if graph else description_summary_prompt
    settings = context.config["extraction"]
    max_tokens = settings["graph" if graph else "description"]["summary_max_new_tokens"]
    tasks, pending, documents = [], {}, {}

    for item in catalog:
        cid = str(item["content_id"])
        source = scene_dir / f"{cid}.jsonl"
        output = output_dir / f"{cid}.json"
        if output.is_file() and not force and not failures.contains(cid, None):
            try:
                existing = reuse_summary_document(output, content_id=cid, arm=arm.name)
            except ExtractionStepError:
                pass
            else:
                if existing["status"] == "complete":
                    clear_dirty(output)
                    documents[cid] = existing
                    continue
                failures.record(cid, None, ", ".join(existing["violations"]) or "invalid summary",
                                existing["text"])
        if not source.is_file():
            output.unlink(missing_ok=True)
            failures.record(cid, None, "missing scene input")
            continue
        records = normalize(read_scene_records(source), source)
        if not records:
            output.unlink(missing_ok=True)
            failures.record(cid, None, "empty scene input")
            continue
        raw_count = sum(r.get("status") == "raw_fallback" for r in records)
        prov = {
            **provenance,
            "scene_input_hash": fingerprint(records),
            "scene_path": str(source),
            "normal_scene_count": len(records) - raw_count,
            "raw_scene_count": raw_count,
        }
        prov["input_hash"] = fingerprint(prov)
        task = QwenGenerationTask(
            task_id=cid,
            image_paths=(),
            prompt=build_prompt(template, records),
            max_new_tokens=max_tokens,
            **generation,
        )
        pending[cid] = (records, prov, task)
        tasks.append(task)

    runtime = QwenRuntime()
    with ExitStack() as resources:
        progress = resources.enter_context(InferenceProgress(
            total=len(tasks), reused=len(documents), empty=failures.count(content_ids),
            desc=f"Summary {arm.name}", unit="summary", progress_factory=tqdm,
        ))

        def receive(cid, text, *, final):
            records, prov, _ = pending[cid]
            normalized, violations = inspect_summary(text)
            event = runtime.current_result or {}
            if event.get("finish_reason") == "length":
                violations.append("max_tokens")
            output = output_dir / f"{cid}.json"
            if normalized and (not violations or final):
                doc = {
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    "content_id": cid,
                    "arm": arm.name,
                    "status": "raw_fallback" if violations else "complete",
                    "text": text if violations else normalized,
                    "scene_count": len(records),
                    "word_count": len(normalized.split()),
                    "violations": violations,
                    "correction_count": 0,
                    "provenance": prov,
                }
                atomic_write_json(output, doc, durable=True)
                clear_dirty(output)
                documents[cid] = doc
            else:
                output.unlink(missing_ok=True)
                documents.pop(cid, None)
            if violations:
                failures.record(cid, None, ", ".join(violations), text)
            else:
                failures.remove(cid, None)
            progress.complete(task_id=cid, failed=bool(violations),
                              raw=bool(final and normalized and violations))
            return bool(violations)

        if tasks:
            generate = resources.enter_context(generator_factory(
                model_path=context.path("models", "qwen"), settings=settings.get("qwen"),
                image_limit=settings["visual_evidence"]["num_keyframes"], runtime=runtime,
                on_progress=progress.update_stats, log=lambda message: print(message, flush=True),
            ))
            generate_penalty_passes(generate, tasks, settings["summary_repetition_penalty"], receive,
                                    log=lambda message: print(message, flush=True))
    failed_count = failures.count(content_ids)
    print(f"[SUMMARY] {arm.name}: outputs={len(documents)} failed={failed_count}", flush=True)
    return result(f"summarize-{arm.name}", content_count=len(documents), failure_count=failed_count)


def qwen_progress(progress, stats):
    progress.update_stats(stats)


@contextmanager
def qwen_generator(
    *,
    model_path: Path,
    settings=None,
    image_limit=6,
    runtime=None,
    log=None,
    on_progress=None,
) -> Iterator[GenerationFunction]:
    def ready(event):
        if runtime:
            runtime.engine_ready(event)
        if log:
            log(
                f"[Qwen] GPU {event['gpu_id']} ready | {event['gpu_name']} | "
                f"settings={event['settings']}"
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
