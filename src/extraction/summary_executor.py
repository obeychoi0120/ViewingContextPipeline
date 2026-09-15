"""Durable summary drafts, one format correction, and final embedded provenance."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4
from tqdm import tqdm
from extraction.progress import InferenceProgress
from extraction.qwen_runtime import QwenRuntime

from artifact_io import atomic_write_json
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool
from extraction.descriptions import description_summary_prompt
from extraction.errors import ExtractionStepError
from extraction.input_tracking import clear_dirty, input_state_path
from extraction.recovery import (
    active_force_run,
    clear_recovery,
    fingerprint,
    generate_with_recovery,
)
from extraction.semantic_graph import graph_summary_prompt
from extraction.step_support import minimal_description_records, minimal_graph_records, result
from extraction.structured_output import OutputValidationError
from extraction.summary_validation import SUMMARY_SCHEMA_VERSION, inspect_summary
from pipeline_runtime import read_json, read_jsonl

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
    recovery_dir = output_dir / ".recovery"
    if force:
        atomic_write_json(recovery_dir / ".force-run", {"force_run_id": str(uuid4())}, durable=True)
    force_run = active_force_run(recovery_dir)
    template = schema.read_text(encoding="utf-8")
    graph = arm.representation == "graph"
    normalize = minimal_graph_records if graph else minimal_description_records
    build_prompt = graph_summary_prompt if graph else description_summary_prompt
    settings = context.config["extraction"]
    max_tokens = settings["graph" if graph else "description"]["summary_max_new_tokens"]
    tasks, pending, drafts, documents, failures = [], {}, {}, {}, {}

    def save_final(cid, document):
        output = output_dir / f"{cid}.json"
        atomic_write_json(output, document, durable=True)
        clear_dirty(output)
        (output_dir / "failures" / f"{cid}.jsonl").unlink(missing_ok=True)
        (output_dir / ".pending" / f"{cid}.json").unlink(missing_ok=True)
        documents[cid] = document
        progress.complete()

    for item in catalog:
        cid = str(item["content_id"])
        source = scene_dir / f"{cid}.jsonl"
        if not source.is_file():
            if arm.model == "gemini" and not (output_dir / f"{cid}.json").exists():
                failures[cid] = "missing Gemini scenes"
                continue
            raise ExtractionStepError(f"missing scene input: {source}")
        records = normalize(read_jsonl(source), source)
        if not records:
            failures[cid] = "empty scene input"
            continue
        if any(
            record.get("provenance", {}).get("arm") != arm.name
            or record.get("provenance", {}).get("representation") != arm.representation
            for record in records
        ):
            raise ExtractionStepError(
                f"scene source provenance mismatch: {source}; rerun extraction"
            )
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
        output = output_dir / f"{cid}.json"
        if output.is_file() and not force and not input_state_path(output).exists():
            try:
                existing = reuse_summary_document(
                    output, content_id=cid, arm=arm.name, scene_count=len(records)
                )
            except ExtractionStepError:
                pass
            else:
                if existing["provenance"] == prov and (
                    not force_run or existing.get("generation", {}).get("force_run_id") == force_run
                ):
                    documents[cid] = existing
                    (output_dir / "failures" / f"{cid}.jsonl").unlink(missing_ok=True)
                    continue
        draft_path = output_dir / ".pending" / f"{cid}.json"
        if draft_path.is_file() and not force:
            draft = read_json(draft_path)
            if draft.get("provenance") == prov and (
                not force_run or draft.get("generation", {}).get("force_run_id") == force_run
            ):
                drafts[cid] = draft
                continue
        tasks.append(task)

    def document(cid, text, *, corrected=False):
        normalized, violations = inspect_summary(text)
        if not normalized:
            raise OutputValidationError("empty summary")
        records, prov, _ = pending[cid]
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "content_id": cid,
            "arm": arm.name,
            "status": "raw_fallback" if violations else "complete",
            "text": text.strip() if violations else normalized,
            "scene_count": len(records),
            "word_count": len(normalized.split()),
            "violations": violations,
            "correction_count": int(corrected),
            "provenance": prov,
        }

    def draft_complete(cid, doc):
        if doc["violations"]:
            # Persist the first response before scheduling the separate correction pass.
            atomic_write_json(output_dir / ".pending" / f"{cid}.json", doc, durable=True)
            drafts[cid] = doc
        else:
            save_final(cid, doc)

    def failed(cid, attempts):
        from pipeline_runtime import write_jsonl

        failures[cid] = attempts[-1]["error"]
        progress.complete(failed=True)
        write_jsonl(output_dir / "failures" / f"{cid}.jsonl", attempts)

    runtime = QwenRuntime()
    with ExitStack() as resources:
        progress = resources.enter_context(
            InferenceProgress(
                total=len(tasks) + len(drafts),
                reused=len(documents),
                empty=len(failures),
                desc=f"Summary {arm.name}",
                unit="summary",
                progress_factory=tqdm,
            )
        )
        generate = resources.enter_context(
            generator_factory(
                model_path=context.path("models", "qwen"),
                settings=settings.get("qwen"),
                image_limit=settings["visual_evidence"]["num_keyframes"],
                runtime=runtime,
                on_progress=progress.update_stats,
                log=lambda message: print(message, flush=True),
            )
        )
        if tasks:
            generate_with_recovery(
                generate,
                tasks,
                penalties=settings["summary_repetition_penalty"],
                directory=recovery_dir,
                identity=provenance,
                validate=lambda cid, text: (document(cid, text), "native"),
                complete=draft_complete,
                failed=failed,
                runtime=runtime,
                rounds_across_batches=arm.model == "gemini",
            )
        correction_tasks = [
            replace(
                pending[cid][2],
                prompt=(
                    pending[cid][2].prompt
                    + "\n\nDraft to correct:\n"
                    + doc["text"]
                    + "\n\nRewrite once into one natural paragraph of at most 200 words, retaining only "
                    "important information supported by the original observations. Return only the paragraph."
                ),
            )
            for cid, doc in drafts.items()
        ]

        def correction_validate(cid, text):
            if not isinstance(text, str) or not text.strip():
                raw = {**drafts[cid], "status": "raw_fallback", "correction_count": 1}
                raw["violations"] = [*raw["violations"], "empty_correction"]
                return raw, "raw_fallback"
            return document(cid, text, corrected=True), "corrected"

        def correction_complete(cid, doc):
            doc["generation"] = {
                **doc.get("generation", {}),
                "force_run_id": force_run,
                "draft": drafts[cid].get("generation"),
            }
            save_final(cid, doc)

        if correction_tasks:
            generate_with_recovery(
                generate,
                correction_tasks,
                penalties=[generation["repetition_penalty"]],
                directory=recovery_dir / "correction",
                identity=provenance,
                validate=correction_validate,
                complete=correction_complete,
                failed=failed,
                runtime=runtime,
            )
    if not failures:
        clear_recovery(output_dir)
    temporary = output_dir / ".pending"
    if temporary.exists() and not any(temporary.iterdir()):
        temporary.rmdir()
    blocking = [
        cid for cid in failures if arm.model != "gemini" or (output_dir / f"{cid}.json").exists()
    ]
    if blocking:
        raise ExtractionStepError(f"summary failed: {blocking[:10]}")
    print(f"[SUMMARY] {arm.name}: complete={len(documents)} missing={len(failures)}", flush=True)
    return result(
        f"summarize-{arm.name}", content_count=len(documents), failure_count=len(failures)
    )


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
