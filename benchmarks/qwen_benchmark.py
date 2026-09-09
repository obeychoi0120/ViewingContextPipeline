"""Export fixed requests, then measure each backend in a fresh Linux process."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool, _visible_gpu_ids
from extraction.descriptions import description_summary_prompt, validate_summary as validate_description
from extraction.evidence import load_images
from extraction.qwen_config import qwen_settings
from extraction.qwen_runtime import result_hash
from extraction.semantic_graph import graph_summary_prompt, parse_or_repair_graph, validate_summary
from extraction.step_support import (
    minimal_description_records, minimal_graph_records, scene_generation_rows, visual_rows,
)
from extraction.steps import _summary_generation_settings
from pipeline_runtime import RunContext, read_json, read_jsonl, write_json


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_requests(context, stage, limit):
    arm = "description" if stage.startswith("description") else "graph"
    summary = "summary" in stage
    settings = context.config["extraction"][arm]
    template = context.config_path("extraction", arm, "summary_prompt" if summary else "scene_prompt").read_text(encoding="utf-8")
    tasks = []
    for visual in visual_rows(context):
        if summary:
            source = "gemini" if stage.endswith("gemini") else "qwen"
            scene_dir = context.description_scene_dir if arm == "description" else context.graph_scene_dir(source)
            path = scene_dir / f"{visual['content_id']}.jsonl"
            records = read_jsonl(path)
            if not records:
                continue
            normalize = minimal_description_records if arm == "description" else minimal_graph_records
            build_prompt = description_summary_prompt if arm == "description" else graph_summary_prompt
            task = QwenGenerationTask(str(visual["content_id"]), (), build_prompt(template, normalize(records, path)),
                                      int(settings["summary_max_new_tokens"]), **_summary_generation_settings(context))
            tasks.append(task)
        else:
            tasks.extend(row["task"] for row in scene_generation_rows(
                visual, prompt=template, max_new_tokens=int(settings["scene_max_new_tokens"]),
                repetition_penalty=float(context.config["extraction"][f"{arm}_repetition_penalty"]),
            ))
        if len(tasks) >= limit:
            break
    tasks = tasks[:limit]
    if not tasks:
        raise ValueError("no prepared requests available for this stage")
    return {
        "schema_version": "qwen-benchmark-requests/v1", "stage": stage,
        "model_path": str(context.path("models", "qwen")),
        "qwen": qwen_settings(context.config["extraction"].get("qwen")),
        "image_limit": context.config["extraction"]["visual_evidence"]["num_keyframes"],
        "requests": [asdict(task) for task in tasks],
        "image_hashes": {path: file_hash(path) for task in tasks for path in task.image_paths},
    }


class GpuMonitor:
    """Sample dedicated GPUs including vLLM child processes; MiB at 0.5 s intervals."""

    def __init__(self, ids):
        self.ids = ids
        self.peaks = {}
        self.samples = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "-i", ",".join(self.ids), "--query-gpu=uuid,memory.used",
                     "--format=csv,noheader,nounits"], text=True, timeout=5,
                )
                for line in output.splitlines():
                    gpu, memory = line.split(",")
                    self.peaks[gpu.strip()] = max(self.peaks.get(gpu.strip(), 0), float(memory))
                self.samples += 1
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def check_output(stage, text):
    try:
        if "summary" in stage:
            (validate_description if stage.startswith("description") else validate_summary)(text)
        elif stage.startswith("graph"):
            parsed = parse_or_repair_graph(text)
            if parsed.graph is None:
                raise ValueError(parsed.error)
        elif not text.strip():
            raise ValueError("empty description")
    except Exception as exc:
        return str(exc)
    return None


def measure(manifest, backend, settings, warmup_count, output_path):
    import torch

    gpu_ids = _visible_gpu_ids(1)
    tasks = [QwenGenerationTask(**{**row, "image_paths": tuple(row["image_paths"])}) for row in manifest["requests"]]
    if not 0 < warmup_count < len(tasks):
        raise ValueError("warmup must leave at least one disjoint measurement request")
    for path, expected in manifest["image_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError(f"benchmark image changed: {path}")
    report = {
        "backend": backend, "requests_hash": result_hash(manifest), "stage": manifest["stage"],
        "warmup_count": warmup_count, "request_count": len(tasks) - warmup_count,
        "settings": settings, "results": [], "runtime": [], "phase": "startup",
        "reference_sha256": file_hash(Path(__file__).with_name("qwen_transformers_reference.py")),
        "startup_seconds": None, "warmup_seconds": None, "steady_seconds": None,
    }
    pool = None
    fatal = None
    measured_start = None
    started = time.monotonic()
    with GpuMonitor(gpu_ids) as monitor:
        try:
            if backend == "vllm":
                pool = QwenWorkerPool(1, manifest["model_path"], settings=settings,
                                      image_limit=manifest["image_limit"], on_runtime=report["runtime"].append)
                pool.wait_ready()

                def run(items, callback):
                    pool.generate(items, lambda task_id, text: callback(
                        task_id, text, pool.last_result["output_tokens"], pool.last_result["prompt_tokens"],
                    ))
            else:
                from importlib.metadata import version
                from benchmarks.qwen_transformers_reference import QwenBackend

                model = QwenBackend.from_pretrained(manifest["model_path"])
                report["runtime"].append({
                    "versions": {name: version(name) for name in ("torch", "transformers")},
                    "gpu_name": torch.cuda.get_device_name(0), "dtype": "bfloat16",
                    "generation_config": model.model.generation_config.to_dict(),
                })
                original = model.model.generate
                counts = {}

                def counted_generate(**kwargs):
                    result = original(**kwargs)
                    counts["prompt"] = kwargs["input_ids"].shape[-1]
                    counts["output"] = result.shape[-1] - counts["prompt"]
                    return result

                model.model.generate = counted_generate

                def run(items, callback):
                    for task in items:
                        images = load_images(list(task.image_paths))
                        try:
                            values = asdict(task)
                            for key in ("task_id", "image_paths"):
                                del values[key]
                            text = model.generate(images, **values)
                            callback(task.task_id, text, counts["output"], counts["prompt"])
                        finally:
                            for image in images:
                                image.close()

            report["startup_seconds"] = time.monotonic() - started
            report["phase"] = "warmup"
            started = time.monotonic()
            run(tasks[:warmup_count], lambda *args: None)
            report["warmup_seconds"] = time.monotonic() - started
            report["phase"] = "steady"
            measured_start = time.monotonic()
            run(tasks[warmup_count:], lambda task_id, text, output, prompt: report["results"].append({
                "task_id": task_id, "text": text, "output_tokens": output, "prompt_tokens": prompt,
            }))
            report["steady_seconds"] = time.monotonic() - measured_start
            report["phase"] = "complete"
        except BaseException as exc:
            fatal = exc
            report["error"] = f"{type(exc).__name__}: {exc}"
            if measured_start is not None:
                report["steady_seconds"] = time.monotonic() - measured_start
        finally:
            if pool is not None:
                pool.abort() if fatal else pool.close()
    # Validation and report serialization are outside the timed inference region.
    for row in report["results"]:
        row["validation_error"] = check_output(manifest["stage"], row["text"])
    report["peak_gpu_memory_mib"] = monitor.peaks
    report["memory_samples"] = monitor.samples
    report["memory_sampling_error"] = monitor.error
    seconds = report["steady_seconds"]
    count = len(report["results"])
    report["requests_per_second"] = count / seconds if seconds else None
    report["output_tokens_per_second"] = sum(row["output_tokens"] for row in report["results"]) / seconds if seconds else None
    report["incomplete_requests"] = report["request_count"] - count
    report["invalid_outputs"] = sum(row["validation_error"] is not None for row in report["results"])
    report["failure_rate"] = (report["incomplete_requests"] + report["invalid_outputs"]) / report["request_count"]
    write_json(output_path, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"results", "runtime"}}, indent=2))
    if fatal:
        raise fatal
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--stage", required=True, choices=["graph-scenes", "description-scenes", "graph-summary-qwen", "graph-summary-gemini", "description-summary"])
    export.add_argument("--limit", type=int, default=144)
    export.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--requests", type=Path, required=True)
    run.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    run.add_argument("--warmup", type=int, default=16)
    run.add_argument("--max-num-seqs", type=int)
    run.add_argument("--max-num-batched-tokens", type=int)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path to preserve earlier measurements")
    if args.command == "export":
        if args.limit < 2:
            parser.error("--limit must be at least 2")
        write_json(args.output, export_requests(RunContext.load(args.run_id), args.stage, args.limit))
    else:
        manifest = read_json(args.requests)
        settings = dict(manifest["qwen"])
        for key in ("max_num_seqs", "max_num_batched_tokens"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        measure(manifest, args.backend, qwen_settings(settings), args.warmup, args.output)


if __name__ == "__main__":
    main()
