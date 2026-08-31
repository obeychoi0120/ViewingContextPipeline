from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class QwenGenerationTask:
    task_id: str
    image_paths: tuple[str, ...]
    prompt: str
    max_new_tokens: int
    do_sample: bool = False
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


def assign_worker_indices(task_count: int, gpu_count: int) -> list[int]:
    if gpu_count <= 0:
        raise ValueError("--gpus must be a positive integer")
    return [index % gpu_count for index in range(task_count)]


class QwenWorkerPool:
    """One persistent Qwen process per requested CUDA device."""

    def __init__(self, gpu_count: int, model_path: str) -> None:
        gpu_ids = _visible_gpu_ids(gpu_count)
        self.gpu_count = gpu_count
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._task_queues: list[Any] = []
        self._processes: list[Any] = []
        for worker_index, gpu_id in enumerate(gpu_ids):
            task_queue = self._context.Queue()
            process = self._context.Process(
                target=_worker_main,
                args=(
                    worker_index,
                    gpu_id,
                    model_path,
                    task_queue,
                    self._result_queue,
                ),
            )
            process.start()
            self._task_queues.append(task_queue)
            self._processes.append(process)

    def generate(
        self,
        tasks: Iterable[QwenGenerationTask],
        on_task_complete: Callable[[str, str], None] | None = None,
    ) -> dict[str, str]:
        task_list = list(tasks)
        task_ids = [task.task_id for task in task_list]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Qwen generation task ids must be unique")
        pending: dict[str, int] = {}
        for task, worker_index in zip(
            task_list,
            assign_worker_indices(len(task_list), self.gpu_count),
        ):
            pending[task.task_id] = worker_index
            self._task_queues[worker_index].put(task)

        results: dict[str, str] = {}
        while pending:
            try:
                result = self._result_queue.get(timeout=5)
            except queue.Empty:
                dead = {
                    worker_index
                    for worker_index in pending.values()
                    if not self._processes[worker_index].is_alive()
                }
                if dead:
                    raise RuntimeError(
                        f"Qwen GPU worker(s) exited before finishing tasks: {sorted(dead)}"
                    )
                continue
            task_id = str(result.get("task_id"))
            if not result.get("ok"):
                worker_index = result.get("worker_index", "unknown")
                gpu_id = result.get("gpu_id", "unknown")
                error = result.get("error", "unknown error")
                raise RuntimeError(
                    f"Qwen worker {worker_index} on GPU {gpu_id} failed:\n{error}"
                )
            if task_id in pending:
                text = str(result["text"])
                results[task_id] = text
                del pending[task_id]
                if on_task_complete is not None:
                    on_task_complete(task_id, text)
        return results

    def close(self) -> None:
        for task_queue in self._task_queues:
            task_queue.put(None)
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def __enter__(self) -> QwenWorkerPool:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _visible_gpu_ids(gpu_count: int) -> list[str]:
    if gpu_count <= 0:
        raise ValueError("--gpus must be a positive integer")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Qwen extraction requires the 'qwen' optional dependencies") from exc
    available = int(torch.cuda.device_count())
    if available < gpu_count:
        raise RuntimeError(
            f"--gpus {gpu_count} requested, but only {available} CUDA device(s) are visible"
        )
    configured = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if configured:
        return configured[:gpu_count]
    return [str(index) for index in range(gpu_count)]


def _worker_main(
    worker_index: int,
    gpu_id: str,
    model_path: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from extraction.backends.qwen import QwenBackend

        backend = QwenBackend.from_pretrained(model_path, use_fc_patch=True)
    except BaseException:
        result_queue.put(
            {
                "ok": False,
                "worker_index": worker_index,
                "gpu_id": gpu_id,
                "task_id": None,
                "error": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            from extraction.evidence import load_images

            text = backend.generate(
                load_images(list(task.image_paths)),
                task.prompt,
                task.max_new_tokens,
                do_sample=task.do_sample,
                seed=task.seed,
                temperature=task.temperature,
                top_p=task.top_p,
                top_k=task.top_k,
            )
            result_queue.put(
                {
                    "ok": True,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "task_id": task.task_id,
                    "text": text,
                }
            )
        except BaseException:
            result_queue.put(
                {
                    "ok": False,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                    "task_id": task.task_id,
                    "error": traceback.format_exc(),
                }
            )
