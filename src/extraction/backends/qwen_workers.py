from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from typing import Callable, Iterable

from extraction.qwen_config import qwen_settings


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
    repetition_penalty: float = 1.0


def _start_worker(context, worker_index, gpu_id, model_path, result_queue, settings, image_limit):
    task_queue = context.Queue(maxsize=2 * settings["max_num_seqs"] + 1)
    process = context.Process(
        target=_worker_main,
        args=(worker_index, gpu_id, model_path, task_queue, result_queue, settings, image_limit),
        daemon=False,
    )
    process.start()
    return task_queue, process


class QwenWorkerPool:
    """One vLLM engine per GPU, with bounded completion-driven admission."""

    def __init__(self, gpu_count, model_path, *, settings=None, image_limit=6,
                 on_runtime=None, on_progress=None):
        self.settings = qwen_settings(settings)
        self.gpu_ids = _visible_gpu_ids(gpu_count)
        self.gpu_count = gpu_count
        self.capacity = 2 * self.settings["max_num_seqs"]
        self.on_runtime = on_runtime
        self.on_progress = on_progress
        self.last_result = None
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._task_queues = []
        self._processes = []
        self._group_leaders = set()
        self._ready_workers = set()
        self._closed = False
        try:
            for index, gpu_id in enumerate(self.gpu_ids):
                tasks, process = _start_worker(
                    self._context, index, gpu_id, model_path, self._result_queue,
                    self.settings, image_limit,
                )
                self._task_queues.append(tasks)
                self._processes.append(process)
        except BaseException:
            self.abort()
            raise

    def wait_ready(self):
        """Separate cold engine startup from timed benchmark requests."""
        try:
            while len(self._ready_workers) < self.gpu_count:
                try:
                    event = self._result_queue.get(timeout=0.5)
                except queue.Empty:
                    if any(not process.is_alive() for process in self._processes):
                        raise RuntimeError("Qwen worker exited during engine startup")
                    continue
                if not self._startup_event(event):
                    raise RuntimeError("wait_ready must be called before submitting requests")
        except BaseException:
            self.abort()
            raise

    def _startup_event(self, event):
        index = event["worker_index"]
        if event.get("kind") == "started":
            if event.get("process_group"):
                self._group_leaders.add(event["process_group"])
        elif not event.get("ok", False):
            raise RuntimeError(f"Qwen worker {index} on GPU {event.get('gpu_id')} "
                               f"failed: {event.get('error')}")
        elif event.get("kind") == "ready":
            self._ready_workers.add(index)
            if self.on_runtime:
                self.on_runtime(event)
        else:
            return False
        return True

    def generate(self, tasks: Iterable[QwenGenerationTask],
                 on_task_complete: Callable[[str, str], None] | None = None) -> dict[str, str]:
        if self._closed:
            raise RuntimeError("Qwen worker pool is closed")
        iterator = iter(tasks)
        pending = {}
        seen = set()
        loads = [0] * self.gpu_count
        exhausted = False
        completed = 0
        generated_tokens = 0
        started = last_progress = time.monotonic()
        results = {}

        def admit(index):
            nonlocal exhausted
            if exhausted:
                return
            task = next(iterator, None)
            if task is None:
                exhausted = True
                return
            if task.task_id in seen:
                raise ValueError("Qwen generation task ids must be unique")
            seen.add(task.task_id)
            pending[task.task_id] = index
            loads[index] += 1
            self._task_queues[index].put(task)

        def report():
            if self.on_progress:
                elapsed = max(time.monotonic() - started, 0.001)
                self.on_progress({
                    "completed": completed, "inflight": len(pending),
                    "requests_per_second": completed / elapsed,
                    "output_tokens_per_second": generated_tokens / elapsed,
                    "gpu_inflight": list(loads),
                })

        try:
            for _ in range(self.capacity):
                for index in range(self.gpu_count):
                    admit(index)
            while pending:
                try:
                    event = self._result_queue.get(timeout=0.5)
                except queue.Empty:
                    dead = [i for i, process in enumerate(self._processes) if not process.is_alive()]
                    if dead:
                        raise RuntimeError(f"Qwen GPU worker(s) exited before finishing tasks: {dead}")
                else:
                    index = event["worker_index"]
                    if not self._startup_event(event):
                        task_id = event["task_id"]
                        if task_id not in pending or pending[task_id] != index:
                            raise RuntimeError(f"Unexpected Qwen completion: {task_id}")
                        del pending[task_id]
                        loads[index] -= 1
                        completed += 1
                        generated_tokens += event.get("output_tokens", 0)
                        self.last_result = event
                        # Refill before parsing/writing the completed request.
                        admit(index)
                        if on_task_complete:
                            on_task_complete(task_id, event["text"])
                        else:
                            results[task_id] = event["text"]
                if time.monotonic() - last_progress >= 30:
                    report()
                    last_progress = time.monotonic()
            report()
            return results
        except BaseException:
            self.abort()
            raise

    def close(self):
        if self._closed:
            return
        try:
            for tasks in self._task_queues:
                tasks.put_nowait(None)
            for process in self._processes:
                process.join(timeout=10)
        finally:
            self._closed = True
            self._force_stop()
            self._dispose_queues()

    def abort(self):
        if self._closed:
            return
        self._closed = True
        self._force_stop()
        self._dispose_queues()

    def _signal_groups(self, sig):
        if os.name != "posix":
            return
        # Each owned worker creates its own session before starting vLLM.
        groups = set(self._group_leaders)
        for process in self._processes:
            try:
                if os.getpgid(process.pid) == process.pid:
                    groups.add(process.pid)
            except ProcessLookupError:
                pass
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass

    def _force_stop(self):
        self._signal_groups(signal.SIGTERM)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=0.5)
        if os.name == "posix":
            self._signal_groups(signal.SIGKILL)
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)

    def _dispose_queues(self):
        for channel in [*self._task_queues, self._result_queue]:
            channel.cancel_join_thread()
            channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close() if exc_type is None else self.abort()


def _visible_gpu_ids(gpu_count):
    if type(gpu_count) is not int or gpu_count <= 0:
        raise ValueError("--gpus must be a positive integer")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Qwen requires the 'qwen' optional dependencies") from exc
    available = int(torch.cuda.device_count())
    if available < gpu_count:
        raise RuntimeError(f"--gpus {gpu_count} requested, but only {available} CUDA device(s) are visible")
    configured = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return configured[:gpu_count] if configured else [str(i) for i in range(gpu_count)]


def _receive(channel):
    try:
        return True, channel.get(timeout=0.25)
    except queue.Empty:
        return False, None


async def _serve_worker(backend, task_queue, result_queue, worker_index, gpu_id, capacity):
    active = set()
    receiver_pool = ThreadPoolExecutor(max_workers=1)
    receiver = None
    stopping = False

    async def generate(task):
        output = await backend.generate(task)
        result_queue.put({
            "kind": "result", "ok": True, "worker_index": worker_index, "gpu_id": gpu_id,
            "task_id": task.task_id, "text": output.text,
            "prompt_tokens": output.prompt_tokens, "output_tokens": output.output_tokens,
            "generation": {key: value for key, value in asdict(task).items()
                           if key not in {"task_id", "image_paths", "prompt"}},
        })

    try:
        while active or not stopping:
            if not stopping and receiver is None and len(active) < capacity:
                receiver = asyncio.get_running_loop().run_in_executor(receiver_pool, _receive, task_queue)
            waiting = active | ({receiver} if receiver is not None else set())
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                if future is receiver:
                    receiver = None
                    received, task = future.result()
                    if received:
                        if task is None:
                            stopping = True
                        else:
                            active.add(asyncio.create_task(generate(task)))
                else:
                    active.remove(future)
                    future.result()
    finally:
        if receiver is not None:
            receiver.cancel()
        for future in active:
            future.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        receiver_pool.shutdown(wait=True, cancel_futures=True)


def _worker_main(worker_index, gpu_id, model_path, task_queue, result_queue,
                 settings=None, image_limit=6):
    process_group = None
    if mp.parent_process() is not None:
        if os.name == "posix":
            os.setsid()
            process_group = os.getpid()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    result_queue.put({"kind": "started", "worker_index": worker_index, "process_group": process_group})

    async def run():
        from extraction.backends.qwen import QwenBackend

        backend = QwenBackend.from_pretrained(model_path, settings=settings, image_limit=image_limit)
        try:
            result_queue.put({
                "kind": "ready", "ok": True, "worker_index": worker_index,
                "gpu_id": gpu_id, "model_path": model_path, **backend.runtime_info,
            })
            await _serve_worker(backend, task_queue, result_queue, worker_index, gpu_id,
                                2 * qwen_settings(settings)["max_num_seqs"])
        finally:
            backend.close()

    try:
        asyncio.run(run())
    except BaseException:
        result_queue.put({
            "ok": False, "worker_index": worker_index, "gpu_id": gpu_id,
            "error": traceback.format_exc(),
        })
