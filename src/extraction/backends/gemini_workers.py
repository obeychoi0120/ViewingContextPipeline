from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Callable, Iterable

from extraction.backends.base import VLMBackend
from extraction.backends.gemini import GeminiBackend
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.evidence import load_images


@dataclass(frozen=True)
class GeminiGenerationOutcome:
    task_id: str
    text: str
    error: str | None = None


class GeminiWorkerPool:
    """Concurrent Vertex calls that stop waiting immediately on Ctrl+C."""

    def __init__(
        self,
        concurrency: int,
        *,
        project_id: str,
        location: str,
        model_id: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
        backend_factory: Callable[[], VLMBackend] | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("Gemini concurrency must be a positive integer")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("Gemini max_output_tokens must be a positive integer")
        self.concurrency = concurrency
        self.max_output_tokens = max_output_tokens
        self._backend_factory = backend_factory or (
            lambda: GeminiBackend.vertex(
                project_id=project_id,
                location=location,
                model_id=model_id,
                temperature=temperature,
                thinking_level=thinking_level,
            )
        )

    def generate(
        self,
        tasks: Iterable[QwenGenerationTask],
        on_task_complete: Callable[[GeminiGenerationOutcome], None] | None = None,
    ) -> dict[str, GeminiGenerationOutcome]:
        task_list = list(tasks)
        task_ids = [task.task_id for task in task_list]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Gemini generation task ids must be unique")
        if not task_list:
            return {}

        task_queue: queue.Queue[QwenGenerationTask] = queue.Queue()
        result_queue: queue.Queue[GeminiGenerationOutcome] = queue.Queue()
        stop = threading.Event()
        for task in task_list:
            task_queue.put(task)

        workers = [
            threading.Thread(
                target=self._worker,
                args=(task_queue, result_queue, stop),
                name=f"gemini-worker-{index}",
                daemon=True,
            )
            for index in range(min(self.concurrency, len(task_list)))
        ]
        for worker in workers:
            worker.start()

        outcomes: dict[str, GeminiGenerationOutcome] = {}
        interrupted = False
        try:
            while len(outcomes) < len(task_list):
                try:
                    outcome = result_queue.get(timeout=0.1)
                except queue.Empty:
                    if not any(worker.is_alive() for worker in workers):
                        missing = sorted(set(task_ids) - set(outcomes))
                        raise RuntimeError(
                            f"Gemini workers stopped before completing tasks: {missing}"
                        )
                    continue
                outcomes[outcome.task_id] = outcome
                if on_task_complete is not None:
                    on_task_complete(outcome)
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            self._discard_pending(task_queue)
            raise
        finally:
            if not interrupted:
                stop.set()
                for worker in workers:
                    worker.join()
        return outcomes

    def _worker(
        self,
        task_queue: queue.Queue[QwenGenerationTask],
        result_queue: queue.Queue[GeminiGenerationOutcome],
        stop: threading.Event,
    ) -> None:
        backend: VLMBackend | None = None
        while not stop.is_set():
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            if stop.is_set():
                return
            try:
                if backend is None:
                    backend = self._backend_factory()
                text = backend.generate(
                    load_images(list(task.image_paths)),
                    task.prompt,
                    self.max_output_tokens or task.max_new_tokens,
                )
                outcome = GeminiGenerationOutcome(task.task_id, text)
            except Exception as exc:  # SDK errors are persisted per scene by the caller.
                outcome = GeminiGenerationOutcome(
                    task.task_id,
                    "",
                    f"{type(exc).__name__}: {exc}",
                )
            result_queue.put(outcome)

    @staticmethod
    def _discard_pending(task_queue: queue.Queue[QwenGenerationTask]) -> None:
        while True:
            try:
                task_queue.get_nowait()
            except queue.Empty:
                return
