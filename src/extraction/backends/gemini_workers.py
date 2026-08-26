from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    """Threaded Vertex Gemini calls with one lazily-created client per worker."""

    def __init__(
        self,
        concurrency: int,
        *,
        project_id: str,
        location: str,
        model_id: str,
        backend_factory: Callable[[], VLMBackend] | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("Gemini concurrency must be a positive integer")
        self.concurrency = concurrency
        self._local = threading.local()
        self._backend_factory = backend_factory or (
            lambda: GeminiBackend.vertex(
                project_id=project_id,
                location=location,
                model_id=model_id,
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
        outcomes: dict[str, GeminiGenerationOutcome] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            pending = {executor.submit(self._generate_one, task): task for task in task_list}
            for future in as_completed(pending):
                task = pending[future]
                try:
                    outcome = GeminiGenerationOutcome(task.task_id, future.result())
                except Exception as exc:  # SDK errors are persisted per scene by the caller.
                    outcome = GeminiGenerationOutcome(
                        task.task_id,
                        "",
                        f"{type(exc).__name__}: {exc}",
                    )
                outcomes[task.task_id] = outcome
                if on_task_complete is not None:
                    on_task_complete(outcome)
        return outcomes

    def _generate_one(self, task: QwenGenerationTask) -> str:
        backend = getattr(self._local, "backend", None)
        if backend is None:
            backend = self._backend_factory()
            self._local.backend = backend
        return backend.generate(
            load_images(list(task.image_paths)),
            task.prompt,
            task.max_new_tokens,
        )
