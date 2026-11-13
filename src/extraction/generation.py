"""Streaming generation and failure-only passes without generation journals."""

from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Callable, Iterator

from extraction.backends.qwen_workers import QwenWorkerPool


def generate_once(generate, tasks, complete):
    admitted, completed = set(), set()

    def requests():
        for task in tasks:
            if task.task_id in admitted:
                raise ValueError(f"duplicate generation task: {task.task_id}")
            admitted.add(task.task_id)
            yield task

    def receive(task_id, text):
        if task_id not in admitted:
            raise RuntimeError(f"unexpected generation result: {task_id}")
        if task_id not in completed:
            complete(task_id, text)
            completed.add(task_id)

    returned = generate(requests(), receive)
    for task_id in admitted - completed:
        if task_id not in returned:
            raise RuntimeError(f"missing generation result: {task_id}")
        receive(task_id, returned[task_id])


def generate_penalty_passes(generate, tasks, penalties, complete, log=None, on_pass=None):
    """Finish each pass before retrying failures; keep only pending tasks in memory.

    complete(task_id, text, final=...) publishes the outcome and returns whether to retry
    (anything but success, REPEAT_OCCURRED included).
    """
    from dataclasses import replace
    from extraction.qwen_config import penalty_schedule

    schedule = penalty_schedule(penalties)
    for index, penalty in enumerate(schedule):
        active, retry = {}, []
        final = index == len(schedule) - 1
        if on_pass:
            on_pass(index + 1, len(schedule), None if index == 0 else len(tasks))
        if log:
            log(f"[vLLM] repetition_penalty={penalty:.2f} pass={index + 1}/{len(schedule)}")

        def requests():
            for task in tasks:
                request = replace(task, repetition_penalty=penalty)
                active[request.task_id] = request
                yield request

        def receive(task_id, text):
            task = active.pop(task_id)
            if complete(task_id, text, final=final) and not final:
                retry.append(task)

        generate_once(generate, requests(), receive)
        if not retry:
            break
        tasks = retry


@contextmanager
def qwen_generator(
    *, model_path: Path, settings=None, image_limit=6, runtime=None, log=None, on_progress=None
) -> Iterator[Callable]:

    def ready(event):
        if runtime:
            runtime.engine_ready(event)
        if log:
            log(
                f"[vLLM] GPU {event['gpu_id']} ready | {event.get('model_family', 'Qwen')} | {event['gpu_name']} | settings={event['settings']}"
            )

    with ExitStack() as resources:
        pool = None

        def generate(tasks, on_task_complete=None):
            nonlocal pool
            if pool is None:
                if log:
                    log("[vLLM] starting GPU workers")
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
