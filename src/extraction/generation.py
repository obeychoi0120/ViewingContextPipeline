"""Streaming generation and failure-only passes without generation journals."""


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

    complete(task_id, text, final=...) publishes the outcome and returns failure.
    """
    from dataclasses import replace
    from extraction.recovery import penalty_schedule

    schedule = penalty_schedule(penalties)
    for index, penalty in enumerate(schedule):
        active, retry = {}, []
        final = index == len(schedule) - 1
        if on_pass:
            on_pass(index + 1, len(schedule), None if index == 0 else len(tasks))
        if log:
            log(f"[Qwen] repetition_penalty={penalty:.2f} pass={index + 1}/{len(schedule)}")

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
