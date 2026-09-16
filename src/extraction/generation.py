"""Stream each request once, without retries or generation journals."""


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
