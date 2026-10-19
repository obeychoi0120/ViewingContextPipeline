from __future__ import annotations

from uuid import uuid4


class QwenRuntime:
    """In-memory engine metadata attached to recovery checkpoints."""

    def __init__(self):
        self.execution_id = str(uuid4())
        self.engines = {}
        self.current_result = None

    def engine_ready(self, event):
        self.engines[event["worker_index"]] = event

    def checkpoint_origin(self):
        if self.current_result is None:
            return None
        event = self.current_result
        return {
            "execution_id": self.execution_id,
            "result": event,
            "engine": self.engines[event["worker_index"]],
        }
