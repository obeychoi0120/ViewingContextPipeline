from __future__ import annotations

import hashlib
import json
from uuid import uuid4


def result_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class QwenRuntime:
    """In-memory engine metadata attached to recovery checkpoints."""

    def __init__(self, on_ready=None):
        self.on_ready = on_ready
        self.execution_id = str(uuid4())
        self.engines = {}
        self.current_result = None

    def engine_ready(self, event):
        self.engines[event["worker_index"]] = event
        if self.on_ready:
            self.on_ready({"execution_id": self.execution_id, "engines": self.engines})

    def checkpoint_origin(self):
        if self.current_result is None:
            return None
        event = self.current_result
        return {
            "execution_id": self.execution_id,
            "result": event,
            "engine": self.engines[event["worker_index"]],
        }
