from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4


def result_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class QwenRuntimeLog:
    """Parent-owned append-only provenance; inference caches retain their schemas."""

    def __init__(self, run_root: Path, stage: str):
        self.path = run_root / "extraction" / "qwen_runtime.jsonl"
        self.stage = stage
        self.execution_id = str(uuid4())
        self.engines = {}
        self.current_result = None
        self.completed = {}
        self._read()

    def _read(self):
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            offset = 0
            size = self.path.stat().st_size
            for raw in handle:
                end = offset + len(raw)
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    if end == size and not raw.endswith(b"\n"):
                        # Only a torn final append can be discarded.
                        with self.path.open("r+b") as repair:
                            repair.truncate(offset)
                        return
                    raise ValueError(f"corrupt Qwen runtime log at byte {offset}: {self.path}") from exc
                if (not isinstance(event, dict) or event.get("schema_version") != "qwen-runtime/v1"
                        or event.get("event") not in {"engine_ready", "result"}
                        or not all(key in event for key in ("stage", "execution_id"))):
                    raise ValueError(f"invalid Qwen runtime record at byte {offset}: {self.path}")
                if event["event"] == "result":
                    if not all(key in event for key in ("task_id", "result_hash", "status")):
                        raise ValueError(f"invalid Qwen result provenance: {self.path}")
                    self.completed[(event["stage"], event.get("artifact_id", event["task_id"]))] = event
                offset = end
            if size and not raw.endswith(b"\n"):
                with self.path.open("ab") as repair:
                    repair.write(b"\n")

    def _append(self, event):
        event = {
            "schema_version": "qwen-runtime/v1", "execution_id": self.execution_id,
            "stage": self.stage, "time": datetime.now(timezone.utc).isoformat(), **event,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))

    def engine_ready(self, event):
        self.engines[event["worker_index"]] = event
        self._append({"event": "engine_ready", **{k: v for k, v in event.items() if k not in {"ok", "kind"}}})

    def classification(self, task_id, value):
        previous = self.completed.get((self.stage, task_id))
        if (previous and previous["status"] in {"complete", "raw_fallback"}
                and previous["result_hash"] == result_hash(value)):
            return "vllm"
        return "legacy_unknown"

    def checkpoint_origin(self):
        if self.current_result is None:
            return None
        event = self.current_result
        return {"execution_id": self.execution_id, "result": event,
                "engine": self.engines[event["worker_index"]]}

    def record_replayed(self, task_id, value, origin):
        if not origin:
            return
        event = origin["result"]
        if event["task_id"] != task_id:
            raise RuntimeError(f"replayed Qwen result has wrong task: {task_id}")
        artifact_id = (f"{task_id.rsplit(':', 1)[0]}:{value['scene_idx']}"
                       if "scene_idx" in value else task_id)
        self._append({
            "event": "result", "task_id": task_id, "artifact_id": artifact_id,
            "status": value.get("status", "complete"), "result_hash": result_hash(value),
            "replayed_from_execution_id": origin["execution_id"],
            "original_engine": origin["engine"],
            **{key: event.get(key) for key in ("worker_index", "gpu_id", "prompt_tokens",
               "output_tokens", "finish_reason", "stop_reason", "generation")},
        })

    def record(self, task_id, value, *, status="complete", artifact_id=None):
        # Fake generators used by CPU tests never claim a real engine execution.
        if self.current_result is None:
            return
        event = self.current_result
        if event["task_id"] != task_id or event["worker_index"] not in self.engines:
            raise RuntimeError(f"Qwen result has no matching engine provenance: {task_id}")
        self._append({
            "event": "result", "task_id": task_id, "status": status,
            "artifact_id": artifact_id or task_id,
            "result_hash": result_hash(value), "worker_index": event["worker_index"],
            "gpu_id": event["gpu_id"], "prompt_tokens": event["prompt_tokens"],
            "output_tokens": event["output_tokens"],
            "finish_reason": event.get("finish_reason"),
            "stop_reason": event.get("stop_reason"),
            "generation": event.get("generation"),
        })
