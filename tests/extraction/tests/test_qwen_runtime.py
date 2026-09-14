from __future__ import annotations

from extraction.qwen_runtime import QwenRuntime, result_hash


def test_result_hash_is_independent_of_key_order():
    value = {"scene_idx": 1, "description": "한글"}
    assert result_hash(value) == result_hash(dict(reversed(list(value.items()))))
    assert result_hash(value) != result_hash({**value, "description": "changed"})


def test_checkpoint_metadata_is_kept_without_creating_runtime_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = QwenRuntime()
    assert runtime.checkpoint_origin() is None
    engine = {"worker_index": 0, "gpu_id": "2", "backend": "vllm",
              "model_path": "model", "versions": {"vllm": "0.28.0"}}
    event = {"task_id": "video:0", "worker_index": 0, "gpu_id": "2",
             "prompt_tokens": 100, "output_tokens": 10}
    runtime.engine_ready(engine)
    runtime.current_result = event
    assert runtime.checkpoint_origin() == {
        "execution_id": runtime.execution_id, "result": event, "engine": engine,
    }
    assert QwenRuntime().execution_id != runtime.execution_id
    assert list(tmp_path.iterdir()) == []
