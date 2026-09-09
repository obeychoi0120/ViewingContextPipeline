from __future__ import annotations

import json

import pytest

from extraction.qwen_runtime import QwenRuntimeLog, result_hash


def ready(log):
    log.engine_ready({"worker_index": 0, "gpu_id": "2", "backend": "vllm",
                      "model_path": "model", "versions": {"vllm": "0.28.0"}})
    log.current_result = {"task_id": "video:0", "worker_index": 0, "gpu_id": "2",
                          "prompt_tokens": 100, "output_tokens": 10}


def test_hash_and_mixed_run_provenance(tmp_path):
    log = QwenRuntimeLog(tmp_path, "scenes")
    value = {"scene_idx": 1, "description": "한글"}
    assert log.classification("video:1", value) == "legacy_unknown"
    ready(log)
    log.record("video:0", value, artifact_id="video:1")
    reread = QwenRuntimeLog(tmp_path, "scenes")
    assert reread.classification("video:1", value) == "vllm"
    assert reread.classification("video:0", value) == "legacy_unknown"
    assert reread.classification("video:1", {**value, "description": "changed"}) == "legacy_unknown"
    assert QwenRuntimeLog(tmp_path, "summary").classification("video:1", value) == "legacy_unknown"
    assert result_hash(value) == result_hash(dict(reversed(list(value.items()))))
    assert reread.execution_id != log.execution_id


@pytest.mark.parametrize("tail", [b'{"event":', b'\xe2\x82'])
def test_torn_final_append_recovered_without_losing_valid_records(tmp_path, tail):
    log = QwenRuntimeLog(tmp_path, "scenes")
    ready(log)
    original = log.path.read_bytes()
    log.path.write_bytes(original + tail)
    QwenRuntimeLog(tmp_path, "scenes")
    assert log.path.read_bytes() == original


def test_valid_last_record_without_newline_is_retained(tmp_path):
    log = QwenRuntimeLog(tmp_path, "scenes")
    ready(log)
    original = log.path.read_bytes()
    log.path.write_bytes(original.rstrip(b"\n"))
    reopened = QwenRuntimeLog(tmp_path, "scenes")
    ready(reopened)
    assert len([json.loads(line) for line in log.path.read_text().splitlines()]) == 2


@pytest.mark.parametrize("bad", [b'bad\n', b'{}\n'])
def test_corrupt_interior_record_is_a_hard_error(tmp_path, bad):
    log = QwenRuntimeLog(tmp_path, "scenes")
    ready(log)
    original = bad + log.path.read_bytes()
    log.path.write_bytes(original)
    with pytest.raises(ValueError, match="Qwen runtime"):
        QwenRuntimeLog(tmp_path, "scenes")
    assert log.path.read_bytes() == original


def test_unmatched_result_cannot_claim_engine_provenance(tmp_path):
    log = QwenRuntimeLog(tmp_path, "scenes")
    ready(log)
    with pytest.raises(RuntimeError, match="matching engine"):
        log.record("different", {})


def test_failed_result_is_not_classified_as_success(tmp_path):
    log = QwenRuntimeLog(tmp_path, "scenes")
    ready(log)
    log.record("video:0", {"error": "invalid"}, status="failed")
    assert QwenRuntimeLog(tmp_path, "scenes").classification("video:0", {"error": "invalid"}) == "legacy_unknown"


def test_history_write_failure_propagates(tmp_path, monkeypatch):
    log = QwenRuntimeLog(tmp_path, "scenes")
    ready(log)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(log.path), "open", fail)
    with pytest.raises(OSError, match="disk full"):
        log.record("video:0", {})
