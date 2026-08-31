from __future__ import annotations

import _thread
import threading
from time import monotonic, sleep

from PIL import Image
import pytest

from extraction.backends.gemini_workers import GeminiWorkerPool
from extraction.backends.qwen_workers import QwenGenerationTask


def test_gemini_pool_completes_out_of_order_and_captures_errors(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8)).save(image_path)
    release_slow = threading.Event()
    factory_threads: set[int] = set()

    class Backend:
        model_id = "fake-gemini"

        def generate(self, images, prompt, max_new_tokens):
            assert len(images) == 1
            assert max_new_tokens == 32
            if prompt == "slow":
                assert release_slow.wait(timeout=2)
            elif prompt == "fast":
                release_slow.set()
            else:
                raise RuntimeError("api unavailable")
            return f"result:{prompt}"

    def factory():
        factory_threads.add(threading.get_ident())
        return Backend()

    pool = GeminiWorkerPool(
        3,
        project_id="project",
        location="global",
        model_id="gemini",
        max_output_tokens=32,
        backend_factory=factory,
    )
    tasks = [
        QwenGenerationTask(name, (str(image_path),), name, 32)
        for name in ("slow", "fast", "error")
    ]
    completion_order: list[str] = []
    outcomes = pool.generate(
        tasks,
        lambda outcome: completion_order.append(outcome.task_id),
    )

    assert completion_order.index("fast") < completion_order.index("slow")
    assert outcomes["slow"].text == "result:slow"
    assert outcomes["fast"].text == "result:fast"
    assert outcomes["error"].text == ""
    assert "RuntimeError: api unavailable" in str(outcomes["error"].error)
    assert len(factory_threads) >= 2


def test_gemini_pool_rejects_duplicate_task_ids() -> None:
    pool = GeminiWorkerPool(
        1,
        project_id="project",
        location="global",
        model_id="gemini",
        backend_factory=lambda: object(),
    )
    task = QwenGenerationTask("same", (), "prompt", 8)
    try:
        pool.generate([task, task])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate task ids must fail")


def test_gemini_pool_propagates_ctrl_c_without_waiting_for_http_call() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend:
        def generate(self, images, prompt, max_new_tokens):
            started.set()
            release.wait(timeout=5)
            return "late result"

    pool = GeminiWorkerPool(
        1,
        project_id="project",
        location="global",
        model_id="gemini",
        backend_factory=BlockingBackend,
    )

    def interrupt_after_start() -> None:
        assert started.wait(timeout=1)
        sleep(0.1)
        _thread.interrupt_main()

    threading.Thread(target=interrupt_after_start, daemon=True).start()
    began = monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            pool.generate([QwenGenerationTask("blocked", (), "prompt", 8)])
        assert monotonic() - began < 1.0
    finally:
        release.set()
