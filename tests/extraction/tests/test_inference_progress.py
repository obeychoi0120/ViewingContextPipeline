from __future__ import annotations

import io
import queue

import pytest
from tqdm import tqdm

from extraction.progress import InferenceProgress, RecentThroughput


def test_throughput_excludes_initialization_and_requires_warmup():
    now = [0.0]
    rate = RecentThroughput(clock=lambda: now[0])
    now[0] = 120
    rate.complete(999)
    assert rate.snapshot()["phase"] == "initializing"
    rate.start()
    for second in range(1, 9):
        now[0] = 120 + second
        rate.complete(10)
    assert not rate.snapshot()["eta_ready"]
    now[0] = 180
    stats = rate.snapshot()
    assert stats["eta_ready"]
    assert stats["inference_elapsed"] == 60
    assert stats["requests_per_second"] == pytest.approx(8 / 60)
    assert stats["output_tokens_per_second"] == pytest.approx(80 / 60)


def test_wall_clock_window_handles_bursts_and_no_completion_periods():
    now = [0.0]
    rate = RecentThroughput(clock=lambda: now[0])
    rate.start()
    # Completions arriving together must not imply near-infinite throughput.
    now[0] = 30
    for _ in range(60):
        rate.complete(5)
    now[0] = 60
    assert rate.snapshot()["requests_per_second"] == 1
    now[0] = 180
    assert rate.snapshot()["requests_per_second"] == pytest.approx(1 / 3)
    now[0] = 210
    assert rate.snapshot()["requests_per_second"] == 0
    assert not rate.snapshot()["eta_ready"]
    assert not rate.events
    for _ in range(18):
        rate.complete(10)
    now[0] = 240
    assert rate.snapshot()["requests_per_second"] == pytest.approx(18 / 180)


def make_progress(**kwargs):
    stream = io.StringIO()
    progress = InferenceProgress(
        desc="Graph", unit="scene",
        progress_factory=lambda **options: tqdm(file=stream, **options), **kwargs,
    )
    return progress, stream


def test_progress_uses_only_pending_scenes_and_separates_failures():
    now = [0.0]
    progress, stream = make_progress(total=100, reused=999, clock=lambda: now[0])
    with progress:
        initial = stream.getvalue()
        assert "scene/s=--" in progress.bar.postfix
        progress.update_stats({"phase": "running", "requests_per_second": 1000,
                               "completed": 500, "output_tokens_per_second": 512})
        progress.complete()
        progress.complete(failed=True)
        # Model retries and cache hits do not count as completed Scenes.
        assert progress.success == progress.failed == 1
        assert progress.bar.n == 0
        assert stream.getvalue() == initial
        now[0] = 10
        progress._render(refresh=True)
        assert progress.bar.n == 2 and progress.bar.total == 100
        assert "ETA=08:10" in progress.bar.postfix
        assert "success=1 failed=1" in progress.bar.postfix
        assert "scene/s=0.20" in progress.bar.postfix
        for field in ("tok/s=", "phase=", "reused=", "inflight=", "requests/s=", "init=", "infer="):
            assert field not in progress.bar.postfix
        rendered = str(progress.bar)
        assert "2/100 scene" in rendered
        assert len(rendered.split("|")[1]) == 20
    assert "scene/s=0.20" in stream.getvalue()
    assert "rate_fmt" not in progress.bar.bar_format
    assert "remaining" not in progress.bar.bar_format
    assert not progress._ticker.is_alive()


def test_scene_rate_does_not_reset_when_gemini_starts_next_content():
    now = [0.0]
    progress, _ = make_progress(total=115947, clock=lambda: now[0])
    with progress:
        for _ in range(82):
            progress.complete()
        now[0] = 91
        progress.update_stats({"phase": "running", "requests_per_second": 0,
                               "completed": 0, "inference_elapsed": 0, "eta_ready": False})
        progress._render(refresh=True)
        assert progress.bar.n == 82
        assert "scene/s=0.90" in progress.bar.postfix
        eta = tqdm.format_interval((115947 - 82) / (82 / 91))
        assert f"ETA={eta}" in progress.bar.postfix
        assert "estimating" not in progress.bar.postfix


def test_timer_recalculates_every_five_seconds_even_without_completions():
    now = [0.0]
    progress, stream = make_progress(total=10, clock=lambda: now[0])

    class TickControl:
        def __init__(self):
            self.commands = queue.Queue()
            self.waits = queue.Queue()

        def wait(self, timeout):
            self.waits.put(timeout)
            if self.commands.get(timeout=2) == "stop":
                return True
            now[0] += timeout
            return False

        def set(self):
            self.commands.put("stop")

    ticks = TickControl()
    progress._stop = ticks
    with progress:
        assert ticks.waits.get(timeout=2) == 5
        initial = stream.getvalue()
        progress.complete()
        progress.complete(failed=True)
        progress.update_stats({"phase": "running", "requests_per_second": 999})
        assert stream.getvalue() == initial
        ticks.commands.put("tick")
        assert ticks.waits.get(timeout=2) == 5
        assert progress.bar.n == 2
        assert "success=1 failed=1" in progress.bar.postfix
        assert "scene/s=0.40" in progress.bar.postfix
        assert "ETA=00:20" in progress.bar.postfix
        # No new requests or callbacks: time alone must update rate and ETA.
        ticks.commands.put("tick")
        assert ticks.waits.get(timeout=2) == 5
        assert progress.bar.n == 2
        assert "scene/s=0.20" in progress.bar.postfix
        assert "ETA=00:40" in progress.bar.postfix
        progress.complete()
    assert progress.bar.n == 3  # Flush the final partial interval immediately.
    assert not progress._ticker.is_alive()


def test_finished_cached_and_interrupted_progress():
    progress, _ = make_progress(total=0, reused=99)
    with progress:
        assert "ETA=00:00" in progress.bar.postfix
    progress, _ = make_progress(total=1)
    with progress:
        progress.complete(failed=True)
    assert "ETA=00:00" in progress.bar.postfix
    assert "success=0 failed=1" in progress.bar.postfix
    progress, _ = make_progress(total=2)
    with pytest.raises(KeyboardInterrupt), progress:
        progress.complete()
        raise KeyboardInterrupt
    assert progress.stats["phase"] == "interrupted"
    assert "ETA=--" in progress.bar.postfix
    assert not progress._ticker.is_alive()
