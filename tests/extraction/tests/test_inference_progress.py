from __future__ import annotations

import io

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


def test_progress_uses_only_pending_requests_and_separates_failures():
    progress, stream = make_progress(total=100, reused=999)
    with progress:
        assert "ETA=initializing" in progress.bar.postfix
        progress.update_stats({"phase": "running", "eta_ready": False,
                               "requests_per_second": 10, "inflight": 64})
        assert "ETA=estimating" in progress.bar.postfix
        progress.complete()
        progress.complete(failed=True)
        assert progress.bar.n == 2 and progress.bar.total == 100
        progress.update_stats({"phase": "running", "eta_ready": True,
                               "requests_per_second": 2, "inflight": 64})
        assert "ETA=00:49" in progress.bar.postfix
        assert "success=1 failed=1 reused=999" in progress.bar.postfix
    # The standard tqdm instantaneous ETA/rate must not appear alongside ours.
    assert "scene/s" not in stream.getvalue()
    assert "remaining" not in progress.bar.bar_format


def test_finished_cached_and_interrupted_progress():
    progress, _ = make_progress(total=0, reused=99)
    with progress:
        assert "phase=finished" in progress.bar.postfix
        assert "ETA=00:00" in progress.bar.postfix
    progress, _ = make_progress(total=1)
    with progress:
        progress.complete(failed=True)
    assert "phase=finished" in progress.bar.postfix
    assert "success=0 failed=1" in progress.bar.postfix
    progress, _ = make_progress(total=2)
    with pytest.raises(KeyboardInterrupt), progress:
        progress.complete()
        raise KeyboardInterrupt
    assert "phase=interrupted" in progress.bar.postfix
    assert "ETA=--" in progress.bar.postfix
