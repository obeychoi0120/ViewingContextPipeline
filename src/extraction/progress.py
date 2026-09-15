"""Request-based progress with a bounded, wall-clock throughput window."""

from __future__ import annotations

from collections import deque
import time
import threading

from tqdm import tqdm


class RecentThroughput:
    WINDOW_SECONDS = 180
    WARMUP_SECONDS = 60
    MIN_COMPLETIONS = 8

    def __init__(self, *, clock=None):
        self.clock = clock or time.monotonic
        self.started = None
        self.events = deque()

    def start(self):
        self.started = self.clock()
        self.events.clear()

    def complete(self, output_tokens=0):
        if self.started is not None:
            self.events.append((self.clock(), output_tokens))

    def snapshot(self):
        now = self.clock()
        elapsed = 0 if self.started is None else now - self.started
        cutoff = now - self.WINDOW_SECONDS
        while self.events and self.events[0][0] <= cutoff:
            self.events.popleft()
        seconds = min(elapsed, self.WINDOW_SECONDS)
        return {
            "phase": "initializing" if self.started is None else "running",
            "inference_elapsed": elapsed,
            "requests_per_second": len(self.events) / seconds if seconds > 0 else 0,
            "output_tokens_per_second": (
                sum(tokens for _, tokens in self.events) / seconds if seconds > 0 else 0
            ),
            "eta_ready": elapsed >= self.WARMUP_SECONDS and len(self.events) >= self.MIN_COMPLETIONS,
        }


class InferenceProgress:
    """Refresh a consistent step-wide completion/rate/ETA snapshot every five seconds."""

    REFRESH_SECONDS = 5.0

    def __init__(self, *, total, desc, unit, reused=0, empty=0, progress_factory=tqdm,
                 clock=None):
        self.bar = progress_factory(
            total=total, desc=desc, unit=unit, mininterval=self.REFRESH_SECONDS,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} {unit} [{elapsed}{postfix}]",
        )
        self.fp = self.bar.fp
        self.total = total
        self.discovered = total or 0
        self.unit = unit
        self.reused = reused
        self.empty = empty
        self.success = self.failed = 0
        self.stats = {"phase": "initializing", "inflight": 0}
        self._clock = clock or time.monotonic
        self._started = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ticker = None

    def __enter__(self):
        self.bar.__enter__()
        self._started = self._clock()
        self._render(refresh=True)
        self._ticker = threading.Thread(
            target=self._refresh_loop, name="inference-progress", daemon=True,
        )
        self._ticker.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join()
        if exc_type is not None:
            self.stats["phase"] = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "error"
        self._render(refresh=True)
        return self.bar.__exit__(exc_type, exc, tb)

    def _refresh_loop(self):
        deadline = self._started + self.REFRESH_SECONDS
        while not self._stop.wait(max(0, deadline - self._clock())):
            self._render(refresh=True)
            # Keep the cadence anchored to the step start; skip missed ticks if
            # writing to the terminal itself took longer than one interval.
            elapsed = max(0, self._clock() - self._started)
            deadline = self._started + (int(elapsed / self.REFRESH_SECONDS) + 1) * self.REFRESH_SECONDS

    def discover(self, count, *, reused=0):
        with self._lock:
            self.discovered += count
            self.reused += reused

    def finish_discovery(self):
        with self._lock:
            self.total = self.discovered
            self.bar.total = self.total

    def complete(self, *, failed=False):
        with self._lock:
            if failed:
                self.failed += 1
            else:
                self.success += 1
        # Counts are collected immediately; only the timer paints them.

    def update_stats(self, stats):
        with self._lock:
            # Backend request statistics may reset for each content or retry
            # batch. They must not reset the step's displayed rate or ETA.
            self.stats = dict(stats)

    def _render(self, *, refresh):
        with self._lock, tqdm.get_lock():
            completed = self.success + self.failed
            remaining = None if self.total is None else max(0, self.total - completed)
            elapsed = max(0, self._clock() - self._started) if self._started is not None else 0
            rate = completed / elapsed if elapsed > 0 else 0
            phase = self.stats.get("phase")
            if phase in {"error", "interrupted"}:
                eta = "--"
            elif remaining == 0:
                eta = "00:00"
            elif remaining is not None and rate > 0:
                eta = tqdm.format_interval(remaining / rate)
            else:
                eta = "estimating"
            fields = f"ETA={eta} success={self.success} failed={self.failed}"
            if self.unit == "scene":
                scene_rate = f"{rate:.2f}" if elapsed > 0 else "--"
                fields += f" scene/s={scene_rate}"
            self.bar.n = completed
            self.bar.set_postfix_str(fields, refresh=refresh)
