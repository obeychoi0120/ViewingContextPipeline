"""Request-based progress with a bounded, wall-clock throughput window."""

from __future__ import annotations

from collections import deque
import time

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
    """Keep cached work out of the bar and expose only the recent-window ETA."""

    def __init__(self, *, total, desc, unit, reused=0, empty=0, progress_factory=tqdm):
        self.bar = progress_factory(
            total=total, desc=desc, unit=unit, mininterval=1, dynamic_ncols=True,
            bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} {unit} [{elapsed}{postfix}]",
        )
        self.fp = self.bar.fp
        self.total = total
        self.reused = reused
        self.empty = empty
        self.success = self.failed = 0
        self.stats = {"phase": "initializing", "inflight": 0}

    def __enter__(self):
        self.bar.__enter__()
        self._render(refresh=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.stats["phase"] = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "error"
        self._render(refresh=True)
        return self.bar.__exit__(exc_type, exc, tb)

    def complete(self, *, failed=False):
        if failed:
            self.failed += 1
        else:
            self.success += 1
        self._render(refresh=False)
        self.bar.update(1)

    def update_stats(self, stats):
        self.stats = dict(stats)
        self._render(refresh=True)

    def _render(self, *, refresh):
        remaining = self.total - self.success - self.failed
        phase = self.stats.get("phase", "initializing")
        rate = self.stats.get("requests_per_second", 0)
        if phase in {"error", "interrupted"}:
            eta = "--"
        elif remaining == 0:
            phase, eta = "finished", "00:00"
        elif phase == "initializing":
            eta = "initializing"
        elif self.stats.get("eta_ready") and rate > 0:
            eta = tqdm.format_interval(remaining / rate)
        else:
            eta = "estimating"
        fields = f"ETA={eta} success={self.success} failed={self.failed}"
        if phase != "initializing" and "output_tokens_per_second" in self.stats:
            fields += f" tok/s={self.stats['output_tokens_per_second']:.1f}"
        self.bar.set_postfix_str(fields, refresh=refresh)
