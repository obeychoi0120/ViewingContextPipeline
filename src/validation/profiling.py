"""Opt-in sampled wall timings, with CUDA completion at measured boundaries.

Nested timings are inclusive; exclusive_seconds removes measured child spans.
No CUDA synchronization, device queries or file writes happen while disabled.
"""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import os
import time
import uuid

from validation.model import torch

PROFILE_VERSION = "recommendation-profile/v1"
_active = ContextVar("recommendation_profile_sample", default=None)


def is_profiling():
    return _active.get() is not None


def span(name, *, cuda=True):
    sample = _active.get()
    return sample.span(name, cuda=cuda) if sample is not None else nullcontext()


def timed(name, *, cuda=True):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with span(name, cuda=cuda):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def workload(**values):
    sample = _active.get()
    if sample is not None:
        for name, value in values.items():
            sample.workload[name] += int(value)


def checkpoint_contexts():
    """Carry the sampled collector into CUDA autograd's recomputation thread."""
    sample = _active.get()

    @contextmanager
    def recompute():
        token = _active.set(sample)
        try:
            yield
        finally:
            _active.reset(token)

    return nullcontext(), recompute()


class Sample:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = defaultdict(float)
        self.exclusive_seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.workload = defaultdict(int)
        self.stack = []

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def span(self, name, *, cuda=True):
        if cuda:
            self.sync()
        path = "/".join([frame[0] for frame in self.stack] + [name])
        frame = [name, 0.0]
        started = time.perf_counter()
        self.stack.append(frame)
        try:
            yield
        finally:
            try:
                if cuda:
                    self.sync()
            finally:
                elapsed = time.perf_counter() - started
                self.stack.pop()
                self.seconds[path] += elapsed
                self.exclusive_seconds[path] += max(0.0, elapsed - frame[1])
                self.calls[path] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed


class RunProfiler:
    def __init__(self, directory, identity, device, every):
        if type(every) is not int or every < 1:
            raise ValueError("profile_every must be a positive integer")
        self.path = directory / "profile.jsonl"
        self.identity = identity
        self.device = torch.device(device)
        self.every = every
        self.session = uuid.uuid4().hex
        self.label = (
            f"{identity['evaluation_date']} {identity['arm']} seed={identity['seed']} "
            f"device={self.device} pid={os.getpid()}"
        )
        self.emit({"kind": "start", "profile_every": every, "first_batches": 3})
        print(f"[Profile] {self.label} enabled every={every} file={self.path}", flush=True)

    def emit(self, record):
        row = {
            "schema_version": PROFILE_VERSION,
            "session_id": self.session,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self.identity,
            "device": str(self.device),
            "pid": os.getpid(),
            **record,
        }
        # Append+close each record: interruption preserves earlier samples;
        # separate sessions distinguish retries. Not part of result cache identity.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sample(self, phase, epoch=0, batch=None, examples=0, *, kind="batch"):
        if batch is not None and batch > 3 and batch % self.every:
            return nullcontext()
        return self._sample(phase, epoch, batch, examples, kind)

    @contextmanager
    def _sample(self, phase, epoch, batch, examples, kind):
        sample = Sample(self.device)
        sample.sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        token = _active.set(sample)
        started = time.perf_counter()
        status = "ok"
        try:
            yield sample
        except BaseException:
            status = "error"
            raise
        finally:
            _active.reset(token)
            # On failure avoid additional CUDA work that could hide the error.
            if status == "ok":
                sample.sync()
            elapsed = time.perf_counter() - started
            row = {
                "kind": kind,
                "phase": phase,
                "epoch": epoch,
                "batch": batch,
                "status": status,
                "examples": examples,
                "wall_seconds": elapsed,
                "examples_per_second": examples / elapsed if elapsed else 0.0,
                "seconds": dict(sample.seconds),
                "exclusive_seconds": dict(sample.exclusive_seconds),
                "calls": dict(sample.calls),
                "workload": dict(sample.workload),
            }
            if self.device.type == "cuda" and status == "ok":
                row["cuda_memory_mib"] = {
                    "allocated": torch.cuda.memory_allocated(self.device) / 1024**2,
                    "reserved": torch.cuda.memory_reserved(self.device) / 1024**2,
                    "peak_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**2,
                }
            self.emit(row)
            stages = " ".join(
                f"{key}={value * 1000:.1f}ms"
                for key, value in sample.seconds.items()
                if "/" not in key
            )
            print(
                f"[Profile] {self.label} phase={phase} epoch={epoch} batch={batch} "
                f"status={status} wall={elapsed:.3f}s {stages}",
                flush=True,
            )
            details = " ".join(
                f"{key}={value * 1000:.1f}ms" for key, value in sample.seconds.items() if "/" in key
            )
            if details:
                print(
                    f"[Profile detail] {self.label} phase={phase} epoch={epoch} "
                    f"batch={batch} {details} workload={dict(sample.workload)} "
                    f"memory_mib={row.get('cuda_memory_mib', {})}",
                    flush=True,
                )

    def phase(self, phase, epoch, seconds, examples):
        self.emit(
            {
                "kind": "phase",
                "phase": phase,
                "epoch": epoch,
                "wall_seconds": seconds,
                "examples": examples,
                "examples_per_second": examples / seconds if seconds else 0.0,
            }
        )


def sampled(profiler, *args, **kwargs):
    return profiler.sample(*args, **kwargs) if profiler is not None else nullcontext()
