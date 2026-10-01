"""Opt-in sampled wall timings, with CUDA completion at measured boundaries.

Nested timings are inclusive; exclusive times can be derived from child spans.
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

PROFILE_VERSION = "recommendation-profile/v2"
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
    def __init__(self, device, operators=False):
        self.device = torch.device(device)
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.workload = defaultdict(int)
        self.stack = []
        self.operators = operators

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def span(self, name, *, cuda=True):
        if cuda:
            self.sync()
        path = "/".join([*self.stack, name])
        started = time.perf_counter()
        self.stack.append(name)
        try:
            marker = torch.profiler.record_function(f"profile::{path}") if self.operators else nullcontext()
            with marker:
                yield
        finally:
            try:
                if cuda:
                    self.sync()
            finally:
                elapsed = time.perf_counter() - started
                self.stack.pop()
                self.seconds[path] += elapsed
                self.calls[path] += 1


def projection_backward_summary(events):
    """Match forward projection scopes to autograd nodes by sequence number.

    Report node-inclusive CPU/device durations, not wall-time shares. Exclude
    engine wrappers to avoid counting both a wrapper and its backward node.
    """
    names = {"node_linear", "context_projection", "title_projection",
             "video_projection", "item_projection"}
    owners = {}
    for event in events:
        if event.scope == 1 or event.sequence_nr < 0 or not event.name.startswith("aten::"):
            continue
        parent = event.cpu_parent
        while parent is not None:
            if parent.name.startswith("profile::"):
                path = parent.name.removeprefix("profile::")
                if path.rsplit("/", 1)[-1] in names:
                    owners[(event.thread, event.sequence_nr)] = path
                    break
            parent = parent.cpu_parent
    groups = {}
    total_cpu = total_device = 0.0
    total_nodes = matched_nodes = 0
    for event in events:
        if event.scope != 1 or event.name.startswith("autograd::engine::"):
            continue
        total_nodes += 1
        total_cpu += event.cpu_time_total
        total_device += event.device_time_total
        path = owners.get((event.fwd_thread, event.sequence_nr))
        if path is None:
            continue
        matched_nodes += 1
        group = groups.setdefault(path, {"nodes": 0, "cpu_total_us": 0.0,
                                         "device_total_us": 0.0, "node_types": {}})
        group["nodes"] += 1
        group["cpu_total_us"] += event.cpu_time_total
        group["device_total_us"] += event.device_time_total
        group["node_types"][event.name] = group["node_types"].get(event.name, 0) + 1
    return {"projection_nodes": groups, "matched_nodes": matched_nodes,
            "all_backward_nodes": total_nodes, "all_backward_cpu_total_us": total_cpu,
            "all_backward_device_total_us": total_device,
            "projection_cpu_pct_of_backward_nodes": (
                100 * sum(g["cpu_total_us"] for g in groups.values()) / total_cpu if total_cpu else None
            ),
            "projection_device_pct_of_backward_nodes": (
                100 * sum(g["device_total_us"] for g in groups.values()) / total_device if total_device else None
            )}


class RunProfiler:
    def __init__(self, directory, identity, device, every, *, operators=False):
        if type(every) is not int or every < 1:
            raise ValueError("profile_every must be a positive integer")
        self.path = directory / "profile.jsonl"
        self.identity = identity
        self.device = torch.device(device)
        self.every = every
        self.operators = operators
        self.operator_captured = False
        self.session = uuid.uuid4().hex
        self.label = (
            f"{identity['evaluation_date']} {identity['arm']} seed={identity['seed']} "
            f"device={self.device} pid={os.getpid()}"
        )
        self.emit({"kind": "start", "profile_every": every, "first_batches": 3,
                   "operator_profiling": operators})
        print(f"[Profile] {self.label} enabled every={every} file={self.path}", flush=True)

    def emit(self, record):
        row = {
            "session_id": self.session,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        if record["kind"] == "start":
            row = {
                "schema_version": PROFILE_VERSION,
                **self.identity,
                "device": str(self.device),
                "pid": os.getpid(),
                **row,
            }
        # Keep the measurement rows small; start holds session-wide context.
        row = {key: value for key, value in row.items() if value is not None and value != {}}
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
        # One warmed training batch per combination/session, never every sample.
        capture = (self.operators and not self.operator_captured and kind == "batch"
                   and phase in {"selection", "refit"} and batch is not None and batch >= 3)
        operator_profile = None
        if capture:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            operator_profile = torch.profiler.profile(activities=activities, record_shapes=True)
            self.operator_captured = True
        sample = Sample(self.device, operators=capture)
        sample.sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        token = _active.set(sample)
        started = time.perf_counter()
        status = "ok"
        try:
            with operator_profile if capture else nullcontext():
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
                "seconds": dict(sample.seconds),
                "calls": dict(sample.calls),
                "workload": dict(sample.workload),
            }
            if capture:
                row["operator_profiled"] = True
                if status == "ok":
                    folder = self.path.parent / "profile_traces" / self.session
                    folder.mkdir(parents=True, exist_ok=True)
                    name = f"{phase}_epoch_{epoch}_batch_{batch}"
                    trace = folder / f"{name}.trace.json"
                    summary_path = folder / f"{name}.operators.json"
                    operator_profile.export_chrome_trace(str(trace))
                    events = operator_profile.events()
                    summary = {
                        "schema_version": "recommendation-operators/v1",
                        "session_id": self.session, "phase": phase, "epoch": epoch,
                        "batch": batch, "device": str(self.device),
                        "projection_backward": projection_backward_summary(events),
                        "operators": [{
                            "name": e.key, "input_shapes": e.input_shapes, "calls": e.count,
                            "cpu_total_us": e.cpu_time_total, "self_cpu_total_us": e.self_cpu_time_total,
                            "device_total_us": e.device_time_total,
                            "self_device_total_us": e.self_device_time_total,
                        } for e in operator_profile.key_averages(group_by_input_shape=True)],
                    }
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
                    row["operator_trace"] = str(trace.relative_to(self.path.parent))
                    row["operator_summary"] = str(summary_path.relative_to(self.path.parent))
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
            }
        )


def sampled(profiler, *args, **kwargs):
    return profiler.sample(*args, **kwargs) if profiler is not None else nullcontext()
