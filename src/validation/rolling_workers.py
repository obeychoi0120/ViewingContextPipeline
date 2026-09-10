"""Spawn isolated workers for independent rolling combinations on assigned GPUs."""

from __future__ import annotations

import multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout
import io
from queue import Empty
import signal
import time
import traceback


class WorkerOutput(io.TextIOBase):
    """Send whole lines to the parent, which owns the terminal and progress bar."""

    def __init__(self, results):
        self.results = results
        self.buffered = ""

    def write(self, value):
        self.buffered += value
        while "\n" in self.buffered:
            line, self.buffered = self.buffered.split("\n", 1)
            self.results.put(("log", line))
        return len(value)

    def flush(self):
        if self.buffered:
            self.results.put(("log", self.buffered))
            self.buffered = ""


def combination_worker(context, device_name, jobs, results):
    # The parent owns Ctrl-C and terminates/joins all its children on interruption.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    output = WorkerOutput(results)
    with redirect_stdout(output), redirect_stderr(output):
        try:
            _consume_combinations(context, device_name, jobs, results)
        finally:
            output.flush()


def _consume_combinations(context, device_name, jobs, results):
    identity = None
    try:
        from validation.model import torch
        from validation.rolling_data import EventTable, iter_jsonl
        from validation.rolling_recommendation import prepare_split, run_combination
        from validation.steps import validation_config

        torch.set_num_threads(1)
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        config = validation_config(context)
        table = EventTable(iter_jsonl(context.cohort_dir / "events.jsonl"))
        previous_date, prepared = None, None
        while (job := jobs.get()) is not None:
            split, identity, branch = job
            if split["evaluation_date"] != previous_date:
                prepared = prepare_split(table, split)
                previous_date = split["evaluation_date"]
            run_combination(context, config, table, split, identity, branch, prepared, device)
            results.put(("complete", identity))
    except BaseException:
        results.put(("error", f"device={device_name} combination={identity}\n{traceback.format_exc()}"))


def run_parallel(context, jobs, devices, progress):
    # CUDA must be initialized in spawned children, never inherited through fork.
    runtime = mp.get_context("spawn")
    pending, results = runtime.Queue(), runtime.Queue()
    processes = []
    completed = 0
    last_refresh = time.monotonic()
    try:
        for job in jobs:
            pending.put(job)
        for device in devices[:len(jobs)]:
            pending.put(None)
            process = runtime.Process(
                target=combination_worker, args=(context, device, pending, results)
            )
            process.start()
            processes.append(process)
        while completed < len(jobs):
            try:
                status, payload = results.get(timeout=1)
            except Empty:
                if all(process.exitcode is not None for process in processes):
                    raise RuntimeError("rolling workers exited before reporting all combinations")
            else:
                if status == "error":
                    raise RuntimeError(f"rolling worker failed: {payload}")
                if status == "log":
                    progress.write(payload, file=progress.fp)
                    last_refresh = time.monotonic()
                elif status == "complete":
                    completed += 1
                    progress.update(1)
            # Keep elapsed time visible even before the first combination completes.
            # Redirected logs get sparse heartbeats instead of one line per second.
            interval = 1 if progress.fp.isatty() else 30
            if time.monotonic() - last_refresh >= interval:
                progress.refresh()
                last_refresh = time.monotonic()
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(
                        f"rolling worker pid={process.pid} exited with code {process.exitcode}"
                    )
        return completed
    finally:
        # Completed files remain reusable; incomplete combinations have no completion marker.
        for process in processes:
            if completed < len(jobs) and process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            process.close()
        pending.cancel_join_thread()
        pending.close()
        results.close()
