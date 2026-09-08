"""Opt-in pytest observer for comparing refactors against an archived checkout.

Load explicitly with ``-p refactor_capture``; it is never part of normal tests.
VCP_CAPTURE_OUTPUT selects the snapshot and VCP_CAPTURE_SOURCE_ROOT verifies
which implementation is being tested. See docs/e2e-refactoring.md.
"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

RESULTS = {}


def pytest_sessionstart(session):
    expected_source = os.environ.get("VCP_CAPTURE_SOURCE_ROOT")
    if expected_source:
        import pipeline_runtime

        assert (
            Path(pipeline_runtime.__file__)
            .resolve()
            .is_relative_to(Path(expected_source).resolve())
        )
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    events = []
    root = item.funcargs.get("tmp_path")
    replacements = [(str(item.config.rootpath), "<repo>")]
    if root is not None:
        replacements.insert(0, (str(root), "<tmp>"))

    def norm(value):
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            for old, new in replacements:
                value = (
                    value.replace(old.replace("\\", "\\\\"), new)
                    .replace(old, new)
                    .replace(old.replace("\\", "/"), new)
                )
            return value.replace("\\", "/")
        if isinstance(value, dict):
            return {
                str(k): (
                    "<fixture-mtime>"
                    if k == "source_mtime_ns"
                    else "<elapsed>"
                    if k == "elapsed_seconds"
                    else norm(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        return value

    def trace(frame, event, arg):
        name = frame.f_code.co_name
        module = frame.f_globals.get("__name__")
        if event == "call":
            local = frame.f_locals
            if name == "generate":
                tasks = local.get("tasks")
                if isinstance(tasks, list) and tasks and dataclasses.is_dataclass(tasks[0]):
                    events.append(["generate", norm(tasks)])
            elif name == "encode":
                texts = local.get("texts", local.get("_texts"))
                if isinstance(texts, list) and all(isinstance(value, str) for value in texts):
                    events.append(["encode", norm(texts)])
            elif name in ("atomic_write_json", "atomic_write_jsonl") and module == "artifact_io":
                value = local.get("value", local.get("rows"))
                if isinstance(value, (dict, list)):
                    events.append([name, norm(local["path"]), norm(value), local["durable"]])
        elif (
            event == "return" and name == "main" and module in ("extraction.cli", "validation.cli")
        ):
            events.append(["cli_return", module, arg])

    sys.setprofile(trace)
    try:
        yield
    finally:
        sys.setprofile(None)
    files = {}
    if root is not None:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".npz", ".pt", ".png"):
                continue
            key = path.relative_to(root).as_posix()
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as arrays:
                        value = {
                            k: [
                                str(arrays[k].dtype),
                                list(arrays[k].shape),
                                hashlib.sha256(arrays[k].tobytes()).hexdigest(),
                            ]
                            for k in arrays.files
                        }
                elif path.suffix == ".pt":
                    import torch

                    doc = torch.load(path, map_location="cpu", weights_only=True)
                    value = {
                        "metadata": norm(doc["metadata"]),
                        "state_dict": {
                            k: [
                                str(v.dtype),
                                list(v.shape),
                                hashlib.sha256(v.numpy().tobytes()).hexdigest(),
                            ]
                            for k, v in doc["state_dict"].items()
                        },
                    }
                elif path.suffix in (".json", ".jsonl"):
                    raw = path.read_text(encoding="utf-8")
                    try:
                        value = norm(
                            json.loads(raw)
                            if path.suffix == ".json"
                            else [json.loads(line) for line in raw.splitlines() if line.strip()]
                        )
                    except ValueError:
                        value = norm(raw)
            except Exception:
                # Corrupt caches and mock checkpoints intentionally use raw file hashes.
                pass
            files[key] = value
    assert item.name not in RESULTS, f"Duplicate snapshot case: {item.name}"
    RESULTS[item.name] = {"events": events, "files": files}


def pytest_sessionfinish(session, exitstatus):
    Path(os.environ["VCP_CAPTURE_OUTPUT"]).write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def compare_snapshots(before_path: Path, after_path: Path) -> dict:
    """Return auditable per-case hashes, plus the observation counts."""
    from collections import Counter

    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    cases = []
    for name in sorted(before.keys() | after.keys()):
        left, right = before.get(name), after.get(name)
        cases.append(
            {
                "case": name,
                "equal": left == right,
                "before_sha256": digest(left),
                "after_sha256": digest(right),
                "event_count": len(right["events"]) if right else 0,
                "file_count": len(right["files"]) if right else 0,
            }
        )
    return {
        "equal": before == after,
        "case_count": len(cases),
        "event_counts": dict(
            Counter(event[0] for case in after.values() for event in case["events"])
        ),
        "artifact_counts": dict(
            Counter(Path(path).suffix for case in after.values() for path in case["files"])
        ),
        "tensor_count": sum(
            len(value.get("state_dict", {}))
            for case in after.values()
            for value in case["files"].values()
            if isinstance(value, dict)
        ),
        "cases": cases,
    }


if __name__ == "__main__":
    report = compare_snapshots(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["equal"] else 1)
