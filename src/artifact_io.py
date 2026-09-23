from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    durable: bool,
    sort_keys: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=sort_keys)
            handle.write("\n")
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        temporary.replace(target)
        if durable:
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    durable: bool,
    sort_keys: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys,
                                        separators=(",", ":")) + "\n")
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        temporary.replace(target)
        if durable:
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
