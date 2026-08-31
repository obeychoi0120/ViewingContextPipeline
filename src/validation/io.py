from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from artifact_io import (
    atomic_write_json as _atomic_write_json,
    atomic_write_jsonl as _atomic_write_jsonl,
)

def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write_json(path, value, durable=True)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_jsonl(path, rows, durable=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

