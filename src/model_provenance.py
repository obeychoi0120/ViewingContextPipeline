"""Identify local checkpoints without rereading multi-gigabyte weights each stage."""

from __future__ import annotations

import hashlib
from pathlib import Path

from extraction.recovery import fingerprint


def local_model_identity(path: Path) -> dict:
    path = path.resolve()
    files = []
    for file in sorted(path.glob("*")):
        if not file.is_file() or file.suffix not in {
            ".json",
            ".txt",
            ".model",
            ".safetensors",
            ".bin",
        }:
            continue
        stat = file.stat()
        record = {"name": file.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if file.suffix in {".json", ".txt"}:
            record["content_hash"] = hashlib.sha256(file.read_bytes()).hexdigest()
        files.append(record)
    return {
        "path": str(path),
        "files_signature": fingerprint(files),
        "signature_policy": "config/tokenizer text hashes; weight filenames, sizes and mtimes",
    }
