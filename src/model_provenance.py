"""Identify local checkpoints without rereading multi-gigabyte weights each stage."""

from __future__ import annotations

import hashlib
from pathlib import Path

from artifact_io import fingerprint


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


def canonical(value):
    if isinstance(value, dict):
        return {
            k: canonical(v)
            for k, v in value.items()
            if k
            not in {
                "run_id",
                "source_run_id",
                "prompt_path",
                "source_path",
                "scene_path",
                "path",
                "created_at",
                "generated_at",
                "reused_from",
            }
        }
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def without_provenance_arm(value):
    """Remove the retired duplicate identity, including nested Scene provenance."""
    if isinstance(value, dict):
        return {key: without_provenance_arm(child) for key, child in value.items() if key != "arm"}
    if isinstance(value, list):
        return [without_provenance_arm(child) for child in value]
    return value
