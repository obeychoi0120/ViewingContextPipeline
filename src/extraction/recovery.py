"""Input fingerprints and generation settings; no recovery state is persisted."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def penalty_schedule(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(type(x) not in (int, float) or not math.isfinite(x)
                         or not 1 <= x <= 2 for x in values):
        raise ValueError("repetition penalty must be a number or non-empty list in [1, 2]")
    return [float(x) for x in values]


def generation_key(task, identity, penalties):
    """Hash task/provenance only; prepared image bytes are trusted until an explicit force run."""
    return fingerprint({"task": asdict(task),
                        "images": "prepared-assets" if task.image_paths else [],
                        "identity": identity, "penalties": penalty_schedule(penalties),
                        "recovery_version": 2})
