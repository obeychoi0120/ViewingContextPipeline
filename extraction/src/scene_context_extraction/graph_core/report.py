from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.video_data_collection.raw_pipeline import modality_dirname


def write_viewing_context_report(
    output_root: str | Path,
    *,
    multimodal: bool,
    mode: str,
    source: str,
    payload: dict[str, Any],
) -> Path:
    path = (
        Path(output_root)
        / "reports"
        / "viewing_context"
        / modality_dirname(multimodal)
        / mode
        / f"scene_context_graph_{source}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
