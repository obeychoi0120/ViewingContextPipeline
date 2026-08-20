from __future__ import annotations

from pathlib import Path


def custom_output_root(base: str | Path) -> Path:
    root = Path(base)
    if root.name == "custom" or root.name.lower().startswith("microlens"):
        return root
    return root / "custom"
