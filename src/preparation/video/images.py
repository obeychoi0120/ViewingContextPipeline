from __future__ import annotations

from pathlib import Path

from PIL import Image


def verified_image_size(path: str | Path) -> tuple[int, int] | None:
    """Return dimensions only when Pillow verifies the complete image payload."""

    target = Path(path)
    if not target.is_file():
        return None
    try:
        with Image.open(target) as image:
            size = image.size
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return None
    return size
