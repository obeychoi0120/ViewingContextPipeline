from __future__ import annotations

import mimetypes
from pathlib import Path

from google.genai import types


def local_image_part(path: str | Path) -> types.Part:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"local image not found: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=mime_type,
    )
