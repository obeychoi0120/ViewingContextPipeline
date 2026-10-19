from __future__ import annotations

class CohortError(RuntimeError):
    pass


def normalize_item_id(value: object) -> str:
    text = str(value).strip()
    if not text.isdigit() or int(text) <= 0:
        raise CohortError(f"invalid item id: {value!r}")
    return str(int(text))


def content_id_for_item(item_id: str) -> str:
    return f"microlens_100k_{int(item_id):05d}"
