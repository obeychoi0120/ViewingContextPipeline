"""Separate frozen title/video features and their explicit availability contract."""

import numpy as np

COMPOSITION_POLICY = "title-video-separate-bge/v1"
ARRAYS = {"title_values", "video_values", "title_available", "video_available"}


def validate_arrays(arrays, count, dim, docs=None):
    if set(arrays) != ARRAYS:
        raise ValueError("legacy/invalid Text embeddings; rerun embed-representations")
    for component in ("title", "video"):
        values = arrays[f"{component}_values"]
        available = arrays[f"{component}_available"]
        if (
            values.shape != (count, dim)
            or values.dtype != np.float32
            or available.shape != (count,)
            or available.dtype != np.bool_
            or not np.isfinite(values).all()
            or np.any(values[~available] != 0)
        ):
            raise ValueError(f"invalid {component} embedding array or availability mask")
        if docs is not None:
            expected = np.array([bool(d[f"{component}_text"].strip()) for d in docs])
            if not np.array_equal(expected, available):
                raise ValueError(f"{component} availability does not match source")


def encode_components(docs, dim, encoder, cache):
    texts = list(
        dict.fromkeys(
            d[f"{c}_text"].strip()
            for d in docs
            for c in ("title", "video")
            if d[f"{c}_text"].strip()
        )
    )
    missing = [t for t in texts if t not in cache]
    if missing:
        encoded = np.asarray(encoder.encode(missing), dtype=np.float32)
        if encoded.shape != (len(missing), dim) or not np.isfinite(encoded).all():
            raise ValueError("invalid BGE component features")
        flags = getattr(encoder, "last_truncated_flags", [False] * len(missing))
        for text, vector, truncated in zip(missing, encoded, flags, strict=True):
            cache[text] = (vector, bool(truncated))
    arrays, truncation = {}, {}
    for c in ("title", "video"):
        values = np.zeros((len(docs), dim), dtype=np.float32)
        available = np.array([bool(d[f"{c}_text"].strip()) for d in docs], dtype=np.bool_)
        truncated = 0
        for i, d in enumerate(docs):
            text = d[f"{c}_text"].strip()
            if text:
                values[i], flag = cache[text]
                truncated += flag
        arrays[f"{c}_values"] = values
        arrays[f"{c}_available"] = available
        truncation[c] = {"text_count": int(available.sum()), "truncated_count": truncated}
    validate_arrays(arrays, len(docs), dim, docs)
    return arrays, truncation
