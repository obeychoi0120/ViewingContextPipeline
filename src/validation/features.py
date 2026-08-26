from __future__ import annotations

import numpy as np

from .config import EncoderConfig


class FeatureError(RuntimeError):
    pass


def encode_bge_texts(settings: EncoderConfig, texts: list[str]) -> np.ndarray:
    """Encode canonical video-context text with the configured local BGE model."""
    if not settings.model_path.is_dir():
        raise FeatureError(f"local encoder path does not exist: {settings.model_path}")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise FeatureError("torch and transformers are required for BGE encoding") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(settings.model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(settings.model_path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), settings.batch_size):
            encoded = tokenizer(
                texts[start : start + settings.batch_size],
                padding=True,
                truncation=True,
                max_length=settings.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            batches.append(hidden.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(batches, axis=0)
        if batches
        else np.empty((0, settings.embedding_dim), dtype=np.float32)
    )
