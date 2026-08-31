from __future__ import annotations

import numpy as np

from .config import EncoderConfig


class FeatureError(RuntimeError):
    pass


def _load_bge_runtime(settings: EncoderConfig):
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
    return torch, tokenizer, model, device


class BGETextEncoder:
    """Load one local BGE runtime and reuse it across representation branches."""

    def __init__(self, settings: EncoderConfig):
        self.settings = settings
        self._torch, self.tokenizer, self.model, self.device = _load_bge_runtime(settings)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.model(**encoded).last_hidden_state[:, 0]
        hidden = self._torch.nn.functional.normalize(hidden, p=2, dim=1)
        return hidden.cpu().numpy().astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.settings.batch_size):
                batches.append(
                    self._encode_batch(texts[start : start + self.settings.batch_size])
                )
        matrix = (
            np.concatenate(batches, axis=0)
            if batches
            else np.empty((0, self.settings.embedding_dim), dtype=np.float32)
        )
        expected_shape = (len(texts), self.settings.embedding_dim)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise FeatureError(f"invalid BGE embedding matrix: {matrix.shape}")
        return matrix
