from __future__ import annotations

import numpy as np

import validation.features as features
from validation.config import EncoderConfig


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


def test_bge_encoder_reuses_one_loaded_runtime(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "bge"
    model_path.mkdir()
    settings = EncoderConfig.model_validate({
        "model_path": model_path,
        "embedding_dim": 1024,
        "max_length": 512,
        "batch_size": 2,
    })
    loads = []

    def fake_load(selected):
        loads.append(selected.model_path)
        return _FakeTorch(), object(), object(), "cpu"

    monkeypatch.setattr(features, "_load_bge_runtime", fake_load)
    encoder = features.BGETextEncoder(settings)
    encoded_batches = []

    def fake_encode_batch(texts):
        encoded_batches.append(list(texts))
        return np.ones((len(texts), settings.embedding_dim), dtype=np.float32)

    monkeypatch.setattr(encoder, "_encode_batch", fake_encode_batch)

    assert encoder.encode(["a", "b", "c"]).shape == (3, 1024)
    assert encoder.encode(["d"]).shape == (1, 1024)
    assert loads == [model_path]
    assert encoded_batches == [["a", "b"], ["c"], ["d"]]
