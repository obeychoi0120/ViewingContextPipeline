from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from validation.model import SASRec, in_batch_loss, pad_sequences  # noqa: E402
from validation.recommendation import popularity_probabilities  # noqa: E402

pytestmark = pytest.mark.torch


def _model(*, dimension: int = 8, item_count: int = 12) -> SASRec:
    features = np.arange(item_count * 16, dtype=np.float32).reshape(item_count, 16)
    features /= float(features.max())
    return SASRec(
        item_count=item_count,
        max_length=10,
        embedding_dim=dimension,
        num_blocks=2,
        num_heads=2,
        dropout=0.0,
        arm="metadata",
        item_features=features,
    )


@pytest.mark.parametrize("arm", ["metadata", "graph", "desc"])
def test_all_item_towers_use_frozen_features_and_trainable_projection(arm) -> None:
    features = np.ones((12, 1024), dtype=np.float32)
    model = SASRec(
        item_count=12,
        max_length=10,
        embedding_dim=8,
        num_blocks=2,
        num_heads=2,
        dropout=0.0,
        arm=arm,
        item_features=features,
    )
    sequences = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
        ]
    )
    probabilities = np.full(13, 1 / 12, dtype=np.float32)
    probabilities[0] = 1.0

    assert model.encode(sequences).shape == (2, 10, 8)
    assert model.score_catalog(sequences).shape == (2, 12)
    loss = in_batch_loss(
        model,
        [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
        torch.device("cpu"),
        probabilities,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.frozen_item_features.requires_grad is False
    assert model.item_projection.weight.grad is not None
    assert model.user_mlp.fc1.weight.grad is not None
    assert model.item_mlp.activation == "relu"
    assert model.user_mlp.activation == "gelu"
    assert model.item_mlp.fc1.out_features == 8
    assert model.user_mlp.fc1.out_features == 8


def test_causal_right_padding_and_last_valid_user_position() -> None:
    model = _model()
    model.eval()
    sequences = pad_sequences(
        [[1, 2, 3], [1, 2, 3, 4]],
        10,
        torch.device("cpu"),
    )

    assert sequences.tolist() == [
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
    ]
    encoded = model.encode(sequences)
    assert torch.allclose(encoded[0, :3], encoded[1, :3], atol=1e-6)
    expected_users = model.user_mlp(torch.stack([encoded[0, 2], encoded[1, 3]]))
    assert torch.allclose(model.user_vectors(sequences), expected_users, atol=1e-6)
    assert torch.isfinite(model.score_catalog(sequences)).all()


def _numpy_corrected_loss(
    user_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    target_ids: list[int],
    histories: list[set[int]],
    probabilities: np.ndarray,
) -> float:
    logits = user_vectors @ candidate_vectors.T
    logits -= np.log(probabilities[np.asarray(target_ids)])[None, :]
    for row_index, history in enumerate(histories):
        for candidate_index, candidate in enumerate(target_ids):
            if candidate_index != row_index and candidate in history:
                logits[row_index, candidate_index] = -1e4
    maxima = logits.max(axis=1, keepdims=True)
    log_denominator = maxima[:, 0] + np.log(np.exp(logits - maxima).sum(axis=1))
    return float(np.mean(log_denominator - np.diag(logits)))


def test_popularity_corrected_duplicate_mask_matches_numpy_reference() -> None:
    class FixedModel:
        max_length = 2
        item_count = 4

        def encode(self, _sequences):
            return torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.5, 0.5], [1.0, 1.0]],
                ],
                requires_grad=True,
            )

        def item_vectors(self, item_ids):
            table = torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
            return table[item_ids]

        @staticmethod
        def user_mlp(values):
            return values

    probabilities = np.asarray([1.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    loss = in_batch_loss(
        FixedModel(),
        [[1, 2, 3], [4, 2, 3]],
        torch.device("cpu"),
        probabilities,
    )
    users = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
        dtype=np.float64,
    )
    candidates = np.asarray(
        [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    target_ids = [2, 3, 2, 3]
    histories = [{1, 2}, {1, 2, 3}, {4, 2}, {4, 2, 3}]
    expected = _numpy_corrected_loss(
        users,
        candidates,
        target_ids,
        histories,
        probabilities,
    )

    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_popularity_distribution_and_non_finite_guards() -> None:
    probabilities = popularity_probabilities([[1, 1, 2], [2, 3]], 4, 1.0)
    assert probabilities.tolist() == pytest.approx([1.0, 0.4, 0.4, 0.2, 0.0])

    model = _model(item_count=4)
    with pytest.raises(RuntimeError, match="positive finite popularity"):
        in_batch_loss(
            model,
            [[1, 2, 3]],
            torch.device("cpu"),
            np.asarray([1.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        )
    with pytest.raises(ValueError, match="finite"):
        SASRec(
            item_count=2,
            max_length=10,
            embedding_dim=8,
            num_blocks=2,
            num_heads=2,
            dropout=0.0,
            arm="metadata",
            item_features=np.asarray([[math.nan], [1.0]], dtype=np.float32),
        )
