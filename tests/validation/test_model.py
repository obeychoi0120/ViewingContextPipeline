from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from validation.config import ValidationConfig  # noqa: E402
from validation.cohort import prepare_cohort  # noqa: E402
from validation.model import SASRec, in_batch_loss, pad_sequences  # noqa: E402
from validation.recommendation import (  # noqa: E402
    popularity_probabilities,
    train_recommendation_arms,
)
from validation.recommendation_contracts import (  # noqa: E402
    ARCHITECTURE_VERSION,
    RECOMMENDATION_ARMS,
    TRAINING_RUN_SCHEMA_VERSION,
)
from pipeline_runtime import read_jsonl, write_json  # noqa: E402

from conftest import config_data  # noqa: E402


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
    assert model.item_mlp.fc1.out_features == 32
    assert model.user_mlp.fc1.out_features == 32


def test_xavier_normal_and_zero_bias_initialization() -> None:
    torch.manual_seed(42)
    model = _model(dimension=64)

    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            expected_variance = 2.0 / (fan_in + fan_out)
            observed_variance = float(module.weight.detach().var(unbiased=False))
            assert observed_variance == pytest.approx(expected_variance, rel=0.5)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias).item() == 0
        if isinstance(module, torch.nn.MultiheadAttention):
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(
                module.in_proj_weight
            )
            expected_variance = 2.0 / (fan_in + fan_out)
            observed_variance = float(
                module.in_proj_weight.detach().var(unbiased=False)
            )
            assert observed_variance == pytest.approx(expected_variance, rel=0.5)
            assert torch.count_nonzero(module.in_proj_bias).item() == 0


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


def test_four_arms_three_seeds_one_epoch_selection_refit_smoke(tmp_path) -> None:
    run_root = tmp_path / "run"
    cohort = run_root / "data" / "cohort"
    representations = run_root / "validation" / "representations"
    recommendations = run_root / "validation" / "recommendations"
    data = config_data(tmp_path)
    data["run_id"] = "train-smoke"
    data["output_dir"] = run_root
    data["model"].update(max_epochs=1, patience=1, seeds=[42, 43, 44])
    config = ValidationConfig.model_validate(data)
    config.dataset.videos_dir.mkdir()
    for item in range(1, 9):
        (config.dataset.videos_dir / f"{item}.mp4").write_bytes(b"video")
    config.dataset.pairs_tsv.write_text(
        "u1\t1 2 3 4 5 6\nu2\t2 3 4 5 6 7\n", encoding="utf-8"
    )
    config.dataset.titles_csv.write_text(
        "".join(f"{item},Title {item}\n" for item in range(1, 9)), encoding="utf-8"
    )
    prepare_cohort(config, plan_only=True)
    prepare_cohort(config, probe=lambda _: 30.0)
    catalog = read_jsonl(cohort / "catalog.jsonl")
    assert [row["item_id"] for row in catalog] == [str(item) for item in range(1, 8)]
    write_json(
        representations / "item_index.json",
        {row["item_id"]: index for index, row in enumerate(catalog)},
    )
    representations.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for branch in RECOMMENDATION_ARMS.values():
        np.savez_compressed(
            representations / f"{branch}_embeddings.npz",
            values=rng.normal(size=(len(catalog), 1024)).astype(np.float32),
        )

    result = train_recommendation_arms(
        config,
        {
            "run_id": "train-smoke",
            "run_root": str(run_root),
            "paths": {
                "representations_dir": str(representations),
                "recommendations_dir": str(recommendations),
            },
        },
    )

    training_runs = read_jsonl(recommendations / "training_runs.jsonl")
    checkpoints = list((recommendations / "checkpoints").glob("**/sasrec.pt"))
    assert len(result["runs"]) == len(training_runs) == len(checkpoints) == 12
    assert len(read_jsonl(recommendations / "per_user_metrics.jsonl")) == 24
    assert {(row["seed"], row["arm"]) for row in training_runs} == {
        (seed, arm) for seed in (42, 43, 44) for arm in RECOMMENDATION_ARMS
    }
    for row in training_runs:
        assert row["schema_version"] == TRAINING_RUN_SCHEMA_VERSION
        assert row["architecture_version"] == ARCHITECTURE_VERSION
        assert row["selection"]["best_validation"]["epoch"] == 1
        assert row["selection"]["epochs_completed"] == 1
        assert row["refit"]["data"] == "train+valid_target"
        assert row["refit"]["epochs_completed"] == 1
