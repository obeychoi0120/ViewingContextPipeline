import numpy as np
import pytest

torch = pytest.importorskip("torch")

from validation.model import SASRec, in_batch_loss, pad_sequences  # noqa: E402
from validation.config import ValidationConfig  # noqa: E402
from validation.recommendation import train_recommendation_arms  # noqa: E402
from pipeline_runtime import read_jsonl, write_json, write_jsonl  # noqa: E402

from conftest import config_data  # noqa: E402


@pytest.mark.parametrize("arm", ["id", "graph", "desc"])
def test_three_item_towers_forward_and_loss(arm) -> None:
    features = None if arm == "id" else np.ones((12, 1024), dtype=np.float32)
    model = SASRec(item_count=12, max_length=10, embedding_dim=8, num_blocks=2, num_heads=2, dropout=0.0, arm=arm, item_features=features)
    sequences = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]])
    assert model.encode(sequences).shape == (2, 10, 8)
    assert model.score_catalog(sequences).shape == (2, 12)
    loss = in_batch_loss(model, [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    if arm != "id":
        assert model.frozen_item_features.requires_grad is False
        assert model.item_projection.weight.grad is not None


def test_short_sequences_are_right_padded_and_score_finitely() -> None:
    model = SASRec(item_count=12, max_length=10, embedding_dim=8, num_blocks=2, num_heads=2, dropout=0.0)
    sequences = pad_sequences([[1, 2, 3], [4, 5]], 10, torch.device("cpu"))

    assert sequences.tolist() == [
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
        [4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert torch.isfinite(model.score_catalog(sequences)).all()


def test_four_arms_three_seeds_one_epoch_training_smoke(tmp_path) -> None:
    run_root = tmp_path / "run"
    cohort = run_root / "data" / "cohort"
    representations = run_root / "validation" / "representations"
    recommendations = run_root / "validation" / "recommendations"
    catalog = [
        {"item_id": str(index), "content_id": f"c{index}"}
        for index in range(1, 9)
    ]
    sequences = [
        {
            "user_id": "u1",
            "train": ["1", "2", "3", "4"],
            "valid_target": "5",
            "test_target": "6",
            "stratum": "5-9",
        },
        {
            "user_id": "u2",
            "train": ["2", "3", "4", "5"],
            "valid_target": "6",
            "test_target": "7",
            "stratum": "5-9",
        },
    ]
    write_jsonl(cohort / "catalog.jsonl", catalog)
    write_jsonl(cohort / "sequences.jsonl", sequences)
    write_json(
        representations / "item_index.json",
        {row["item_id"]: index for index, row in enumerate(catalog)},
    )
    representations.mkdir(parents=True, exist_ok=True)
    for branch in ("graph_qwen", "graph_gemini", "desc"):
        np.savez_compressed(
            representations / f"{branch}_embeddings.npz",
            values=np.ones((len(catalog), 1024), dtype=np.float32),
        )

    data = config_data(tmp_path)
    data["model"].update(max_epochs=1, patience=1, seeds=[42, 43, 44])
    data["evaluation"]["cutoffs"] = [4]
    config = ValidationConfig.model_validate(data)
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

    assert len(result["runs"]) == 12
    assert len(read_jsonl(recommendations / "training_runs.jsonl")) == 12
    assert len(read_jsonl(recommendations / "per_user_metrics.jsonl")) == 24
