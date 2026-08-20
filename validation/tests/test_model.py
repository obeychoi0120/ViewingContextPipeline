import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vc_validation.model import SASRec, in_batch_loss


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
