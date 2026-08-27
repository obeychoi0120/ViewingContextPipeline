from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np

# PyTorch requires this to make CUDA matrix multiplications deterministic. Keep
# an explicit caller choice when one is already configured.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class TorchUnavailableError(RuntimeError):
    pass


ArmKind = Literal["id", "graph", "desc"]


if nn is not None:
    class ResidualItemMLP(nn.Module):
        def __init__(self, dimension: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(dimension)
            self.fc1 = nn.Linear(dimension, dimension * 2)
            self.fc2 = nn.Linear(dimension * 2, dimension)
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            residual = self.fc2(self.dropout(nn.functional.gelu(self.fc1(self.norm(values)))))
            return values + self.dropout(residual)


    class SASRec(nn.Module):
        def __init__(self, item_count: int, max_length: int, embedding_dim: int, num_blocks: int, num_heads: int, dropout: float, *, arm: ArmKind = "id", item_features: "torch.Tensor | np.ndarray | None" = None):
            super().__init__()
            self.max_length = max_length
            self.arm = arm
            if arm == "id":
                if item_features is not None:
                    raise ValueError("ID arm does not accept item_features")
                self.item_embedding = nn.Embedding(item_count + 1, embedding_dim, padding_idx=0)
                self.item_projection = None
                self.register_buffer("frozen_item_features", None)
            else:
                if item_features is None:
                    raise ValueError(f"{arm} arm requires item_features")
                features = torch.as_tensor(item_features, dtype=torch.float32)
                if features.ndim != 2 or features.shape[0] != item_count:
                    raise ValueError("item_features must have one row per catalog item")
                padding = torch.zeros((1, features.shape[1]), dtype=features.dtype)
                self.register_buffer("frozen_item_features", torch.cat([padding, features], dim=0), persistent=False)
                self.item_embedding = None
                self.item_projection = nn.Linear(features.shape[1], embedding_dim)
            self.item_mlp = ResidualItemMLP(embedding_dim, dropout)
            self.position_embedding = nn.Embedding(max_length, embedding_dim)
            layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(embedding_dim)

        def item_vectors(self, item_ids: "torch.Tensor") -> "torch.Tensor":
            if self.arm == "id":
                values = self.item_embedding(item_ids)
            else:
                values = self.item_projection(self.frozen_item_features[item_ids])
            values = self.item_mlp(values)
            return values.masked_fill(item_ids.eq(0).unsqueeze(-1), 0.0)

        def catalog_vectors(self) -> "torch.Tensor":
            count = self.item_embedding.num_embeddings - 1 if self.arm == "id" else self.frozen_item_features.shape[0] - 1
            ids = torch.arange(1, count + 1, device=self.position_embedding.weight.device)
            return self.item_vectors(ids)

        def encode(self, sequences: "torch.Tensor") -> "torch.Tensor":
            positions = torch.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
            hidden = self.dropout(self.item_vectors(sequences) + self.position_embedding(positions))
            causal = torch.triu(torch.ones(sequences.shape[1], sequences.shape[1], dtype=torch.bool, device=sequences.device), diagonal=1)
            hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=sequences.eq(0))
            return self.norm(hidden)

        def score_catalog(self, sequences: "torch.Tensor") -> "torch.Tensor":
            encoded = self.encode(sequences)
            positions = torch.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
            last_positions = positions.masked_fill(sequences.eq(0), 0).max(dim=1).values
            user = encoded[torch.arange(len(sequences), device=sequences.device), last_positions]
            return user @ self.catalog_vectors().T
else:
    class SASRec:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise TorchUnavailableError("PyTorch is required; install the project with the 'train' extra")


def require_torch() -> None:
    if torch is None:
        raise TorchUnavailableError("PyTorch is required; install the project with the 'train' extra")


def seed_everything(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def pad_sequences(sequences: list[list[int]], max_length: int, device: "torch.device") -> "torch.Tensor":
    result = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        values = sequence[-max_length:]
        if values:
            # Right padding avoids fully masked queries under a causal attention
            # mask. Left-padded queries can produce NaNs in the CUDA Transformer,
            # which were previously misreported as rank-one predictions.
            result[index, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return result


def in_batch_loss(model: SASRec, batch: list[list[int]], device: "torch.device") -> "torch.Tensor":
    inputs, targets = [], []
    for sequence in batch:
        values = sequence[-(model.max_length + 1):]
        inputs.append(values[:-1])
        targets.append(values[1:])
    padded_input = pad_sequences(inputs, model.max_length, device)
    padded_target = pad_sequences(targets, model.max_length, device)
    hidden = model.encode(padded_input)
    active = padded_target.ne(0)
    user_vectors = hidden[active]
    target_ids = padded_target[active]
    candidate_vectors = model.item_vectors(target_ids)
    logits = user_vectors @ candidate_vectors.T
    active_positions = active.nonzero(as_tuple=False)
    target_list = target_ids.tolist()
    for row_index, (batch_index, position) in enumerate(active_positions.tolist()):
        history = set(padded_input[batch_index, :position + 1].tolist()) - {0}
        if history:
            mask = torch.tensor([int(item) in history for item in target_list], dtype=torch.bool, device=device)
            mask[row_index] = False
            logits[row_index, mask] = -1e4
    if not torch.isfinite(logits).all():
        raise RuntimeError("training produced non-finite logits")
    loss = nn.functional.cross_entropy(logits, torch.arange(len(target_ids), device=device))
    if not torch.isfinite(loss):
        raise RuntimeError("training produced a non-finite loss")
    return loss


def catalog_score_batches(model: SASRec, histories: list[list[int]], *, batch_size: int, device: "torch.device"):
    model.eval()
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            batch = pad_sequences(histories[start:start + batch_size], model.max_length, device)
            scores = model.score_catalog(batch)
            if not torch.isfinite(scores).all():
                raise RuntimeError("catalog scoring produced non-finite values")
            yield start, scores.cpu().numpy().astype(np.float32)


def save_checkpoint(path: Path, model: SASRec, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)
