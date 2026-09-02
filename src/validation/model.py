from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np

# PyTorch requires this for deterministic CUDA matrix multiplications. Respect an
# explicit caller choice when one is already configured.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class TorchUnavailableError(RuntimeError):
    pass


ArmKind = Literal["metadata", "graph", "desc"]


if nn is not None:

    class ResidualMLP(nn.Module):
        def __init__(
            self,
            dimension: int,
            *,
            activation: Literal["relu", "gelu"],
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dimension)
            self.fc1 = nn.Linear(dimension, dimension * 4)
            self.fc2 = nn.Linear(dimension * 4, dimension)
            self.activation = activation

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            hidden = self.fc1(self.norm(values))
            hidden = (
                nn.functional.relu(hidden)
                if self.activation == "relu"
                else nn.functional.gelu(hidden)
            )
            return values + self.fc2(hidden)

    class SASRec(nn.Module):
        def __init__(
            self,
            item_count: int,
            max_length: int,
            embedding_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float,
            *,
            arm: ArmKind,
            item_features: "torch.Tensor | np.ndarray",
        ) -> None:
            super().__init__()
            self.max_length = max_length
            self.arm = arm
            features = torch.as_tensor(item_features, dtype=torch.float32)
            if features.ndim != 2 or features.shape[0] != item_count:
                raise ValueError("item_features must have one row per catalog item")
            if not torch.isfinite(features).all():
                raise ValueError("item_features must contain only finite values")
            padding = torch.zeros((1, features.shape[1]), dtype=features.dtype)
            self.register_buffer(
                "frozen_item_features",
                torch.cat([padding, features], dim=0),
                persistent=False,
            )
            self.item_projection = nn.Linear(features.shape[1], embedding_dim)
            self.item_mlp = ResidualMLP(
                embedding_dim,
                activation="relu",
            )
            self.position_embedding = nn.Embedding(max_length, embedding_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(embedding_dim)
            self.user_mlp = ResidualMLP(
                embedding_dim,
                activation="gelu",
            )
            self.apply(self._reset_parameters)
            for module in self.modules():
                if isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_normal_(module.in_proj_weight)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)

        @staticmethod
        def _reset_parameters(module: "nn.Module") -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_normal_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        @property
        def item_count(self) -> int:
            return int(self.frozen_item_features.shape[0] - 1)

        def item_vectors(self, item_ids: "torch.Tensor") -> "torch.Tensor":
            values = self.item_projection(self.frozen_item_features[item_ids])
            values = self.item_mlp(values)
            return values.masked_fill(item_ids.eq(0).unsqueeze(-1), 0.0)

        def catalog_vectors(self) -> "torch.Tensor":
            ids = torch.arange(
                1,
                self.item_count + 1,
                device=self.position_embedding.weight.device,
            )
            return self.item_vectors(ids)

        def encode(self, sequences: "torch.Tensor") -> "torch.Tensor":
            positions = torch.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
            hidden = self.dropout(self.item_vectors(sequences) + self.position_embedding(positions))
            causal = torch.triu(
                torch.ones(
                    sequences.shape[1],
                    sequences.shape[1],
                    dtype=torch.bool,
                    device=sequences.device,
                ),
                diagonal=1,
            )
            hidden = self.encoder(
                hidden,
                mask=causal,
                src_key_padding_mask=sequences.eq(0),
            )
            return self.norm(hidden)

        def user_vectors(self, sequences: "torch.Tensor") -> "torch.Tensor":
            encoded = self.encode(sequences)
            positions = torch.arange(sequences.shape[1], device=sequences.device).unsqueeze(0)
            last_positions = positions.masked_fill(sequences.eq(0), 0).max(dim=1).values
            users = encoded[
                torch.arange(len(sequences), device=sequences.device),
                last_positions,
            ]
            return self.user_mlp(users)

        def score_catalog(self, sequences: "torch.Tensor") -> "torch.Tensor":
            return self.user_vectors(sequences) @ self.catalog_vectors().T
else:

    class SASRec:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise TorchUnavailableError(
                "PyTorch is required; install the project with the 'train' extra"
            )


def require_torch() -> None:
    if torch is None:
        raise TorchUnavailableError(
            "PyTorch is required; install the project with the 'train' extra"
        )


def seed_everything(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def pad_sequences(
    sequences: list[list[int]],
    max_length: int,
    device: "torch.device",
) -> "torch.Tensor":
    result = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        values = sequence[-max_length:]
        if values:
            # Right padding avoids fully masked queries under a causal attention
            # mask. Only valid positions are selected for training and scoring.
            result[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return result


def in_batch_loss(
    model: SASRec,
    batch: list[list[int]],
    device: "torch.device",
    popularity_probabilities: "torch.Tensor | np.ndarray",
) -> "torch.Tensor":
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for sequence in batch:
        values = sequence[-(model.max_length + 1) :]
        inputs.append(values[:-1])
        targets.append(values[1:])
    padded_input = pad_sequences(inputs, model.max_length, device)
    padded_target = pad_sequences(targets, model.max_length, device)
    hidden = model.encode(padded_input)
    active = padded_target.ne(0)
    user_vectors = model.user_mlp(hidden[active])
    target_ids = padded_target[active]
    candidate_vectors = model.item_vectors(target_ids)
    logits = user_vectors @ candidate_vectors.T

    probabilities = torch.as_tensor(
        popularity_probabilities,
        dtype=logits.dtype,
        device=device,
    )
    if probabilities.ndim != 1 or len(probabilities) != model.item_count + 1:
        raise ValueError("popularity probabilities must have one value per item plus padding")
    candidate_probabilities = probabilities[target_ids]
    if not torch.isfinite(candidate_probabilities).all() or candidate_probabilities.le(0).any():
        raise RuntimeError("in-batch candidates require positive finite popularity")
    logits = logits - torch.log(candidate_probabilities).unsqueeze(0)

    active_positions = active.nonzero(as_tuple=False)
    target_list = target_ids.tolist()
    for row_index, (batch_index, position) in enumerate(active_positions.tolist()):
        positive = int(target_ids[row_index])
        history = set(padded_input[batch_index, : position + 1].tolist()) - {0}
        history.add(positive)
        mask = torch.tensor(
            [int(item) in history for item in target_list],
            dtype=torch.bool,
            device=device,
        )
        mask[row_index] = False
        logits[row_index, mask] = -1e4
    if not torch.isfinite(logits).all():
        raise RuntimeError("training produced non-finite logits")
    labels = torch.arange(len(target_ids), device=device)
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("training produced a non-finite loss")
    return loss


def catalog_score_batches(
    model: SASRec,
    histories: list[list[int]],
    *,
    batch_size: int,
    device: "torch.device",
):
    model.eval()
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            batch = pad_sequences(histories[start : start + batch_size], model.max_length, device)
            scores = model.score_catalog(batch)
            if not torch.isfinite(scores).all():
                raise RuntimeError("catalog scoring produced non-finite values")
            yield start, scores.cpu().numpy().astype(np.float32)


def save_checkpoint(path: Path, model: SASRec, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)
