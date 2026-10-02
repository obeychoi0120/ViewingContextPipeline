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
            self.fc1 = nn.Linear(dimension, dimension)
            self.fc2 = nn.Linear(dimension, dimension)
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
            item_features: "dict[str, np.ndarray] | torch.Tensor | np.ndarray",
        ) -> None:
            super().__init__()
            self.max_length = max_length
            self.arm = arm
            if embedding_dim % 4:
                raise ValueError("recommendation dimension must be divisible by four")
            if isinstance(item_features, dict):
                titles = torch.as_tensor(item_features["title_values"], dtype=torch.float32)
                videos = torch.as_tensor(item_features["video_values"], dtype=torch.float32)
                title_available = torch.as_tensor(
                    item_features["title_available"], dtype=torch.bool
                )
                video_available = torch.as_tensor(
                    item_features["video_available"], dtype=torch.bool
                )
            else:
                # Standalone callers may supply a single component. Production uses explicit masks.
                features = torch.as_tensor(item_features, dtype=torch.float32)
                titles = features if arm == "metadata" else torch.zeros_like(features)
                videos = torch.zeros_like(features) if arm == "metadata" else features
                title_available = titles.ne(0).any(dim=1)
                video_available = videos.ne(0).any(dim=1)
            if (
                titles.ndim != 2
                or titles.shape != videos.shape
                or titles.shape[0] != item_count
                or title_available.shape != (item_count,)
                or video_available.shape != (item_count,)
                or not torch.isfinite(titles).all()
                or not torch.isfinite(videos).all()
            ):
                raise ValueError(
                    "invalid component feature arrays: require matching shapes and finite values"
                )
            for name, values in (
                ("title_features", titles),
                ("video_features", videos),
                ("title_available", title_available),
                ("video_available", video_available),
            ):
                padding = values.new_zeros((1, *values.shape[1:]))
                self.register_buffer(name, torch.cat([padding, values.detach()]), persistent=False)
            if arm == "metadata":
                self.item_projection = nn.Linear(titles.shape[1], embedding_dim)
            else:
                self.title_projection = nn.Linear(titles.shape[1], embedding_dim // 4)
                self.video_projection = nn.Linear(videos.shape[1], embedding_dim * 3 // 4)
            self.item_norm = nn.LayerNorm(embedding_dim, eps=1e-5)
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
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=num_blocks,
                enable_nested_tensor=False,
            )
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
            return int(self.title_features.shape[0] - 1)

        def item_vectors(self, item_ids: "torch.Tensor") -> "torch.Tensor":
            from validation.profiling import span

            title_present = self.title_available[item_ids]
            video_present = self.video_available[item_ids]
            if self.arm == "metadata":
                with span("item_projection"):
                    values = self.item_projection(self.title_features[item_ids])
                available = title_present
            else:
                with span("title_projection"):
                    title = self.title_projection(self.title_features[item_ids])
                with span("video_projection"):
                    video = self.video_projection(self.video_features[item_ids])
                title = title.masked_fill(~title_present.unsqueeze(-1), 0.0)
                video = video.masked_fill(~video_present.unsqueeze(-1), 0.0)
                values = torch.cat([title, video], dim=-1)
                available = title_present | video_present
            return self.item_norm(values).masked_fill(~available.unsqueeze(-1), 0.0)

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


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


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
