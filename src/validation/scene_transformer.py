"""Self-attention across each video's scenes, after the scene MLP and before mean pooling."""

import math

from validation.model import torch, nn
from validation.profiling import span, workload

POSITIONS = ("none", "sinusoidal_valid_scene_rank")
ATTENTION = ("all_scenes", "self_only")


def scene_positions(count, dimension, device, dtype):
    """Unit-norm sinusoidal encoding of each valid scene's rank within its video.

    The raw sinusoid has norm sqrt(dimension / 2) (16 at 512), whereas scene MLP
    outputs from unit-norm BGE features start near norm 1; unscaled positions
    would dominate the pre-LayerNorm input and hide scene content from attention.
    """
    rank = torch.arange(count, device=device, dtype=dtype).unsqueeze(-1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dimension)
    )
    table = torch.zeros((count, dimension), device=device, dtype=dtype)
    table[:, 0::2] = torch.sin(rank * frequency)
    table[:, 1::2] = torch.cos(rank * frequency)
    return table * math.sqrt(2 / dimension)


class SceneBlock(nn.Module):
    """Pre-LN block of SASRec size whose sublayer outputs start switched off.

    LayerNorm rescales scene vectors of norm ~1 to norm ~sqrt(dimension), so an
    ordinary pre-LN block adds sublayer outputs about 10x larger than its input
    and the video vector no longer matches mean/gnn in scale. Zero-initialized
    residual scales (ReZero) make every variant start exactly as mean pooling;
    any departure is learned from the recommendation loss.
    """

    def __init__(self, dimension, *, heads, ffn_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.linear1 = nn.Linear(dimension, dimension * ffn_ratio)
        self.linear2 = nn.Linear(dimension * ffn_ratio, dimension)
        # Same dropout placement as torch.nn.TransformerEncoderLayer.
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.zeros(2))  # [attention, FFN]

    def forward(self, values, mask=None, positions=None):
        # Positions enter only the sublayer inputs, never the residual stream that is pooled:
        # mean(positions) depends only on the scene count and would add a length signal.
        hidden = self.norm1(values if positions is None else values + positions)
        hidden = self.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)[0]
        values = values + self.residual_scale[0] * self.dropout1(hidden)
        hidden = self.norm2(values if positions is None else values + positions)
        hidden = self.linear2(self.dropout(nn.functional.relu(self.linear1(hidden))))
        return values + self.residual_scale[1] * self.dropout2(hidden)


class SceneTransformer(nn.Module):
    def __init__(self, dimension, *, layers, heads, ffn_ratio, dropout, position, attention):
        super().__init__()
        if position not in POSITIONS or attention not in ATTENTION:
            raise ValueError("unsupported scene Transformer position or attention scope")
        self.position = position
        self.attention = attention
        # No final LayerNorm: it would remove the exact mean-pooling start and
        # rescale scene vectors to norm ~sqrt(dimension).
        self.layers = nn.ModuleList(
            SceneBlock(dimension, heads=heads, ffn_ratio=ffn_ratio, dropout=dropout)
            for _ in range(layers)
        )

    def residual_scales(self):
        return [layer.residual_scale.detach().cpu().tolist() for layer in self.layers]

    def _encode(self, values, mask, positions):
        for layer in self.layers:
            values = layer(values, mask, positions)
        return values

    def forward(self, scenes, groups):
        output = torch.zeros_like(scenes)
        for indices in groups:
            with span("scene_transformer", cuda=True):
                values = scenes[indices]
                size = indices.shape[1]
                mask = None
                if self.attention == "self_only":
                    # Same parameters as all_scenes, but no information crosses scenes.
                    mask = ~torch.eye(size, dtype=torch.bool, device=scenes.device)
                positions = None
                if self.position != "none":
                    positions = scene_positions(size, values.shape[-1], values.device, values.dtype)
                updated = self._encode(values, mask, positions)
                output = output.index_copy(0, indices.reshape(-1), updated.flatten(0, 1))
            workload(scene_pairs=indices.shape[0] * size**2)
        return output
