"""Learned scene weights: attention video pooling and an attention-weighted scene GCN."""

import math
from contextlib import contextmanager

from validation.model import torch, nn
from validation.scene_gcn import SceneGCN

POOL_STATISTICS = (
    "normalized_entropy", "max_weight_ratio", "first_scene_weight_ratio", "last_scene_weight_ratio"
)
ROW_STATISTICS = ("normalized_entropy", "max_weight_ratio", "self_weight_ratio")


def unit_rms(values):
    """Scale-free score input: gnn's unit L2 direction (same eps) rescaled to unit RMS.

    Scene vectors have norm ~1, so raw query/key and score-MLP inputs give scores
    near 0 and weights that stay almost uniform for hundreds of AdamW steps (as
    the Transformer's pre-LN avoids). Only the scores see this; values do not.
    """
    return nn.functional.normalize(values, dim=-1, eps=1e-8) * math.sqrt(values.shape[-1])


def segment_softmax(scores, indices, count):
    """Softmax of each video's scene scores, stabilized by the detached video maximum."""
    maxima = scores.new_full((count,), -math.inf)
    maxima.scatter_reduce_(0, indices, scores.detach(), reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima[indices])
    # Each video's maximum contributes exp(0) = 1, so no total can vanish.
    totals = weights.new_zeros(count).index_add_(0, indices, weights)
    return weights / totals[indices]


class SceneAttentionPool(nn.Module):
    """Video readout z = Σ β_s h_s, β = softmax_s(wᵀ tanh(W ĥ_s + b)), ĥ = unit_rms(h).

    No information crosses scenes before the readout. w starts at zero, so every
    video starts with uniform weights (mean pooling), and the output stays a
    convex combination of scene vectors, keeping the mean/gnn vector scale.
    w has no bias: softmax ignores a shift shared by all scenes of a video.
    """

    def __init__(self, dimension, hidden):
        super().__init__()
        self.hidden = nn.Linear(dimension, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.start_uniform()

    def start_uniform(self):
        nn.init.zeros_(self.score.weight)

    def weights(self, scenes, videos, count):
        scores = self.score(torch.tanh(self.hidden(unit_rms(scenes)))).squeeze(-1)
        return segment_softmax(scores, videos, count)

    def forward(self, scenes, videos, count):
        weights = self.weights(scenes, videos, count)
        output = scenes.new_zeros((count, scenes.shape[-1]))
        return output.index_add_(0, videos, scenes * weights.unsqueeze(-1))


class SceneAttentionGCN(SceneGCN):
    """gnn with its cosine adjacency replaced by single-head scaled dot-product attention.

    gnn weights unit directions u by the fixed form u_iᵀu_j; here the same
    directions (at unit RMS) pass a learned form W_qᵀW_k and a row softmax. The
    title-free input, self-loop, residual ReLU projection, layer count and
    all-pairs connectivity are those of SceneGCN.
    """

    adjacency_span = "scene_attention"

    def __init__(self, dimension, similarity_dim, key_dim):
        super().__init__(dimension)
        self.query = nn.Linear(similarity_dim, key_dim, bias=False)
        self.key = nn.Linear(similarity_dim, key_dim, bias=False)
        self.reset_query_key()

    def reset_query_key(self):
        """N(0, 1/(2·fan_in)): unit-RMS inputs give query/key entries of variance 1/2 and
        attention logits of std 0.5 for any key width, as the scene Transformer's xavier
        in_proj on pre-LN inputs. Xavier here would give std 1.5 (near one-hot weights).
        """
        for layer in (self.query, self.key):
            nn.init.normal_(layer.weight, std=(2 * layer.in_features) ** -0.5)

    def adjacency(self, features):
        features = unit_rms(features)
        scores = self.query(features) @ self.key(features).transpose(-1, -2)
        return (scores / math.sqrt(self.key.out_features)).softmax(dim=-1)


def pool_weight_statistics(weights, videos, count):
    """Sums over videos with at least two scenes; a video's scenes are contiguous and ordered."""
    sizes = weights.new_zeros(count).index_add_(0, videos, torch.ones_like(weights))
    rank = torch.arange(len(videos), device=videos.device) - (sizes.cumsum(0) - sizes)[videos]
    entropy = weights.new_zeros(count).index_add_(0, videos, -torch.special.xlogy(weights, weights))
    maxima = weights.new_zeros(count).scatter_reduce_(0, videos, weights, "amax")
    first = weights.new_zeros(count).index_add_(0, videos, weights * (rank == 0))
    last = weights.new_zeros(count).index_add_(0, videos, weights * (rank == sizes[videos] - 1))
    multiple = sizes >= 2
    size = sizes[multiple]
    return int(multiple.sum()), {
        "normalized_entropy": (entropy[multiple] / size.log()).sum(),
        "max_weight_ratio": (maxima[multiple] * size).sum(),
        "first_scene_weight_ratio": (first[multiple] * size).sum(),
        "last_scene_weight_ratio": (last[multiple] * size).sum(),
    }


def row_weight_statistics(adjacency):
    """Sums over the attention rows of a (videos, S, S) group; single scenes are skipped."""
    size = adjacency.shape[-1]
    if size < 2:
        return 0, dict.fromkeys(ROW_STATISTICS, 0.0)
    entropy = -torch.special.xlogy(adjacency, adjacency).sum(-1) / math.log(size)
    return entropy.numel(), {
        "normalized_entropy": entropy.sum(),
        "max_weight_ratio": adjacency.amax(-1).sum() * size,
        "self_weight_ratio": adjacency.diagonal(dim1=-2, dim2=-1).sum() * size,
    }


@contextmanager
def recorded_weights(module):
    """Collect learned scene-weight statistics on every forward of `module`.

    The yielded report is complete once the block exits. Ratios are S times a
    weight: 1 is the uniform (mean aggregation) value and S a one-hot weight.
    Normalized entropy is entropy / log S: 1 uniform, 0 one-hot.
    """
    pooled = isinstance(module, SceneAttentionPool)
    names = POOL_STATISTICS if pooled else ROW_STATISTICS
    totals = dict.fromkeys(names, 0.0)
    report = {"videos" if pooled else "rows": 0}

    def add(count, values):
        report["videos" if pooled else "rows"] += count
        for name, value in values.items():
            totals[name] += float(value)

    @torch.no_grad()
    def collect(_, args, __):
        if pooled:
            add(*pool_weight_statistics(module.weights(*args), *args[1:]))
        else:
            for indices in args[2]:
                add(*row_weight_statistics(module.adjacency(args[1][indices])))

    hook = module.register_forward_hook(collect)
    try:
        yield report
    finally:
        hook.remove()
    count = report["videos" if pooled else "rows"]
    report.update({f"{name}_mean": total / count if count else None for name, total in totals.items()})
    norms = {"score": module.score} if pooled else {"query": module.query, "key": module.key}
    report.update({f"{name}_weight_norm": float(layer.weight.detach().norm())
                   for name, layer in norms.items()})
