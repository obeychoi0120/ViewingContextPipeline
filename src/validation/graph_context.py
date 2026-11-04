"""Shared scene encoder and activity hypergraph contracts."""

GRAPH_MODEL = {
    "hidden_dim": 128,
    "scene_dim": 512,
    "title_dim": 128,
    "fusion": "role_gather_scene_residual_mlp_then_mean",
    "scene_readout": "title_entity_action_context_concat",
    "projections": "separate_title_entity_action_context",
    "roles": ["actor", "target", "tool", "place"],
    "role_fusion": "action_plus_relu_linear_640_to_128",
    "scene_mlp": {"hidden_dim": 512, "activation": "relu", "normalization": "none",
                  "linear_layers": 1, "formula": "x_plus_relu_linear_x"},
    "aggregation": "mean",
    "input_dim": 1024,
    "state_lookup": "embedding",
}

SCENE_SEGMENTS_VERSION = "scene-segments/v2"
EXTENSION_MODELS = {
    "segments": {
        "version": "segments-v1",
        "source": "scene_segments sidecar of the source bundle ([scene row, segment] incidence)",
        "sidecar_version": SCENE_SEGMENTS_VERSION,
        "segment": "u_k = relu(W_u [mean_{s in k} h_s; log|k|; |k| / scenes in video])",
        "scene": "h_s = h_s + g_s relu(W_s [h_s; mean_{k contains s} u_k]) for scenes in a segment",
        "video": "z = z + g_v W_v mean_{k in video} u_k (each segment weighted equally)",
        "gates": "learned_scalars_init_0",
        "weight_init": "xavier_normal_private_generator_seeded_by_torch_initial_seed",
        "bias_init": "zeros",
        "applied_after": "scene_mlp_without_scene_exchange",
    },
}


def check_extensions(extensions, aggregation):
    extensions = frozenset(extensions)
    if aggregation != "mean":
        raise ValueError("scene aggregation must be mean")
    if extensions - EXTENSION_MODELS.keys():
        raise ValueError("unknown scene extension: " + ", ".join(sorted(extensions - EXTENSION_MODELS.keys())))
    return extensions
