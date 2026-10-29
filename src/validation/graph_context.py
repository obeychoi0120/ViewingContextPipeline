"""Scene encoder dimensions and internal aggregation model constants."""

GRAPH_ARCHITECTURE = "sasrec-role-graph/v9"
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
GNN_ARCHITECTURE = "sasrec-role-graph/v10"
GNN_MODEL = {
    **GRAPH_MODEL,
    "fusion": "role_gather_scene_residual_mlp_then_gcn_mean",
    "aggregation": "gnn",
    "gcn": {
        "layers": 1,
        "hidden_dim": 512,
        "similarity_input": "entity_action_context_before_scene_mlp",
        "similarity_dim": 384,
        "similarity": "nonnegative_cosine",
        "similarity_eps": 1e-8,
        "self_loop": "unit_diagonal",
        "normalization": "symmetric_degree",
        "residual": True,
        "activation": "relu",
        "bias": True,
        "detach_similarity": False,
        "connectivity": "all_pairs_within_video",
    },
}
TRANSFORMER_ARCHITECTURE = "sasrec-role-graph/v11"
# One block shared by every Transformer variant, with SASRec block sizes.
SCENE_TRANSFORMER = {"layers": 1, "heads": 2, "ffn_ratio": 4}
# Variants differ only in position input and attention scope, so paired runs
# separate scene interaction (transformer vs transformer_local) from scene
# order (transformer_pos vs transformer) at an identical parameter count.
TRANSFORMER_VARIANTS = {
    "transformer": {"position": "none", "attention": "all_scenes"},
    "transformer_pos": {"position": "sinusoidal_valid_scene_rank", "attention": "all_scenes"},
    "transformer_local": {"position": "none", "attention": "self_only"},
}
ATTENTION_ARCHITECTURE = "sasrec-role-graph/v12"
# Learned scene weights where the Transformer variants cannot isolate them:
# attn_pool learns only the video readout weights (no scene exchange), and
# gnn_attn changes only how gnn sets its neighbour weights (learned, not cosine).
ATTENTION_AGGREGATIONS = ("attn_pool", "gnn_attn")
SCENE_AGGREGATIONS = ("mean", "gnn", *TRANSFORMER_VARIANTS, *ATTENTION_AGGREGATIONS, "self_only")
# Aggregations that pass messages between a video's scenes (S² pairwise cost).
SCENE_EXCHANGES = frozenset({"gnn", "gnn_attn", *TRANSFORMER_VARIANTS})
ATTENTION_POOL_MODEL = {
    **GRAPH_MODEL,
    "fusion": "role_gather_scene_residual_mlp_then_attention_pool",
    "aggregation": "attn_pool",
    "attention_pool": {
        "input": "scene_mlp_output",
        "scene_exchange": "none",
        "score_input": "unit_l2_times_sqrt_dim",
        "score_input_eps": 1e-8,
        "score": "linear_tanh_linear",
        "hidden_dim": 128,
        "score_bias": False,
        "score_init": "zero_output_uniform_weights",
        "normalization": "softmax_within_video",
        "stabilization": "detached_video_max",
    },
}
GNN_ATTENTION_MODEL = {
    **GNN_MODEL,
    "fusion": "role_gather_scene_residual_mlp_then_attention_gcn_mean",
    "aggregation": "gnn_attn",
    "gcn": {
        **GNN_MODEL["gcn"],
        "query_key_input": "unit_l2_times_sqrt_dim",
        "similarity": "scaled_dot_product",
        "heads": 1,
        "query_key_dim": 128,
        "query_key_bias": False,
        # Initial attention logit std 0.5, as the scene Transformer's xavier in_proj.
        "query_key_init": "normal_std_1_over_sqrt_2_fan_in",
        "self_loop": "included_in_softmax",
        "normalization": "row_softmax",
    },
}


def transformer_model(aggregation):
    return {
        **GRAPH_MODEL,
        "fusion": "role_gather_scene_residual_mlp_then_transformer_mean",
        "aggregation": aggregation,
        "transformer": {
            **SCENE_TRANSFORMER,
            **TRANSFORMER_VARIANTS[aggregation],
            "hidden_dim": 512,
            "ffn_dim": 512 * SCENE_TRANSFORMER["ffn_ratio"],
            "activation": "relu",
            "normalization": "pre_layernorm",
            "final_norm": False,
            "residual_scale": "learned_scalar_per_sublayer_init_0",
            "dropout": "validation.model.dropout",
            "input": "scene_mlp_output",
            "connectivity": "within_video",
            "position_scale": "unit_norm",
            "position_injection": "sublayer_inputs_only",
            "attention_init": "xavier_normal_in_proj",
        },
    }


SCENE_SEGMENTS_VERSION = "scene-segments/v1"
# Optional stages after gnn_attn's scene exchange, chosen per Arm (arm_registry.Arm.extensions):
#   segments    scene-segment hypergraph (scene_segments sidecar of the source bundle): scenes pass to their
#               segments and back, and the video readout adds a per-segment mean; both gates start at 0.
#   video_node  the title leaves the scene vectors and starts a video node that gathers from the scenes,
#               returns to them and gathers again; the video node is the item vector.
# With both, the video node gathers from the segment-updated scenes and the segment readout is added to it.
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
        "applied_after": "attention_gcn",
    },
    "video_node": {
        "version": "video-node-v1",
        "scene_readout": "zero_title_entity_action_context_concat",
        "init": "v0 = scene_mlp([title, 0, 0, 0]) (the title-only fallback vector)",
        "steps": [
            "v1 = v0 + relu(W_up [v0; mean_s h_s])",
            "h_s' = h_s + relu(W_down [h_s; v1])",
            "v2 = v1 + relu(W_up [v1; mean_s h_s'])",
        ],
        "shared": "W_up used in steps 1 and 3",
        "linear": "1024_to_512_with_bias",
        "readout": "v2",
        "videos_without_scenes": "v0",
        "applied_after": "scene_exchange_and_earlier_extensions",
    },
}


def check_extensions(extensions, aggregation):
    """Extensions are gnn_attn-only stages; returns them as a frozenset."""
    extensions = frozenset(extensions)
    if extensions - EXTENSION_MODELS.keys():
        raise ValueError("unknown scene extension: " + ", ".join(sorted(extensions - EXTENSION_MODELS.keys())))
    if extensions and aggregation != "gnn_attn":
        raise ValueError("scene extensions require the gnn_attn aggregation")
    return extensions

