"""Bind source-level frozen features and Arm-specific models to recommendation results."""

from artifact_io import fingerprint


def read_state(context, branch):
    from arm_registry import feature_source, registry
    from artifact_io import read_json

    source = feature_source(registry(context.config)[branch])
    root = context.representations_dir / source
    manifest = read_json(root / "manifest.json")
    return {
        "input_hash": manifest["input_hash"],
        "shareable": True,
        "recommendation_hash": fingerprint(manifest),
        "sources": [{"source": source, "input_hash": manifest["input_hash"],
                     "version": manifest["version"]}],
        "statistics": read_json(root / "statistics.json"),
    }


def recommendation_identity(context, branch):
    value = read_state(context, branch).get("recommendation_hash")
    if value is None:
        raise ValueError(f"missing embedding provenance: {branch}; run embed-representations")
    from dataclasses import asdict
    from arm_registry import RECOMMENDATION_ARM_CONTRACT, feature_source, registry
    from validation.recommendation_contracts import ITEM_MODEL, ARCHITECTURE_VERSION, EVALUATION_CONTRACT
    from model_provenance import canonical, local_model_identity
    from validation.role_corruption import CORRUPTION

    arm = registry(context.config)[branch]
    experiment_arm = asdict(arm)
    if not arm.extensions:
        # Arms without extensions keep their earlier identity (and reusable results).
        experiment_arm.pop("extensions")
        experiment_arm.pop("segments")
    identity = {
        "arm_contract": RECOMMENDATION_ARM_CONTRACT,
        "evaluation_contract": dict(EVALUATION_CONTRACT),
        "architecture_version": ARCHITECTURE_VERSION,
        "encoder": {
            **context.config["validation"]["encoder"],
            "model": canonical(local_model_identity(context.path("models", "bge"))),
        },
        "item_model": {
            **ITEM_MODEL,
            "kind": arm.representation,
            "scene_mlp": branch != "base",
            "item_mlp": branch == "base",
            "description_projection_dim": 384 if arm.representation == "description" else None,
            "attention": {"query_key_input_dim": 384, "query_key_dim": 128,
                          "normalization": "unit_rms_then_scaled_dot_product_row_softmax",
                          "layers": 1, "value_dim": 512,
                          "residual": True, "activation": "relu", "pooling": "mean"}
            if arm.aggregation != "mean" else None,
        },
        "experiment_arm": experiment_arm,
        "embedding_hash": value,
        "representation_mode": arm.representation,
        "scene_aggregation": arm.aggregation if branch != "base" else None,
        "role_corruption": dict(CORRUPTION) if branch == "a2" else None,
    }
    if arm.extensions:
        from validation.graph_context import EXTENSION_MODELS

        identity["item_model"]["extensions"] = {e: EXTENSION_MODELS[e] for e in sorted(arm.extensions)}
    if "segments" in arm.extensions:
        from validation.graph_inputs import scene_segments_identity

        identity["scene_segments"] = scene_segments_identity(
            context.representations_dir / feature_source(arm), arm.segments
        )
    return identity
