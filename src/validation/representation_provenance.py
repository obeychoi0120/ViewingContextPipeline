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

    arm = registry(context.config)[branch]
    from rec_experiments import experiment_definition

    experiment_arm = asdict(arm)
    experiment_arm["extensions"] = list(arm.extensions)
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
            "scene_mlp": arm.representation != "metadata",
            "item_mlp": arm.representation == "metadata",
            "description_projection_dim": 384 if arm.representation == "description" else None,
            "role_fusion": arm.role_fusion,
            "scene_mlp_config": {"input_dim": 512, "output_dim": 512,
                                 "formula": "x_plus_relu_linear_x"} if arm.representation != "metadata" else None,
        },
        "experiment_arm": experiment_arm,
        "experiment_definition": experiment_definition(arm),
        "embedding_hash": value,
        "representation_mode": arm.representation,
        "scene_aggregation": arm.aggregation if arm.representation != "metadata" else None,
    }
    if arm.extensions:
        from validation.graph_context import EXTENSION_MODELS

        identity["item_model"]["extensions"] = {e: EXTENSION_MODELS[e] for e in sorted(arm.extensions)}
    if "segments" in arm.extensions:
        from validation.graph_inputs import scene_segments_identity

        identity["scene_segments"] = scene_segments_identity(
            context.representations_dir / feature_source(arm)
        )
    return identity
