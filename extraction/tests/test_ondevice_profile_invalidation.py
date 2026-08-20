from __future__ import annotations

from src.video_data_collection.raw_pipeline import (
    build_content_paths,
    invalidate_pts_dependent_artifacts,
)


def test_pts_invalidation_removes_model_specific_ondevice_contexts_and_profile(tmp_path) -> None:
    output_root = tmp_path / "output"
    paths = build_content_paths(
        tmp_path / "assets",
        "demo",
        output_root=output_root,
    )
    scene_contexts = [
        output_root / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_qwen" / "demo_scene_context.jsonl",
        output_root / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_mistral" / "demo_scene_context.jsonl",
        output_root / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_gaussa_gemma4_e2b_v0_3" / "demo_scene_context.jsonl",
    ]
    profiles = [
        output_root / "viewing_context" / "img_only" / "fixed_15s" / "video_context_graph_qwen" / "demo_context_graph_ond.json",
        output_root / "viewing_context" / "img_only" / "fixed_15s" / "video_context_graph_mistral" / "demo_context_graph_ond.json",
        output_root / "viewing_context" / "img_only" / "fixed_15s" / "video_context_graph_gaussa_gemma4_e2b_v0_3" / "demo_context_graph_ond.json",
    ]
    failures = [
        output_root / "failures" / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_qwen" / "demo_failures.jsonl",
        output_root / "failures" / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_mistral" / "demo_failures.jsonl",
        output_root / "failures" / "viewing_context" / "img_only" / "fixed_15s" / "scene_context_graph_gaussa_gemma4_e2b_v0_3" / "demo_failures.jsonl",
    ]
    vp_ref = output_root / "video_profile" / "fixed_15s" / "demo_profile.json"
    for scene_context in scene_contexts:
        scene_context.parent.mkdir(parents=True)
        scene_context.write_text("{}\n", encoding="utf-8")
    for profile in profiles:
        profile.parent.mkdir(parents=True)
        profile.write_text("{}\n", encoding="utf-8")
    for failure in failures:
        failure.parent.mkdir(parents=True)
        failure.write_text("{}\n", encoding="utf-8")
    vp_ref.parent.mkdir(parents=True)
    vp_ref.write_text("{}\n", encoding="utf-8")

    invalidate_pts_dependent_artifacts(paths)

    assert not any(scene_context.exists() for scene_context in scene_contexts)
    assert not any(profile.exists() for profile in profiles)
    assert not any(failure.exists() for failure in failures)
    assert not vp_ref.exists()
