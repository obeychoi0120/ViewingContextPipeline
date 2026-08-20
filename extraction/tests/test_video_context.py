from __future__ import annotations

import json

from src.scene_context_extraction.graph_core.video_context import (
    aggregate_scene_context,
    video_context_is_valid,
    write_video_context,
)


def test_aggregate_scene_context_matches_legacy_summary_shape() -> None:
    profile, warnings = aggregate_scene_context(
        "demo",
        [{"scene_idx": 0, "vlm_visual_graph": {}}],
    )

    assert profile == {
        "content_axes_4d": {
            "subject_sociality": 0.0,
            "media_syntheticity": 0.0,
            "setting_context": 0.0,
            "utility_orientation": 0.0,
        },
        "content_axis_distribution": {
            "subject_sociality": {"neutral": 1.0},
            "media_syntheticity": {"neutral": 1.0},
            "setting_context": {"neutral": 1.0},
            "utility_orientation": {"neutral": 1.0},
        },
        "top_styles": [{"id": "style:mixed", "count": 1}],
        "top_moods": [{"id": "mood:neutral", "count": 1}],
        "top_scene_functions": [{"id": "scene_function:unknown", "count": 1}],
        "top_entities": [],
        "top_motifs": [],
    }
    assert warnings
    assert all(warning.startswith("scene_000: ") for warning in warnings)


def test_write_video_context_uses_exact_wrapper_and_skips_null_graphs(tmp_path) -> None:
    source = tmp_path / "scene_context.jsonl"
    source.write_text(
        json.dumps({"scene_idx": 0, "vlm_visual_graph": None}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "video_context_graph_ref" / "demo_context_graph_ref.json"

    document = write_video_context("demo", source, output)

    assert list(document) == [
        "content_id",
        "source_scene_context_path",
        "context",
        "aggregation_warnings",
    ]
    assert document["context"]["top_entities"] == []
    assert video_context_is_valid(
        output,
        content_id="demo",
        source_scene_context_path=source,
    )

    source.write_text(
        json.dumps({"scene_idx": 0, "vlm_visual_graph": {}}) + "\n",
        encoding="utf-8",
    )
    assert not video_context_is_valid(
        output,
        content_id="demo",
        source_scene_context_path=source,
    )
