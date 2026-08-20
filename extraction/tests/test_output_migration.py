from __future__ import annotations

import json

from src.migration.migrate_output_contract import (
    apply_migration,
    build_items,
    preflight,
)
from src.migration.verify_output_contract import build_inventory


def _write_graph(path, content_id: str, value: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "content_id": content_id,
                "source_scene_context_path": "legacy.jsonl",
                "profile": {"content_axes_4d": {"subject_sociality": value}},
                "aggregation_warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_dry_run_preflight_does_not_change_files(tmp_path) -> None:
    output = tmp_path / "output"
    source = output / "viewing_context" / "fixed_15s" / "video_profile_graph_ref"
    graph = source / "demo_profile_graph_ref.json"
    _write_graph(graph, "demo")
    before = graph.read_bytes()

    report = preflight(build_items(output))

    assert report["errors"] == []
    assert graph.read_bytes() == before
    assert not (
        output
        / "viewing_context"
        / "img_only"
        / "fixed_15s"
        / "video_context_graph_ref"
        / "demo_context_graph_ref.json"
    ).exists()


def test_conflicting_duplicates_block_all_changes(tmp_path) -> None:
    output = tmp_path / "output"
    first = output / "viewing_context" / "fixed_15s" / "video_profile_graph_ref"
    second = output / "viewing_context" / "fixed_15s" / "video_profile_graph_ref_canonical"
    _write_graph(first / "demo_profile_graph_ref.json", "demo", 0.0)
    _write_graph(second / "demo_profile_graph_ref.json", "demo", 1.0)
    items = build_items(output)
    report = preflight(items)

    assert any("conflicting duplicate target" in error for error in report["errors"])
    try:
        apply_migration(items, report)
    except ValueError as exc:
        assert "preflight failed" in str(exc)
    else:
        raise AssertionError("conflicting migration unexpectedly applied")
    assert first.is_dir()
    assert second.is_dir()


def test_apply_rewrites_graph_wrapper_and_provenance(tmp_path) -> None:
    output = tmp_path / "output"
    source = output / "viewing_context" / "fixed_15s" / "video_profile_graph_ref"
    _write_graph(source / "demo_profile_graph_ref.json", "demo")
    items = build_items(output)
    report = preflight(items)

    apply_migration(items, report)

    target = (
        output
        / "viewing_context"
        / "img_only"
        / "fixed_15s"
        / "video_context_graph_ref"
        / "demo_context_graph_ref.json"
    )
    document = json.loads(target.read_text(encoding="utf-8"))
    assert set(document) == {
        "content_id",
        "source_scene_context_path",
        "context",
        "aggregation_warnings",
    }
    assert "viewing_context\\img_only\\fixed_15s" in document["source_scene_context_path"] or "viewing_context/img_only/fixed_15s" in document["source_scene_context_path"]
    assert build_inventory(output)["valid"] is True
