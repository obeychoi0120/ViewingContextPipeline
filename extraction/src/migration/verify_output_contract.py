from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("fixed_15s", "fixed_30s", "shot_wise")
TRACKS = ("img_only", "multimodal")
SOURCES = ("ref", "qwen", "mistral", "gaussa_gemma4_e2b_v0_3")


def _file_ids(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.name[: -len(suffix)]
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    }


def build_inventory(output_root: Path) -> dict[str, object]:
    metadata_ids = _file_ids(output_root / "asset" / "metadata", ".json")
    modes: dict[str, object] = {}
    for mode in MODES:
        asset_root = output_root / "asset" / mode
        ref_ids = _file_ids(asset_root / "ref_jsonl", "_ref.jsonl")
        frame_ids = {
            path.name
            for path in (asset_root / "resized_keyframes").iterdir()
            if path.is_dir()
        } if (asset_root / "resized_keyframes").is_dir() else set()
        tracks: dict[str, object] = {}
        for track in TRACKS:
            root = output_root / "viewing_context" / track / mode
            tracks[track] = {
                source: {
                    "scene_context_count": len(
                        _file_ids(root / f"scene_context_graph_{source}", "_scene_context.jsonl")
                        | _file_ids(root / f"scene_context_graph_{source}", "_scene_context_ref.jsonl")
                    ),
                    "video_context_count": len(
                        _file_ids(root / f"video_context_graph_{source}", "_context_graph_ref.json")
                        | _file_ids(root / f"video_context_graph_{source}", "_context_graph_ond.json")
                    ),
                }
                for source in SOURCES
            }
        modes[mode] = {
            "ref_jsonl_count": len(ref_ids),
            "resized_keyframe_content_count": len(frame_ids),
            "video_profile_count": len(
                _file_ids(output_root / "video_profile" / mode, "_profile.json")
            ),
            "metadata_without_ref_jsonl": sorted(metadata_ids - ref_ids),
            "ref_jsonl_without_keyframes": sorted(ref_ids - frame_ids),
            "keyframes_without_ref_jsonl": sorted(frame_ids - ref_ids),
            "tracks": tracks,
        }

    forbidden: list[str] = []
    legacy_directories = {
        "video_profile_gt",
        "video_profile_ref",
        "video_profile_graph_ref",
        "video_profile_graph_ref_canonical",
        "video_profile_graph_qwen",
        "video_profile_graph_mistral",
        "video_profile_graph_gauss_gemma4_e2b",
    }
    for path in output_root.rglob("*"):
        if path.is_dir() and path.name in legacy_directories:
            forbidden.append(str(path))
        elif path.is_file() and any(
            token in path.name
            for token in ("_profile_graph_", "_profile_ref.json", "_profile_gt.json")
        ):
            forbidden.append(str(path))
    for path in (output_root / "metadata",):
        if path.exists():
            forbidden.append(str(path))
    for mode in MODES:
        legacy_mode = output_root / "viewing_context" / mode
        if legacy_mode.exists():
            forbidden.append(str(legacy_mode))
    return {
        "schema_version": "canonical-output-inventory/v1",
        "metadata_count": len(metadata_ids),
        "modes": modes,
        "forbidden_legacy_paths": sorted(set(forbidden)),
        "valid": not forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory and validate the canonical output tree.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    inventory = build_inventory(output_root)
    report = output_root / "reports" / "migration" / "canonical_inventory.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0 if inventory["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
