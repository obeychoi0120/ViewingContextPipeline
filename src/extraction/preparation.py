from __future__ import annotations

import math
from typing import Any

from extraction.data_preparation.fixed30 import visual_evidence_matches
from extraction.data_preparation.media import cached_duration, save_duration
from extraction.data_preparation.microlens import prepare_catalog
from extraction.evidence_reuse import (
    SOURCE_IDENTITY_KEYS,
    copy_matching_evidence,
    donor_inventory,
    evidence_paths,
    source_matches_inventory,
)
from extraction.errors import ExtractionStepError
from extraction.step_support import result, visual_rows
from pipeline_runtime import RunContext, write_json
from visual_sampling import build_fixed_windows


def prepare_input_data(
    context: RunContext,
    *,
    force: bool = False,
    reuse_run_id: str | None = None,
) -> dict[str, Any]:
    donor = None
    if reuse_run_id is not None:
        if force:
            raise ExtractionStepError("--reuse-run-id cannot be combined with --force")
        donor = RunContext.load(reuse_run_id, root=context.root)
        if donor.run_root.resolve() == context.run_root.resolve():
            raise ExtractionStepError("--reuse-run-id must name a different run")
    context.initialize()
    print("[PREPARE INPUT] Loading prepared cohort...", flush=True)
    cohort = context.require_ready_cohort()
    catalog = cohort["catalog"]
    assets_root = context.cohort_dir / "source_assets"
    if context.config["schema_version"] == "viewing-context-config/v4":
        (context.cohort_dir / "media_preflight.json").unlink(missing_ok=True)
    settings = context.config["extraction"]["visual_evidence"]
    image_size = tuple(settings["image_resolution"])
    sampling = {key: settings[key] for key in ("scene_duration", "num_keyframes")}
    donors = donor_inventory(donor.run_root) if donor is not None else {}
    pending = []
    reused_target = reused_donor = 0
    print(
        f"[PREPARE INPUT] Checking sources and reusable keyframes for {len(catalog)} videos...",
        flush=True,
    )
    for item, inventory in zip(catalog, cohort["inventory"], strict=True):
        if not source_matches_inventory(inventory):
            raise ExtractionStepError(
                f"source video changed or is missing after prepare-cohort: {item['content_id']}; "
                "repair missing assets and rerun prepare-cohort; changed inputs need a new run_id"
            )
        timestamp, frames = evidence_paths(
            context.run_root, item["content_id"], sampling["scene_duration"],
        )
        if any(
            not path.resolve().is_relative_to(context.run_root.resolve())
            for path in (timestamp, frames)
        ):
            raise ExtractionStepError("evidence destination must remain inside the target run")
        duration = cached_duration(assets_root, inventory)
        donor_row = donors.get(item["item_id"])
        if donor is not None and donor_row is not None:
            donor_row = dict(donor_row)
            donor_row["duration_seconds"] = cached_duration(
                donor.cohort_dir / "source_assets", donor_row,
            )
            if (
                duration is None
                and donor_row.get("eligible") is True
                and donor_row["duration_seconds"] is not None
                and all(donor_row.get(key) == inventory[key] for key in SOURCE_IDENTITY_KEYS)
            ):
                duration = donor_row["duration_seconds"]
                save_duration(assets_root, inventory, duration)
        item["duration_seconds"] = inventory["duration_seconds"] = duration
        if not force and visual_evidence_matches(
            timestamp, frames, image_size, item["duration_seconds"], **sampling,
        ):
            reused_target += 1
        elif donor is not None and copy_matching_evidence(
            target_root=context.run_root,
            donor_root=donor.run_root,
            current=inventory,
            donor=donor_row,
            image_size=image_size,
            **sampling,
        ):
            reused_donor += 1
        else:
            pending.append({**inventory, **item})
    print(
        f"[PREPARE INPUT] reused_target={reused_target} reused_donor={reused_donor} "
        f"pending={len(pending)}",
        flush=True,
    )
    if pending:
        print(
            f"[PREPARE INPUT] Probing durations as needed and extracting resized keyframes: "
            f"{len(pending)} videos, "
            f"{image_size[0]}x{image_size[1]}, scene={sampling['scene_duration']}s, "
            f"up to {sampling['num_keyframes']} frames/scene...",
            flush=True,
        )
        prepared = prepare_catalog(
            pending,
            assets_root=assets_root,
            output_root=context.run_root,
            image_size=image_size,
            **sampling,
            force=force,
        )
        if prepared["failed"] or prepared["succeeded"] != len(pending):
            raise ExtractionStepError(f"visual evidence preparation is incomplete: {prepared}")
    else:
        (context.cohort_dir / "preparation_failures.jsonl").unlink(missing_ok=True)
    print("[PREPARE INPUT] Verifying prepared timestamps and images...", flush=True)
    for item, inventory in zip(catalog, cohort["inventory"], strict=True):
        item["duration_seconds"] = cached_duration(assets_root, inventory)
        timestamp, frames = evidence_paths(
            context.run_root, item["content_id"], sampling["scene_duration"],
        )
        if not visual_evidence_matches(
            timestamp, frames, image_size, item["duration_seconds"], **sampling,
        ):
            raise ExtractionStepError(f"invalid prepared visual evidence for {item['content_id']}")
    rows = visual_rows(context)
    if context.config["schema_version"] == "viewing-context-config/v4":
        _write_media_preflight(context, catalog, cohort["inventory"])
    print(
        f"[EVIDENCE] reused_target={reused_target} reused_donor={reused_donor} "
        f"extracted={len(pending)}",
        flush=True,
    )
    return {
        **result("prepare-input-data", content_count=len(rows)),
        "reused_target": reused_target,
        "reused_donor": reused_donor,
        "extracted": len(pending),
    }


def _write_media_preflight(context: RunContext, catalog: list, inventory: list) -> None:
    sampling = context.config["extraction"]["visual_evidence"]
    scene_count = frame_count = 0
    for row in catalog:
        windows = build_fixed_windows(
            row["duration_seconds"],
            scene_duration=sampling["scene_duration"],
            num_keyframes=sampling["num_keyframes"],
        )
        scene_count += len(windows)
        frame_count += sum(len(w["keyframe_timestamps"]) for w in windows)
    write_json(context.cohort_dir / "media_preflight.json", {
        "video_count": len(catalog),
        "duration_seconds": sum(r["duration_seconds"] for r in catalog),
        "source_bytes": sum(r["source_file_size"] for r in inventory),
        "scene_count": scene_count,
        "keyframe_count": frame_count,
        "uncompressed_rgb_bytes": frame_count * math.prod(sampling["image_resolution"]) * 3,
        "storage_note": "RGB payload estimate; PNG size and extraction outputs are additional/variable",
        "sampling": sampling,
    })
    print(
        f"[PREPARE INPUT] Media report saved: videos={len(catalog)} "
        f"scenes={scene_count} keyframes={frame_count}",
        flush=True,
    )
