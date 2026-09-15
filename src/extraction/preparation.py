from __future__ import annotations

from extraction.data_preparation.fixed30 import visual_evidence_matches
from extraction.data_preparation.media import cached_duration
from extraction.data_preparation.microlens import prepare_catalog
from extraction.evidence_reuse import source_matches_inventory
from extraction.errors import ExtractionStepError
from extraction.step_support import result
from pipeline_logging import log_step_start
from visual_sampling import build_fixed_windows


def prepare_input_data(context, *, force=False):
    log_step_start(context, "prepare-input-data", force=force)
    context.initialize()
    cohort = context.require_ready_cohort()
    assets_root = context.cohort_dir / "source_assets"
    settings = context.config["extraction"]["visual_evidence"]
    image_size = tuple(settings["image_resolution"])
    sampling = {key: settings[key] for key in ("scene_duration", "num_keyframes")}
    pending = []
    for item, inventory in zip(cohort["catalog"], cohort["inventory"], strict=True):
        if not source_matches_inventory(inventory):
            raise ExtractionStepError(f"source changed since prepare-cohort: {item['content_id']}")
        cid = str(item["content_id"])
        timestamp = (
            assets_root / cid / "assets" / f"timestamp_fixed_{sampling['scene_duration']}s.json"
        )
        frames = context.keyframes_dir / cid
        duration = cached_duration(assets_root, inventory)
        if force or not visual_evidence_matches(
            timestamp, frames, image_size, duration, **sampling
        ):
            pending.append({**inventory, **item, "duration_seconds": duration})
    prepared = {"reused_frames": 0, "extracted_frames": 0}
    if pending:
        prepared = prepare_catalog(
            pending,
            assets_root=assets_root,
            output_root=context.evidence_dir,
            image_size=image_size,
            **sampling,
            force=force,
        )
        if prepared["failed"]:
            raise ExtractionStepError(f"visual evidence preparation incomplete: {prepared}")
    else:
        (context.cohort_dir / "preparation_failures.jsonl").unlink(missing_ok=True)
    scene_count = frame_count = 0
    for item, inventory in zip(cohort["catalog"], cohort["inventory"], strict=True):
        cid = str(item["content_id"])
        duration = cached_duration(assets_root, inventory)
        timestamp = (
            assets_root / cid / "assets" / f"timestamp_fixed_{sampling['scene_duration']}s.json"
        )
        if not visual_evidence_matches(
            timestamp, context.keyframes_dir / cid, image_size, duration, **sampling
        ):
            raise ExtractionStepError(f"invalid visual evidence for {cid}")
        windows = build_fixed_windows(duration, **sampling)
        scene_count += len(windows)
        frame_count += sum(len(w["keyframe_timestamps"]) for w in windows)
    print(
        f"[EVIDENCE] videos={len(cohort['catalog'])} scenes={scene_count} keyframes={frame_count} "
        f"prepared_videos={len(pending)} reused_frames={frame_count - prepared['extracted_frames']} "
        f"new_frames={prepared['extracted_frames']} shared_frames={context.keyframes_dir}",
        flush=True,
    )
    return result("prepare-input-data", content_count=len(cohort["catalog"]))
