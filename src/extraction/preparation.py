from __future__ import annotations

from typing import Any

from extraction.data_preparation.fixed30 import visual_evidence_matches
from extraction.data_preparation.microlens import prepare_catalog
from extraction.evidence_reuse import (
    copy_matching_evidence,
    donor_inventory,
    evidence_paths,
    source_matches_inventory,
)
from extraction.errors import ExtractionStepError
from extraction.step_support import result, visual_rows
from pipeline_runtime import RunContext


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
    cohort = context.require_ready_cohort()
    catalog = cohort["catalog"]
    settings = context.config["extraction"]["visual_evidence"]
    image_size = tuple(settings["image_resolution"])
    donors = donor_inventory(donor.run_root) if donor is not None else {}
    pending = []
    reused_target = reused_donor = 0
    for item, inventory in zip(catalog, cohort["inventory"], strict=True):
        if not source_matches_inventory(inventory):
            raise ExtractionStepError(
                f"source video changed or is missing after prepare-cohort: {item['content_id']}; "
                "repair missing assets and rerun prepare-cohort; changed inputs need a new run_id"
            )
        timestamp, frames = evidence_paths(context.run_root, item["content_id"])
        if any(
            not path.resolve().is_relative_to(context.run_root.resolve())
            for path in (timestamp, frames)
        ):
            raise ExtractionStepError("evidence destination must remain inside the target run")
        if not force and visual_evidence_matches(
            timestamp, frames, image_size, item["duration_seconds"]
        ):
            reused_target += 1
        elif donor is not None and copy_matching_evidence(
            target_root=context.run_root,
            donor_root=donor.run_root,
            current=inventory,
            donor=donors.get(item["item_id"]),
            image_size=image_size,
        ):
            reused_donor += 1
        else:
            pending.append(item)
    if pending:
        prepared = prepare_catalog(
            pending,
            assets_root=context.cohort_dir / "source_assets",
            output_root=context.run_root,
            image_size=image_size,
            force=force,
        )
        if prepared["failed"] or prepared["succeeded"] != len(pending):
            raise ExtractionStepError(f"visual evidence preparation is incomplete: {prepared}")
    else:
        (context.cohort_dir / "preparation_failures.jsonl").unlink(missing_ok=True)
    for item in catalog:
        timestamp, frames = evidence_paths(context.run_root, item["content_id"])
        if not visual_evidence_matches(timestamp, frames, image_size, item["duration_seconds"]):
            raise ExtractionStepError(f"invalid prepared visual evidence for {item['content_id']}")
    rows = visual_rows(context)
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
