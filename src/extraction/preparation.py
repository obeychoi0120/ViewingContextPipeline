from __future__ import annotations

from typing import Any

from extraction.data_preparation.microlens import prepare_catalog
from extraction.errors import ExtractionStepError
from extraction.step_support import require_file, result, visual_rows
from pipeline_runtime import RunContext, read_jsonl


def prepare_input_data(
    context: RunContext,
    *,
    force: bool = False,
) -> dict[str, Any]:
    context.initialize()
    if not force:
        try:
            rows = visual_rows(context)
        except ExtractionStepError:
            pass
        else:
            return result("prepare-input-data", content_count=len(rows))
    catalog_path = require_file(context.cohort_dir / "catalog.jsonl", "cohort catalog")
    catalog = read_jsonl(catalog_path)
    settings = context.config["extraction"]["visual_evidence"]
    prepared = prepare_catalog(
        catalog,
        assets_root=context.cohort_dir / "source_assets",
        output_root=context.run_root,
        image_size=tuple(settings["image_resolution"]),
        force=force,
    )
    if prepared["failed"] or prepared["succeeded"] != len(catalog):
        raise ExtractionStepError(
            f"visual evidence preparation is incomplete: {prepared}"
        )
    rows = visual_rows(context)
    return result("prepare-input-data", content_count=len(rows))
