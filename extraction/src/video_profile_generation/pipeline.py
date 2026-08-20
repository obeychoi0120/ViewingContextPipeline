from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypeVar

from google.genai import types
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.common.gemini import make_extraction_config

from .config import VideoProfileConfig
from .inputs import InputError, load_content_bundle, load_manifest_content_ids
from .models import (
    VideoProfileDetails,
    VideoProfileDocument,
    VideoProfileSummary,
    build_video_profile_details_model,
)
from .ontology import (
    OntologyContractError,
    ViewingOntologyContract,
    load_viewing_ontology_contract,
    validate_profile_details,
)
from .prompt import (
    DETAILS_SYSTEM_INSTRUCTION,
    SUMMARY_SYSTEM_INSTRUCTION,
    build_details_contents,
    build_summary_contents,
)


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
CONTENT_WORKERS = 4


class PipelineError(RuntimeError):
    """Raised for pipeline-level failures that cannot be assigned to one video."""


@dataclass
class PipelineSummary:
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: dict[str, str] = field(default_factory=dict)


def run_pipeline(
    *,
    client: Any,
    config: VideoProfileConfig,
    project_root: Path,
    manifest_path: str | Path | None = None,
    content_ids: Iterable[str] | None = None,
    overwrite: bool = False,
) -> PipelineSummary:
    try:
        manifest_ids = load_manifest_content_ids(
            manifest_path
            or project_root / "contracts" / "manifest.csv"
        )
    except InputError as exc:
        raise PipelineError(f"invalid input manifest: {exc}") from exc
    if content_ids is None:
        selected_ids = manifest_ids
    else:
        selected_ids = list(dict.fromkeys(content_ids))
        unknown_ids = [
            content_id
            for content_id in selected_ids
            if content_id not in manifest_ids
        ]
        if unknown_ids:
            raise PipelineError(
                "content IDs are not present in the input manifest: "
                + ", ".join(unknown_ids)
            )
    if not selected_ids:
        raise PipelineError("no content IDs selected")

    local_output_dir = config.resolve_local_output_dir(project_root)
    local_input_dir = config.resolve_local_input_dir(project_root)
    failure_dir = local_input_dir / "failures" / "video_profile" / config.shot_interval
    report_dir = local_input_dir / "reports" / "video_profile" / config.shot_interval
    try:
        ontology = load_viewing_ontology_contract(
            config.resolve_ontology_contract_path(project_root)
        )
    except OntologyContractError as exc:
        raise PipelineError(f"invalid viewing ontology contract: {exc}") from exc
    summary = PipelineSummary()
    details_response_model = build_video_profile_details_model(ontology.payload)
    summary_generate_config = make_extraction_config(
        system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
        response_schema=VideoProfileSummary,
        thinking_level=config.gemini_thinking_level,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    details_generate_config = make_extraction_config(
        system_instruction=DETAILS_SYSTEM_INSTRUCTION,
        response_schema=details_response_model,
        thinking_level=config.gemini_thinking_level,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )

    with (
        ThreadPoolExecutor(max_workers=CONTENT_WORKERS) as executor,
        tqdm(
            total=len(selected_ids),
            desc="Video profiles",
            unit="video",
            dynamic_ncols=True,
        ) as progress,
    ):
        futures = {
            executor.submit(
                _process_content,
                content_id=content_id,
                client=client,
                model=config.gemini_model,
                summary_generate_config=summary_generate_config,
                details_generate_config=details_generate_config,
                details_response_model=details_response_model,
                ontology=ontology,
                local_input_dir=local_input_dir,
                local_output_dir=local_output_dir,
                shot_interval=config.shot_interval,
                overwrite=overwrite,
            ): content_id
            for content_id in selected_ids
        }
        for future in as_completed(futures):
            content_id = futures[future]
            try:
                status = future.result()
                if status == "skipped":
                    summary.skipped += 1
                    tqdm.write(
                        f"[skip] {content_id}: valid local profile already exists"
                    )
                else:
                    summary.succeeded += 1
                    tqdm.write(f"[success] {content_id}")
                (failure_dir / f"{content_id}_failure.json").unlink(missing_ok=True)
            except Exception as exc:
                summary.failed += 1
                summary.errors[content_id] = str(exc)
                _write_text_atomic(
                    failure_dir / f"{content_id}_failure.json",
                    json.dumps(
                        {"content_id": content_id, "error": str(exc)},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )
                tqdm.write(f"[failed] {content_id}: {exc}")
            progress.update()
            progress.set_postfix(
                succeeded=summary.succeeded,
                skipped=summary.skipped,
                failed=summary.failed,
            )
    _write_text_atomic(
        report_dir / "generation_report.json",
        json.dumps(
            {
                "mode": config.shot_interval,
                "selected_count": len(selected_ids),
                "succeeded": summary.succeeded,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "errors": summary.errors,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    return summary


def _process_content(
    *,
    content_id: str,
    client: Any,
    model: str,
    summary_generate_config: types.GenerateContentConfig,
    details_generate_config: types.GenerateContentConfig,
    details_response_model: type[BaseModel],
    ontology: ViewingOntologyContract,
    local_input_dir: Path,
    local_output_dir: Path,
    shot_interval: str,
    overwrite: bool,
) -> str:
    local_path = local_output_dir / f"{content_id}_profile.json"

    if local_path.is_file() and not overwrite:
        existing_text = local_path.read_text(encoding="utf-8")
        try:
            existing = VideoProfileDocument.model_validate_json(existing_text)
        except ValidationError as exc:
            raise PipelineError(
                f"existing local profile is invalid; rerun with --overwrite: {exc}"
            ) from exc
        if existing.ontology_version != ontology.version:
            raise PipelineError(
                "existing local profile uses a different ontology; "
                "rerun with --overwrite"
            )
        try:
            validate_profile_details(
                VideoProfileDetails(
                    profile=existing.profile,
                    descriptive_context=existing.descriptive_context,
                ),
                ontology,
            )
        except OntologyContractError as exc:
            raise PipelineError(
                "existing local profile violates the configured ontology; "
                f"rerun with --overwrite: {exc}"
            ) from exc
        return "skipped"

    bundle = load_content_bundle(
        local_input_dir,
        content_id,
        shot_interval,
    )
    summary_response = client.models.generate_content(
        model=model,
        contents=build_summary_contents(
            bundle,
            ontology.payload,
        ),
        config=summary_generate_config,
    )
    profile_summary = _parse_response(
        summary_response,
        VideoProfileSummary,
        stage="summary",
    )
    details_response = client.models.generate_content(
        model=model,
        contents=build_details_contents(
            bundle,
            ontology.payload,
        ),
        config=details_generate_config,
    )
    generated_details = _parse_response(
        details_response,
        details_response_model,
        stage="details",
    )
    profile_details = VideoProfileDetails.model_validate(
        generated_details.model_dump(mode="json")
    )
    validate_profile_details(profile_details, ontology)
    document = VideoProfileDocument(
        title=bundle.metadata["title"],
        channel=bundle.metadata["channel"],
        upload_date=bundle.metadata["upload_date"],
        meta_desc=bundle.metadata["description"],
        summary=profile_summary.summary,
        profile=profile_details.profile,
        descriptive_context=profile_details.descriptive_context,
    )
    output_text = (
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n"
    )

    _write_text_atomic(local_path, output_text)
    return "succeeded"


def _parse_response(
    response: Any,
    response_model: type[ResponseModel],
    *,
    stage: str,
) -> ResponseModel:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise PipelineError(
                f"Gemini {stage} response failed validation: {exc}"
            ) from exc

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise PipelineError(f"Gemini returned no structured {stage} response")
    try:
        return response_model.model_validate_json(text)
    except ValidationError as exc:
        raise PipelineError(
            f"Gemini {stage} response failed validation: {exc}"
        ) from exc


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
