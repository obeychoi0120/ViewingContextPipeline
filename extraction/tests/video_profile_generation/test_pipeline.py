from __future__ import annotations

import json
from threading import Event, Lock
from types import SimpleNamespace

from google.genai import types
import pytest

from src.video_profile_generation.config import VideoProfileConfig
from src.video_profile_generation.models import (
    VideoProfileDocument,
    VideoProfileSummary,
)
from src.video_profile_generation import pipeline
from src.video_profile_generation.pipeline import PipelineError, run_pipeline

from .conftest import (
    CONTENT_ID,
    FakeModels,
    profile_payload,
    summary_payload,
    write_local_inputs,
    write_manifest,
    write_ontology_contract,
)


def make_config(
    tmp_path,
    shot_interval: str = "fixed_15s",
) -> VideoProfileConfig:
    write_manifest(tmp_path)
    return VideoProfileConfig(
        gcp_project_id="project-id",
        gemini_location="global",
        gemini_model="gemini-3.5-flash",
        gemini_thinking_level="medium",
        shot_interval=shot_interval,
        ontology_contract_path=write_ontology_contract(tmp_path),
        local_output_dir="output/video_profile",
    )


def test_pipeline_writes_only_local_output(
    tmp_path,
    input_objects: dict[str, str],
    fake_client,
) -> None:
    summary = run_pipeline(
        client=fake_client,
        config=make_config(tmp_path),
        project_root=tmp_path,
        manifest_path=write_manifest(tmp_path),
    )

    assert summary.succeeded == 1
    assert summary.failed == 0
    local_text = (
        tmp_path / "output/video_profile/fixed_15s" / f"{CONTENT_ID}_profile.json"
    ).read_text(encoding="utf-8")
    document = VideoProfileDocument.model_validate_json(local_text)
    assert document.title == "매일 사용하는 고효율 앱테크"
    assert document.channel == "예브라"
    assert document.meta_desc.endswith("#앱테크")
    assert document.summary == summary_payload()["summary"]
    assert document.ontology_version == "viewing-ontology-contract/v3"
    assert list(document.profile.model_dump()) == [
        "topic",
        "content_type",
        "intent",
        "media_form",
        "presentation",
    ]
    assert len(fake_client.models.calls) == 2
    summary_call, details_call = fake_client.models.calls
    assert summary_call["config"].response_schema is VideoProfileSummary
    assert details_call["config"].response_schema.__name__ == (
        "VideoProfileDetailsResponse"
    )
    assert (
        summary_call["config"].media_resolution
        == types.MediaResolution.MEDIA_RESOLUTION_LOW
    )
    assert (
        details_call["config"].media_resolution
        == types.MediaResolution.MEDIA_RESOLUTION_LOW
    )

    summary_parts = summary_call["contents"].parts
    details_parts = details_call["contents"].parts
    assert [part.model_dump() for part in summary_parts[:-1]] == [
        part.model_dump() for part in details_parts[:-1]
    ]
    assert (
        "VideoProfileSummary schema"
        in json.loads(summary_parts[-1].text)["instruction"]
    )
    assert (
        "VideoProfileDetails schema"
        in json.loads(details_parts[-1].text)["instruction"]
    )


def test_shot_wise_pipeline_scopes_local_inputs_and_output(
    tmp_path,
    input_objects: dict[str, str],
    fake_client,
) -> None:
    shot_wise_objects = {
        name.replace("/fixed_15s/", "/shot_wise/"): value
        for name, value in input_objects.items()
    }
    write_local_inputs(tmp_path, shot_wise_objects)
    summary = run_pipeline(
        client=fake_client,
        config=make_config(tmp_path, "shot_wise"),
        project_root=tmp_path,
        manifest_path=write_manifest(tmp_path),
    )

    output_path = (
        tmp_path
        / "output/video_profile/shot_wise"
        / f"{CONTENT_ID}_profile.json"
    )
    image_parts = [
        part.inline_data
        for call in fake_client.models.calls
        for part in call["contents"].parts
        if part.inline_data is not None
    ]
    assert summary.succeeded == 1
    assert output_path.is_file()
    assert len(image_parts) == 4
    assert all(part.data == b"image" for part in image_parts)


def test_pipeline_skips_valid_existing_local_profile(
    tmp_path,
    input_objects: dict[str, str],
) -> None:
    existing = {
        "ontology_version": "viewing-ontology-contract/v3",
        "title": "매일 사용하는 고효율 앱테크",
        "channel": "예브라",
        "upload_date": "20260701",
        "meta_desc": "설명",
        "summary": summary_payload()["summary"],
        "profile": profile_payload()["profile"],
        "descriptive_context": profile_payload()["descriptive_context"],
    }
    local_path = (
            tmp_path / "output/video_profile/fixed_15s" / f"{CONTENT_ID}_profile.json"
    )
    local_path.parent.mkdir(parents=True)
    existing_text = json.dumps(existing, ensure_ascii=False)
    local_path.write_text(existing_text, encoding="utf-8")
    models = FakeModels(SimpleNamespace(parsed=profile_payload()))

    summary = run_pipeline(
        client=SimpleNamespace(models=models),
        config=make_config(tmp_path),
        project_root=tmp_path,
        content_ids=[CONTENT_ID],
    )

    assert summary.skipped == 1
    assert models.calls == []
    assert local_path.read_text(encoding="utf-8") == existing_text


def test_pipeline_rejects_existing_profile_with_unknown_concept(
    tmp_path,
    input_objects: dict[str, str],
) -> None:
    invalid_profile = profile_payload()
    invalid_profile["profile"]["topic"][0]["concept_id"] = "unknown/topic"
    existing = {
        "ontology_version": "viewing-ontology-contract/v3",
        "title": "매일 사용하는 고효율 앱테크",
        "channel": "예브라",
        "upload_date": "20260701",
        "meta_desc": "설명",
        "summary": summary_payload()["summary"],
        "profile": invalid_profile["profile"],
        "descriptive_context": invalid_profile["descriptive_context"],
    }
    local_path = (
            tmp_path / "output/video_profile/fixed_15s" / f"{CONTENT_ID}_profile.json"
    )
    local_path.parent.mkdir(parents=True)
    local_path.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
    models = FakeModels()

    summary = run_pipeline(
        client=SimpleNamespace(models=models),
        config=make_config(tmp_path),
        project_root=tmp_path,
        content_ids=[CONTENT_ID],
    )

    assert summary.failed == 1
    assert "violates the configured ontology" in summary.errors[CONTENT_ID]
    assert models.calls == []


@pytest.mark.parametrize("invalid_case", ["concept", "enum", "cardinality"])
def test_pipeline_rejects_generated_details_outside_contract(
    tmp_path,
    input_objects: dict[str, str],
    invalid_case: str,
) -> None:
    invalid_details = profile_payload()
    if invalid_case == "concept":
        invalid_details["profile"]["topic"][0]["concept_id"] = "unknown/topic"
    elif invalid_case == "enum":
        invalid_details["profile"]["media_form"] = ["unknown-form"]
    else:
        invalid_details["profile"]["topic"] = [
            {
                "concept_id": "products/consumption",
                "extension_label": f"표현 {index}",
            }
            for index in range(7)
        ]
    models = FakeModels(
        SimpleNamespace(parsed=summary_payload(), text=None),
        SimpleNamespace(parsed=invalid_details, text=None),
    )

    summary = run_pipeline(
        client=SimpleNamespace(models=models),
        config=make_config(tmp_path),
        project_root=tmp_path,
        content_ids=[CONTENT_ID],
    )

    assert summary.failed == 1
    assert "details response failed validation" in summary.errors[CONTENT_ID]


def test_pipeline_continues_after_one_content_failure(
    tmp_path,
    input_objects: dict[str, str],
    fake_client,
) -> None:

    summary = run_pipeline(
        client=fake_client,
        config=make_config(tmp_path),
        project_root=tmp_path,
        manifest_path=write_manifest(
            tmp_path,
            content_ids=["missing-content", CONTENT_ID],
        ),
        content_ids=["missing-content", CONTENT_ID],
    )

    assert summary.failed == 1
    assert summary.succeeded == 1
    assert "missing-content" in summary.errors


def test_pipeline_rejects_content_id_outside_manifest(
    tmp_path,
    input_objects: dict[str, str],
    fake_client,
) -> None:
    with pytest.raises(PipelineError, match="not present in the input manifest"):
        run_pipeline(
            client=fake_client,
            config=make_config(tmp_path),
            project_root=tmp_path,
            content_ids=["outside-contract"],
        )


def test_pipeline_processes_four_contents_and_assigns_next_on_completion(
    tmp_path,
    input_objects: dict[str, str],
    monkeypatch,
) -> None:
    content_ids = [f"content-{index}" for index in range(5)]
    initial_workers_started = Event()
    release_waiting_workers = Event()
    replacement_started = Event()
    lock = Lock()
    started: list[str] = []
    active = 0
    max_active = 0

    def fake_process_content(**kwargs) -> str:
        nonlocal active, max_active
        content_id = kwargs["content_id"]
        with lock:
            started.append(content_id)
            start_number = len(started)
            active += 1
            max_active = max(max_active, active)
            if active == 4:
                initial_workers_started.set()
        try:
            if start_number == 1:
                assert initial_workers_started.wait(timeout=2)
            elif start_number <= 4:
                assert release_waiting_workers.wait(timeout=2)
            else:
                replacement_started.set()
                release_waiting_workers.set()
            return "succeeded"
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(pipeline, "_process_content", fake_process_content)

    summary = run_pipeline(
        client=SimpleNamespace(),
        config=make_config(tmp_path),
        project_root=tmp_path,
        manifest_path=write_manifest(tmp_path, content_ids=content_ids),
        content_ids=content_ids,
    )

    assert replacement_started.is_set()
    assert max_active == 4
    assert started == content_ids
    assert summary.succeeded == 5
    assert summary.failed == 0


def test_overwrite_regenerates_existing_profile(
    tmp_path,
    input_objects: dict[str, str],
    fake_client,
) -> None:
    local_path = (
            tmp_path / "output/video_profile/fixed_15s" / f"{CONTENT_ID}_profile.json"
    )
    local_path.parent.mkdir(parents=True)
    local_path.write_text('{"obsolete":true}', encoding="utf-8")

    summary = run_pipeline(
        client=fake_client,
        config=make_config(tmp_path),
        project_root=tmp_path,
        content_ids=[CONTENT_ID],
        overwrite=True,
    )

    assert summary.succeeded == 1
    assert len(fake_client.models.calls) == 2
    regenerated = VideoProfileDocument.model_validate_json(
        local_path.read_text(encoding="utf-8")
    )
    assert regenerated.channel == "예브라"


def test_summary_failure_stops_before_details_and_writes_nothing(
    tmp_path,
    input_objects: dict[str, str],
) -> None:
    models = FakeModels(SimpleNamespace(parsed={"summary": ""}, text=None))

    summary = run_pipeline(
        client=SimpleNamespace(models=models),
        config=make_config(tmp_path),
        project_root=tmp_path,
        content_ids=[CONTENT_ID],
    )

    assert summary.failed == 1
    assert "summary response failed validation" in summary.errors[CONTENT_ID]
    assert len(models.calls) == 1
    assert not (
        tmp_path
        / "output/video_profile/fixed_15s"
        / f"{CONTENT_ID}_profile.json"
    ).exists()


def test_details_failure_writes_no_partial_profile(
    tmp_path,
    input_objects: dict[str, str],
) -> None:
    models = FakeModels(
        SimpleNamespace(parsed=summary_payload(), text=None),
        SimpleNamespace(parsed={"profile": {"topic": []}}, text=None),
    )

    summary = run_pipeline(
        client=SimpleNamespace(models=models),
        config=make_config(tmp_path),
        project_root=tmp_path,
        content_ids=[CONTENT_ID],
    )

    assert summary.failed == 1
    assert "details response failed validation" in summary.errors[CONTENT_ID]
    assert len(models.calls) == 2
    assert not (
        tmp_path
        / "output/video_profile/fixed_15s"
        / f"{CONTENT_ID}_profile.json"
    ).exists()
