from __future__ import annotations

from google.genai import types
from pydantic import BaseModel

from viewing_context_pipeline.extraction.common.gemini import RETRYABLE_HTTP_STATUS_CODES
from viewing_context_pipeline.extraction.common.gemini import create_client
from viewing_context_pipeline.extraction.common.gemini import make_extraction_config


class ResponseSchema(BaseModel):
    value: str


def test_generate_config_uses_schema_and_thinking_without_sampling_parameters() -> None:
    config = make_extraction_config(
        system_instruction="Analyze the input.",
        response_schema=ResponseSchema,
        thinking_level="medium",
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )

    assert config.response_mime_type == "application/json"
    assert config.response_schema is ResponseSchema
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level.value == "MEDIUM"
    assert config.thinking_config.include_thoughts is None
    assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.seed is None
    assert config.candidate_count is None


def test_vertex_client_uses_bounded_retry_without_403(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_client(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("viewing_context_pipeline.extraction.common.gemini.genai.Client", fake_client)
    assert create_client("project-id", "global", bounded_retry=True) is sentinel
    retry = captured["http_options"].retry_options
    assert retry.attempts == 4
    assert retry.http_status_codes == RETRYABLE_HTTP_STATUS_CODES
    assert 403 not in retry.http_status_codes
