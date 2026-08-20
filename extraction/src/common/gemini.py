from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types


RETRYABLE_HTTP_STATUS_CODES = [408, 429, 500, 502, 503, 504]


def create_client(
    project_id: str,
    location: str,
    *,
    bounded_retry: bool = False,
) -> genai.Client:
    kwargs: dict[str, Any] = {
        "vertexai": True,
        "project": project_id,
        "location": location,
    }
    if bounded_retry:
        retry_options = types.HttpRetryOptions(
            attempts=4,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=RETRYABLE_HTTP_STATUS_CODES,
        )
        kwargs["http_options"] = types.HttpOptions(
            timeout=300_000,
            retry_options=retry_options,
        )
    return genai.Client(**kwargs)


def make_extraction_config(
    system_instruction: str | None = None,
    thinking_level: str | None = None,
    media_resolution: types.MediaResolution | None = None,
    response_schema: type[Any] | None = None,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {}
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    if media_resolution is not None:
        kwargs["media_resolution"] = media_resolution
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


def parse_json_response(text: str) -> Any:
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return json.loads(clean_text.strip())
