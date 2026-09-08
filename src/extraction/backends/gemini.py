from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Sequence

RETRYABLE_HTTP_STATUS_CODES = [408, 429, 500, 502, 503, 504]


class GeminiEmptyResponseError(RuntimeError):
    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "Gemini returned an empty response; "
            + json.dumps(diagnostics, ensure_ascii=False)
        )


@dataclass
class GeminiBackend:
    client: Any
    model_id: str
    temperature: float = 0.0
    thinking_level: str | None = None
    media_resolution: str | None = None

    @classmethod
    def vertex(
        cls,
        *,
        project_id: str,
        model_id: str,
        location: str = "global",
        temperature: float = 0.0,
        thinking_level: str | None = None,
        media_resolution: str | None = None,
    ) -> GeminiBackend:
        genai, types = _google_genai()
        retry = types.HttpRetryOptions(
            attempts=4,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=RETRYABLE_HTTP_STATUS_CODES,
        )
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=types.HttpOptions(timeout=300_000, retry_options=retry),
        )
        return cls(
            client=client,
            model_id=model_id,
            temperature=temperature,
            thinking_level=thinking_level,
            media_resolution=media_resolution,
        )

    def generate(
        self,
        images: Sequence[Any],
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        _, types = _google_genai()
        contents: list[Any] = []
        for image in images:
            contents.append(_image_part(image, types))
        contents.append(types.Part.from_text(text=prompt))
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": max_new_tokens,
        }
        if self.thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        if self.media_resolution:
            config_kwargs["media_resolution"] = types.MediaResolution(
                self.media_resolution
            )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            candidates = getattr(response, "candidates", None) or []
            feedback = getattr(response, "prompt_feedback", None)
            usage = getattr(response, "usage_metadata", None)
            raise GeminiEmptyResponseError({
                "candidates": [
                    {
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "finish_message": getattr(candidate, "finish_message", None),
                    }
                    for candidate in candidates
                ],
                "prompt_feedback": (
                    feedback.model_dump(mode="json", exclude_none=True)
                    if feedback is not None else None
                ),
                "usage_metadata": (
                    usage.model_dump(mode="json", exclude_none=True)
                    if usage is not None else None
                ),
            })
        return text


def _image_part(image: Any, types: Any) -> Any:
    if not hasattr(image, "save"):
        raise TypeError("GeminiBackend images must be PIL-compatible objects")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def _google_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini extraction requires the 'gemini' optional dependencies"
        ) from exc
    return genai, types
