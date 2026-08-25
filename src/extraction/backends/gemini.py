from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Sequence

from extraction.multimodal import (
    shot_reference_text,
    validate_image_reference_alignment,
)


RETRYABLE_HTTP_STATUS_CODES = [408, 429, 500, 502, 503, 504]


@dataclass
class GeminiBackend:
    client: Any
    model_id: str
    thinking_level: str | None = None

    @classmethod
    def vertex(
        cls,
        *,
        project_id: str,
        model_id: str,
        location: str = "global",
        thinking_level: str | None = None,
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
        return cls(client=client, model_id=model_id, thinking_level=thinking_level)

    def generate(
        self,
        images: Sequence[Any],
        prompt: str,
        max_new_tokens: int,
        references: Sequence[dict[str, Any]] = (),
    ) -> str:
        _, types = _google_genai()
        if references:
            validate_image_reference_alignment(len(images), list(references))
        contents: list[Any] = []
        for index, image in enumerate(images):
            contents.append(_image_part(image, types))
            if references:
                contents.append(
                    types.Part.from_text(text=shot_reference_text(references[index]))
                )
        contents.append(types.Part.from_text(text=prompt))
        config_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": max_new_tokens,
        }
        if self.thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")
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
