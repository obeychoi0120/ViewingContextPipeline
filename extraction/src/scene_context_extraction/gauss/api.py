from __future__ import annotations

import json
import re
import ssl
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from time import perf_counter, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
SUPPORTED_IMAGE_URL_FIELDS = frozenset({"mode", "content_id", "filename"})
REQUIRED_IMAGE_URL_FIELDS = frozenset({"content_id", "filename"})
MAX_IMAGES_PER_REQUEST = 4
MAX_RETRY_DELAY_SECONDS = 30.0


class GaussApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class GaussApiGeneration:
    text: str
    generated_tokens: int
    finish_reason: str
    generation_seconds: float


class GaussApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        image_url_template: str,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        ssl_cert_file: str | None = None,
    ) -> None:
        self.base_url = validate_http_url(
            base_url,
            field_name="API_BASE_URL",
        ).rstrip("/")
        self.model = require_nonempty_string(model, "API_MODEL")
        self.image_url_template = validate_image_url_template(image_url_template)
        self.timeout_seconds = require_positive_number(
            timeout_seconds,
            "API_TIMEOUT_SECONDS",
        )
        self.max_retries = require_nonnegative_int(
            max_retries,
            "API_MAX_RETRIES",
        )
        self.retry_base_seconds = require_positive_number(
            retry_base_seconds,
            "API_RETRY_BASE_SECONDS",
        )
        self.ssl_context = create_ssl_context(ssl_cert_file)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GaussApiClient":
        return cls(
            base_url=config.get("API_BASE_URL", ""),
            model=config.get("API_MODEL", ""),
            image_url_template=config.get("IMAGE_URL_TEMPLATE", ""),
            timeout_seconds=config.get("API_TIMEOUT_SECONDS", 180),
            max_retries=config.get("API_MAX_RETRIES", 3),
            retry_base_seconds=config.get("API_RETRY_BASE_SECONDS", 1.0),
            ssl_cert_file=config.get("API_SSL_CERT_FILE"),
        )

    def ensure_ready(self) -> None:
        response = self._request_json("GET", "models")
        models = response.get("data")
        if not isinstance(models, list):
            raise GaussApiError("Gauss models response must contain a data list")
        model_ids = {
            item.get("id")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if self.model not in model_ids:
            raise GaussApiError(
                f"API_MODEL {self.model!r} is not available from "
                f"{self.base_url}/models"
            )

    def image_urls(
        self,
        image_paths: list[str],
        *,
        mode: str,
        content_id: str,
    ) -> list[str]:
        if not image_paths:
            raise ValueError("at least one image is required for Gauss inference")
        if len(image_paths) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"Gauss inference accepts at most {MAX_IMAGES_PER_REQUEST} images, "
                f"got {len(image_paths)}"
            )

        template_values = {
            "mode": quote(require_nonempty_string(mode, "mode"), safe=""),
            "content_id": quote(
                require_nonempty_string(content_id, "content_id"),
                safe="",
            ),
        }
        urls: list[str] = []
        for image_path in image_paths:
            filename = Path(image_path).name
            if not filename:
                raise ValueError(f"image path has no filename: {image_path!r}")
            url = self.image_url_template.format(
                **template_values,
                filename=quote(filename, safe=""),
            )
            urls.append(validate_http_url(url, field_name="generated image URL"))
        return urls

    def generate(
        self,
        image_urls: list[str],
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        shot_reference_texts: list[str] | None = None,
    ) -> GaussApiGeneration:
        if not 1 <= len(image_urls) <= MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"Gauss inference requires 1-{MAX_IMAGES_PER_REQUEST} image URLs, "
                f"got {len(image_urls)}"
            )
        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": validate_http_url(url, field_name="image URL"),
                },
            }
            for url in image_urls
        ]
        reference_texts = shot_reference_texts or []
        if reference_texts and len(reference_texts) != len(image_content):
            raise ValueError(
                "Gauss multimodal input requires one shot_reference per image"
            )
        if reference_texts:
            user_content = []
            for image_part, reference_text in zip(image_content, reference_texts):
                user_content.append(image_part)
                user_content.append({"type": "text", "text": reference_text})
        else:
            user_content = image_content
        user_content.append({"type": "text", "text": user_message})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_tokens": require_positive_int(max_tokens, "max_tokens"),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": require_positive_int(top_k, "top_k"),
        }

        started_at = perf_counter()
        response = self._request_json("POST", "chat/completions", payload)
        generation_seconds = perf_counter() - started_at
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GaussApiError(
                "Gauss completion response must contain at least one choice"
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise GaussApiError("Gauss completion choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"),
            str,
        ):
            raise GaussApiError(
                "Gauss completion choice must contain string message.content"
            )

        usage = response.get("usage")
        completion_tokens = (
            usage.get("completion_tokens", 0)
            if isinstance(usage, dict)
            else 0
        )
        if type(completion_tokens) is not int or completion_tokens < 0:
            completion_tokens = 0
        finish_reason = choice.get("finish_reason")
        return GaussApiGeneration(
            text=message["content"],
            generated_tokens=completion_tokens,
            finish_reason=(
                finish_reason if isinstance(finish_reason, str) else ""
            ),
            generation_seconds=generation_seconds,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        for retry_index in range(self.max_retries + 1):
            request = Request(
                url=url,
                data=body,
                headers=headers,
                method=method,
            )
            try:
                with urlopen(
                    request,
                    timeout=self.timeout_seconds,
                    context=self.ssl_context,
                ) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise GaussApiError(
                        "Gauss API response must be a JSON object"
                    )
                return parsed
            except HTTPError as exc:
                if (
                    exc.code not in RETRYABLE_HTTP_STATUS
                    or retry_index >= self.max_retries
                ):
                    content_type = (
                        exc.headers.get("Content-Type") if exc.headers else None
                    )
                    response_detail = f" {exc.reason}" if exc.reason else ""
                    if content_type:
                        response_detail += f" (Content-Type: {content_type})"
                    api_message = safe_http_error_message(exc, content_type)
                    if api_message:
                        response_detail += f": {api_message}"
                    raise GaussApiError(
                        f"Gauss API {method} {url} failed with HTTP {exc.code}"
                        f"{response_detail}"
                    ) from exc
                delay = retry_delay_seconds(
                    retry_index,
                    self.retry_base_seconds,
                    exc.headers.get("Retry-After") if exc.headers else None,
                )
            except (URLError, TimeoutError) as exc:
                if retry_index >= self.max_retries:
                    reason = exc.reason if isinstance(exc, URLError) else exc
                    raise GaussApiError(
                        f"Gauss API {method} {url} failed after "
                        f"{self.max_retries + 1} attempts: {reason}"
                    ) from exc
                delay = retry_delay_seconds(
                    retry_index,
                    self.retry_base_seconds,
                )
            sleep(delay)

        raise AssertionError("unreachable Gauss API retry state")


def create_ssl_context(cert_file: Any) -> ssl.SSLContext | None:
    if cert_file is None:
        return None
    cert_path = Path(require_nonempty_string(cert_file, "API_SSL_CERT_FILE"))
    if not cert_path.is_file():
        raise ValueError(
            f"API_SSL_CERT_FILE does not exist or is not a file: {cert_path}"
        )
    try:
        return ssl.create_default_context(cafile=str(cert_path))
    except OSError as exc:
        raise ValueError(
            f"API_SSL_CERT_FILE could not be loaded: {cert_path}"
        ) from exc


def safe_http_error_message(
    error: HTTPError,
    content_type: str | None,
) -> str:
    if not content_type or "json" not in content_type.lower():
        return ""
    try:
        raw = error.read(8192).decode("utf-8", errors="replace")
        parsed = json.loads(raw)
    except (AttributeError, OSError, TypeError, UnicodeError, json.JSONDecodeError):
        return ""
    if not isinstance(parsed, dict):
        return ""

    detail: Any = parsed.get("error")
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("detail")
    if not isinstance(detail, str):
        detail = parsed.get("message") or parsed.get("detail")
    if not isinstance(detail, str):
        return ""

    normalized = " ".join(detail.split())
    redacted = re.sub(
        r"https?://[^\s\"']+",
        "<redacted-url>",
        normalized,
        flags=re.IGNORECASE,
    )
    if len(redacted) > 500:
        return f"{redacted[:497]}..."
    return redacted


def validate_image_url_template(template: Any) -> str:
    value = require_nonempty_string(template, "IMAGE_URL_TEMPLATE")
    try:
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(value)
            if field_name is not None
        }
    except ValueError as exc:
        raise ValueError("IMAGE_URL_TEMPLATE has invalid braces") from exc
    unknown = fields - SUPPORTED_IMAGE_URL_FIELDS
    if unknown:
        raise ValueError(
            f"IMAGE_URL_TEMPLATE has unsupported placeholders: {sorted(unknown)}"
        )
    missing = REQUIRED_IMAGE_URL_FIELDS - fields
    if missing:
        raise ValueError(
            f"IMAGE_URL_TEMPLATE is missing required placeholders: {sorted(missing)}"
        )
    sample = value.format(
        mode="fixed_15s",
        content_id="content",
        filename="0000.png",
    )
    validate_http_url(sample, field_name="IMAGE_URL_TEMPLATE")
    return value


def validate_http_url(value: Any, *, field_name: str) -> str:
    text = require_nonempty_string(value, field_name)
    parsed = urlparse(text)
    has_invalid_character = any(
        character.isspace() or character in '<>"{}|\\^`'
        for character in text
    )
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or has_invalid_character
    ):
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    return text


def require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def retry_delay_seconds(
    retry_index: int,
    retry_base_seconds: float,
    retry_after: str | None = None,
) -> float:
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = -1.0
        if delay >= 0:
            return min(delay, MAX_RETRY_DELAY_SECONDS)
    return min(
        retry_base_seconds * (2**retry_index),
        MAX_RETRY_DELAY_SECONDS,
    )


def api_workers_from_config(config: dict[str, Any]) -> int:
    return require_positive_int(config.get("API_WORKERS", 1), "API_WORKERS")
