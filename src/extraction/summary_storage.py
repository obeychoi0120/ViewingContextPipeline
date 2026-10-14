"""Require nonempty summary text; formatting and word counts are prompt guidance."""

from __future__ import annotations

import re
from artifact_io import read_json, read_jsonl
from extraction.errors import ExtractionStepError, UnsupportedArtifactError
from extraction.token_usage import validate_tokens

SUMMARY_SCHEMA_VERSION = "video-summary/v5"


class SummaryContractError(ValueError):
    pass


def inspect_summary(text: str) -> tuple[str, list[str]]:
    if not isinstance(text, str) or not text.strip():
        return "", ["empty"]
    return text.strip(), []


def validate_summary(text: str) -> str:
    normalized, violations = inspect_summary(text)
    if violations:
        raise SummaryContractError(", ".join(violations))
    return normalized


def render_summary_prompt(template, scenes, *, english_title=None):
    title = (english_title or "").strip() or "(unavailable)"
    values = {"scenes": scenes, "english_title": title}
    rendered = re.sub(r"\{(scenes|english_title)\}", lambda match: values[match.group(1)], template)
    return rendered


def validate_summary_template(template, uses_title):
    if "{scenes}" not in template:
        raise ValueError("Summary schema requires {scenes}")
    if ("{english_title}" in template) != uses_title:
        raise ValueError("Summary schema {english_title} must match the Arm title policy")


def reuse_summary_document(output_path, *, content_id, arm):
    try:
        doc = read_json(output_path)
        if (
            doc.get("schema_version") != SUMMARY_SCHEMA_VERSION
            or doc.get("status") not in {"complete", "failed"}
        ):
            raise UnsupportedArtifactError(
                f"unsupported Summary schema/status: {output_path}; "
                f"expected {SUMMARY_SCHEMA_VERSION} complete/failed"
            )
        summary_model_from_document(doc, path=output_path)
        validate_tokens(doc.get("tokens"))
        if (
            doc.get("content_id") != content_id
            or doc.get("arm") != arm
            or (type(doc.get("scene_count")) is not int)
            or (doc["scene_count"] < 0)
            or (not isinstance(doc.get("provenance"), dict))
        ):
            raise ValueError("summary identity, provenance, status, or scene count mismatch")
        if doc["status"] == "failed":
            if doc.get("text") != "" or doc.get("word_count") != 0 or (not doc.get("violations")):
                raise ValueError("failed summary requires empty text and a failure reason")
            return doc
        if doc["scene_count"] == 0:
            raise ValueError("successful summary requires scenes")
        text, violations = inspect_summary(doc.get("text"))
        if not text or doc.get("word_count") != len(text.split()):
            raise ValueError("empty summary or incorrect word count")
        if doc["status"] == "complete" and (violations or doc.get("violations") != []):
            raise ValueError("normal summary violates the summary contract")
        return doc
    except (ValueError, KeyError, TypeError) as exc:
        raise ExtractionStepError(f"invalid summary {output_path}: {exc}") from exc


def summary_model_from_document(doc, *, path):
    """Read the explicit model field; never infer missing generation provenance."""
    provenance = doc.get("provenance")
    model = provenance.get("summary_model") if isinstance(provenance, dict) else None
    if model not in {"qwen3vl", "qwen3.5", "gemini"}:
        raise UnsupportedArtifactError(f"missing/unsupported Summary summary_model: {path}")
    return model


def summary_failure_rows(directory):
    """Read the summary failure log without modifying it."""
    paths = [directory / "failures.jsonl"]
    rows = {}
    for path in paths:
        if path.is_file():
            for row in read_jsonl(path):
                rows[str(row["content_id"])] = row
    return rows


def check_summary_model(directory, model):
    for path in directory.glob("*.json"):
        try:
            doc = read_json(path)
        except ValueError:
            continue
        if (
            doc.get("schema_version") != SUMMARY_SCHEMA_VERSION
            or doc.get("status") not in {"complete", "failed"}
        ):
            raise UnsupportedArtifactError(
                f"unsupported Summary schema/status: {path}; "
                f"expected {SUMMARY_SCHEMA_VERSION} complete/failed"
            )
        if summary_model_from_document(doc, path=path) != model:
            raise ExtractionStepError(f"summary model mismatch in {path}; use --force or a new run")
    for row in summary_failure_rows(directory).values():
        if row.get("summary_model") not in {"qwen3vl", "qwen3.5", "gemini"}:
            raise UnsupportedArtifactError(f"missing/unsupported summary_model in {directory / 'failures.jsonl'}")
        if row["summary_model"] != model:
            raise ExtractionStepError(
                f"summary failure model mismatch in {directory}; use --force or a new run"
            )
