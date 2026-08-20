from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SeedQuery:
    query: str
    group: str = ""
    category: str = ""
    relevance_language: str = ""
    region_code: str = ""


@dataclass(frozen=True)
class PipelineConfig:
    seed_queries: list[SeedQuery]
    max_results_per_query: int = 8
    max_results_per_group: int = 16
    region_code: str = "KR"
    relevance_language: str = "ko"
    video_caption: str = "closedCaption"


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    seed_queries = _parse_seed_queries(raw.get("seed_queries", []))
    if not seed_queries:
        raise ValueError("config must contain at least one seed query")

    max_results = int(raw.get("max_results_per_query", 8))
    if max_results < 1:
        raise ValueError("max_results_per_query must be >= 1")

    max_results_per_group = int(raw.get("max_results_per_group", 16))
    if max_results_per_group < 1:
        raise ValueError("max_results_per_group must be >= 1")

    video_caption = str(raw.get("video_caption", "closedCaption"))
    if video_caption not in {"any", "closedCaption", "none"}:
        raise ValueError("video_caption must be one of: any, closedCaption, none")

    return PipelineConfig(
        seed_queries=seed_queries,
        max_results_per_query=max_results,
        max_results_per_group=max_results_per_group,
        region_code=str(raw.get("region_code", "KR")),
        relevance_language=str(raw.get("relevance_language", "ko")),
        video_caption=video_caption,
    )


def _parse_seed_queries(items: Any) -> list[SeedQuery]:
    queries: list[SeedQuery] = []
    for item in items:
        if isinstance(item, str):
            query = item.strip()
            group = ""
        elif isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            group = str(item.get("group", "")).strip()
            category = str(item.get("category", "")).strip()
            relevance_language = str(item.get("relevance_language", "")).strip()
            region_code = str(item.get("region_code", "")).strip()
        else:
            continue

        if query:
            queries.append(
                SeedQuery(
                    query=query,
                    group=group,
                    category=category,
                    relevance_language=relevance_language,
                    region_code=region_code,
                )
            )

    return queries
