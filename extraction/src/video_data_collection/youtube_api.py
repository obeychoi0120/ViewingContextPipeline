from __future__ import annotations

import os
from collections import OrderedDict
import time
from typing import Any
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from googleapiclient.discovery import build

from .config import PipelineConfig


def build_youtube_client(
    api_key: str | None = None,
    config_path: str = "config/video_data_collection.json",
) -> Any:
    load_dotenv("config/.env")
    resolved_api_key = (
        (api_key or "").strip()
        or os.getenv("YOUTUBE_API_KEY", "").strip()
    )
    if not resolved_api_key:
        raise RuntimeError(
            "YouTube API key is missing. Set YOUTUBE_API_KEY in .env or pass --youtube-api-key."
        )
    http = build_proxy_http_from_env()
    if http is not None:
        return build("youtube", "v3", developerKey=resolved_api_key, http=http)
    return build("youtube", "v3", developerKey=resolved_api_key)


def build_proxy_http_from_env() -> Any | None:
    proxy_url = first_proxy_url_from_env()
    if not proxy_url:
        return None

    try:
        import socks
        import httplib2
    except ImportError as exc:
        raise RuntimeError(
            "HTTP(S)_PROXY is configured, but PySocks is not installed. "
            "Run `pip install PySocks` or `pip install -r requirements.txt`."
        ) from exc

    parsed = urlparse(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    if not parsed.hostname:
        return None

    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80

    proxy_info = httplib2.ProxyInfo(
        proxy_type=socks.PROXY_TYPE_HTTP,
        proxy_host=parsed.hostname,
        proxy_port=port,
        proxy_user=unquote(parsed.username) if parsed.username else None,
        proxy_pass=unquote(parsed.password) if parsed.password else None,
    )
    return httplib2.Http(proxy_info=proxy_info, disable_ssl_certificate_validation=True)


def first_proxy_url_from_env() -> str:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def collect_video_candidates(youtube: Any, config: PipelineConfig, sleep_sec: float = 0.2) -> list[dict[str, str]]:
    candidates = _search_candidates(youtube, config, sleep_sec=sleep_sec)
    return [
        {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            **candidate,
        }
        for video_id, candidate in candidates.items()
    ]


def _search_candidates(youtube: Any, config: PipelineConfig, sleep_sec: float) -> OrderedDict[str, dict[str, str]]:
    candidates: OrderedDict[str, dict[str, str]] = OrderedDict()
    group_counts: dict[str, int] = {}

    for seed in config.seed_queries:
        remaining = config.max_results_per_query
        page_token = None

        while remaining > 0:
            group_remaining = _group_remaining(seed.group, config.max_results_per_group, group_counts)
            if group_remaining <= 0:
                break

            batch_size = min(50, remaining, group_remaining)
            response = (
                youtube.search()
                .list(
                    part="snippet",
                    q=seed.query,
                    type="video",
                    maxResults=batch_size,
                    pageToken=page_token,
                    regionCode=seed.region_code or config.region_code,
                    relevanceLanguage=seed.relevance_language or config.relevance_language,
                    videoCaption=config.video_caption,
                    safeSearch="none",
                )
                .execute()
            )

            for item in response.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if not video_id:
                    continue

                if video_id in candidates:
                    candidates[video_id]["seed_query"] = _append_unique(
                        candidates[video_id]["seed_query"], seed.query
                    )
                    candidates[video_id]["seed_group"] = _append_unique(
                        candidates[video_id]["seed_group"], seed.group
                    )
                    candidates[video_id]["seed_category"] = _append_unique(
                        candidates[video_id].get("seed_category", ""), seed.category
                    )
                    candidates[video_id]["search_language"] = _append_unique(
                        candidates[video_id].get("search_language", ""),
                        seed.relevance_language or config.relevance_language,
                    )
                else:
                    if _group_remaining(seed.group, config.max_results_per_group, group_counts) <= 0:
                        continue
                    candidates[video_id] = {
                        "seed_query": seed.query,
                        "seed_group": seed.group,
                        "seed_category": seed.category,
                        "search_language": seed.relevance_language or config.relevance_language,
                    }
                    if seed.group:
                        group_counts[seed.group] = group_counts.get(seed.group, 0) + 1

            remaining -= batch_size
            page_token = response.get("nextPageToken")
            if not page_token:
                break
            time.sleep(sleep_sec)

    return candidates


def _group_remaining(group: str, max_results_per_group: int, group_counts: dict[str, int]) -> int:
    if not group:
        return max_results_per_group
    return max_results_per_group - group_counts.get(group, 0)


def _append_unique(existing: str, new_value: str) -> str:
    if not new_value:
        return existing
    values = [part for part in existing.split("|") if part]
    if new_value not in values:
        values.append(new_value)
    return "|".join(values)
