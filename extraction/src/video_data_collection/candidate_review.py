from __future__ import annotations

import csv
import json
import re
import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .raw_pipeline import video_id_from_url
from .ytdlp_utils import ytdlp_base_opts


DEFAULT_ALLOWED_LANGUAGES = ("ko", "en")
DEFAULT_MIN_DURATION_SEC = 120
DEFAULT_MAX_DURATION_SEC = 7200
DEFAULT_MIN_SCRIPT_CHARS = 400
DEFAULT_REVIEW_SLEEP_SEC = 2.0
REVIEW_CSV_FIELDNAMES = [
    "video_id",
    "url",
    "seed_query",
    "seed_group",
    "seed_category",
    "search_language",
    "title",
    "channel",
    "channel_id",
    "upload_date",
    "duration_sec",
    "view_count",
    "language",
    "availability",
    "live_status",
    "webpage_url",
    "caption_languages",
    "script_language",
    "script_path",
    "script_chars",
    "script_excerpt",
    "decision",
    "needs_visual_review",
    "reasons",
    "error",
]

CATEGORY_BY_GROUP = {
    "animation": "Animation",
    "business_society": "News",
    "documentary": "Documentary",
    "entertainment_variety": "Variety",
    "game": "Game",
    "gaming": "Game",
    "movie_drama": "Movie",
    "music": "Music",
    "music_culture": "Music",
    "news": "News",
    "review": "Shopping",
    "science_tech": "Tech",
    "shopping": "Shopping",
    "sports": "Sports",
    "tech": "Tech",
    "travel": "Travel",
    "vlog_lifestyle": "Travel",
}


class YouTubeRateLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoListItem:
    category: str
    source: str
    number: int
    list_id: str
    url: str


def write_dict_csv(rows: list[dict[str, object]], output_path: str | Path, fieldnames: list[str] | None = None) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = stable_fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_dict_csv(row: dict[str, object], output_path: str | Path, fieldnames: list[str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def ensure_review_csv_schema(output_path: str | Path) -> None:
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header == REVIEW_CSV_FIELDNAMES:
        return
    rows = read_dict_csv(path)
    write_dict_csv(rows, path, fieldnames=REVIEW_CSV_FIELDNAMES)


def read_dict_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stable_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(key)
    return names


def enrich_candidates(
    candidates: Iterable[dict[str, str]],
    script_dir: str | Path,
    allowed_languages: tuple[str, ...] = DEFAULT_ALLOWED_LANGUAGES,
    min_duration_sec: int = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: int = DEFAULT_MAX_DURATION_SEC,
    min_script_chars: int = DEFAULT_MIN_SCRIPT_CHARS,
    show_progress: bool = False,
    sleep_sec: float = 0.0,
    existing_rows: Iterable[dict[str, object]] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, object]]:
    candidate_list = list(candidates)
    rows: list[dict[str, object]] = list(existing_rows or [])
    processed_keys = {
        key for key in (candidate_key(row) for row in rows if review_row_is_complete(row))
        if key
    }
    pending_candidates = [
        candidate for candidate in candidate_list
        if candidate_key(candidate) not in processed_keys
    ]
    if output_path is not None and rows:
        path = Path(output_path)
        if not path.exists() or path.stat().st_size == 0:
            write_dict_csv(rows, path, fieldnames=REVIEW_CSV_FIELDNAMES)

    iterable: Iterable[dict[str, str]] = pending_candidates
    if show_progress:
        from tqdm import tqdm

        iterable = tqdm(pending_candidates, desc="Reviewing candidates", unit="video")

    for index, candidate in enumerate(iterable, start=1):
        row = enrich_candidate(
            candidate,
            script_dir=script_dir,
            allowed_languages=allowed_languages,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            min_script_chars=min_script_chars,
        )
        rows.append(row)
        key = candidate_key(row)
        if key:
            processed_keys.add(key)
        if output_path is not None:
            append_dict_csv(row, output_path, REVIEW_CSV_FIELDNAMES)
        if sleep_sec > 0 and index < len(pending_candidates):
            time.sleep(sleep_sec)
    return rows


def candidate_key(row: dict[str, object]) -> str:
    video_id = str(row.get("video_id") or "").strip()
    if video_id:
        return f"id:{video_id}"
    url = str(row.get("url") or row.get("webpage_url") or "").strip()
    parsed_video_id = video_id_from_url(url)
    if parsed_video_id:
        return f"id:{parsed_video_id}"
    return f"url:{url}" if url else ""


def review_row_is_complete(row: dict[str, object]) -> bool:
    decision = str(row.get("decision") or "").strip()
    reasons = str(row.get("reasons") or "").strip()
    if not decision:
        return False
    return not reasons.startswith("metadata_or_script_error:")


def enrich_candidate(
    candidate: dict[str, str],
    script_dir: str | Path,
    allowed_languages: tuple[str, ...] = DEFAULT_ALLOWED_LANGUAGES,
    min_duration_sec: int = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: int = DEFAULT_MAX_DURATION_SEC,
    min_script_chars: int = DEFAULT_MIN_SCRIPT_CHARS,
) -> dict[str, object]:
    url = candidate.get("url", "")
    video_id = candidate.get("video_id") or video_id_from_url(url)
    result: dict[str, object] = {**candidate, "video_id": video_id, "url": url}
    try:
        info = extract_video_info(url)
        metadata = metadata_from_info(info)
        script = collect_script(info, video_id=video_id, script_dir=script_dir, allowed_languages=allowed_languages)
        result.update(metadata)
        result.update(script)
        result.update(
            judge_candidate(
                result,
                allowed_languages=allowed_languages,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                min_script_chars=min_script_chars,
            )
        )
    except Exception as exc:
        if is_youtube_rate_limit_error(exc):
            raise YouTubeRateLimitError(
                "YouTube rate limit detected. Stop now and resume after the cooldown; "
                "the current candidate was not written to the review CSV."
            ) from exc
        result.update(
            {
                "decision": "needs_visual_review",
                "needs_visual_review": "true",
                "reasons": f"metadata_or_script_error:{type(exc).__name__}",
                "error": str(exc),
            }
        )
    return result


def is_youtube_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "rate-limited by youtube" in text
        or "rate limited by youtube" in text
        or "current session has been rate-limited" in text
        or "this content isn't available, try again later" in text
    )


def extract_video_info(url: str) -> dict[str, Any]:
    import yt_dlp

    opts = {**ytdlp_base_opts(), "quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def metadata_from_info(info: dict[str, Any]) -> dict[str, object]:
    return {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_id": info.get("channel_id") or "",
        "upload_date": info.get("upload_date") or "",
        "duration_sec": info.get("duration") or "",
        "view_count": info.get("view_count") or "",
        "language": normalize_language(info.get("language") or ""),
        "availability": info.get("availability") or "",
        "live_status": info.get("live_status") or "",
        "webpage_url": info.get("webpage_url") or "",
        "caption_languages": "|".join(caption_languages(info)),
    }


def collect_script(
    info: dict[str, Any],
    video_id: str,
    script_dir: str | Path,
    allowed_languages: tuple[str, ...] = DEFAULT_ALLOWED_LANGUAGES,
) -> dict[str, object]:
    track = select_caption_track(info, allowed_languages)
    if not track:
        return {"script_language": "", "script_path": "", "script_chars": 0, "script_excerpt": ""}

    text = caption_text_from_url(str(track["url"]))
    path = Path(script_dir) / f"{safe_filename(video_id or 'unknown')}.{track['language']}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "script_language": track["language"],
        "script_path": str(path),
        "script_chars": len(text),
        "script_excerpt": text[:500].replace("\r", " ").replace("\n", " "),
    }


def select_caption_track(info: dict[str, Any], allowed_languages: tuple[str, ...]) -> dict[str, str] | None:
    for source_key in ("subtitles", "automatic_captions"):
        captions = info.get(source_key) or {}
        for language in allowed_languages:
            key = matching_caption_language(captions.keys(), language)
            if not key:
                continue
            entries = captions.get(key) or []
            entry = preferred_caption_entry(entries)
            if entry and entry.get("url"):
                return {"language": language, "url": str(entry["url"])}
    return None


def matching_caption_language(keys: Iterable[str], language: str) -> str:
    for key in keys:
        normalized = normalize_language(key)
        if normalized == language:
            return key
    for key in keys:
        normalized = normalize_language(key)
        if normalized.startswith(f"{language}-"):
            return key
    return ""


def preferred_caption_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for ext in ("vtt", "srv3", "ttml"):
        for entry in entries:
            if str(entry.get("ext") or "").lower() == ext:
                return entry
    return entries[0] if entries else None


def caption_text_from_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, timeout=30, context=context) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    return clean_caption_text(raw)


def clean_caption_text(raw: str) -> str:
    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "WEBVTT" or stripped.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if "-->" in stripped or re.fullmatch(r"\d+", stripped):
            continue
        stripped = re.sub(r"<[^>]+>", "", stripped)
        stripped = re.sub(r"\{[^}]+\}", "", stripped)
        stripped = stripped.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        if stripped and (not lines or lines[-1] != stripped):
            lines.append(stripped)
    return " ".join(lines)


def caption_languages(info: dict[str, Any]) -> list[str]:
    languages: list[str] = []
    for source_key in ("subtitles", "automatic_captions"):
        for key in (info.get(source_key) or {}).keys():
            language = normalize_language(key)
            if language and language not in languages:
                languages.append(language)
    return languages


def judge_candidate(
    row: dict[str, object],
    allowed_languages: tuple[str, ...] = DEFAULT_ALLOWED_LANGUAGES,
    min_duration_sec: int = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: int = DEFAULT_MAX_DURATION_SEC,
    min_script_chars: int = DEFAULT_MIN_SCRIPT_CHARS,
) -> dict[str, str]:
    reasons: list[str] = []
    duration = int_or_zero(row.get("duration_sec"))
    script_chars = int_or_zero(row.get("script_chars"))
    language_value = str(row.get("script_language") or row.get("language") or row.get("search_language") or "")
    language = normalize_language(first_pipe_value(language_value) or language_value)
    title = str(row.get("title") or "").lower()
    availability = str(row.get("availability") or "").lower()
    live_status = str(row.get("live_status") or "").lower()

    if availability in {"private", "premium_only", "subscriber_only"}:
        reasons.append(f"availability:{availability}")
    if live_status and live_status not in {"not_live", "was_live"}:
        reasons.append(f"live_status:{live_status}")
    if "#shorts" in title or "쇼츠" in title:
        reasons.append("shorts_title")
    if duration and duration < min_duration_sec:
        reasons.append(f"too_short:{duration}")
    if duration and duration > max_duration_sec:
        reasons.append(f"too_long:{duration}")
    if language and language not in allowed_languages:
        reasons.append(f"language:{language}")

    hard_reasons = [
        reason for reason in reasons
        if reason.startswith(("availability:", "live_status:", "shorts_title", "too_short:", "too_long:", "language:"))
    ]
    if hard_reasons:
        return {"decision": "reject", "needs_visual_review": "false", "reasons": "|".join(reasons)}

    if script_chars >= min_script_chars and language in allowed_languages:
        return {"decision": "accept", "needs_visual_review": "false", "reasons": "metadata_script_ok"}

    review_reasons = reasons[:]
    if not language:
        review_reasons.append("unknown_language")
    if script_chars < min_script_chars:
        review_reasons.append(f"short_or_missing_script:{script_chars}")
    return {"decision": "needs_visual_review", "needs_visual_review": "true", "reasons": "|".join(review_reasons)}


def read_video_list(path: str | Path) -> list[VideoListItem]:
    current_category = ""
    items: list[VideoListItem] = []
    seen_list_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            category_match = re.fullmatch(r"\[([A-Za-z]+)\]", line)
            if category_match:
                current_category = category_match.group(1)
                continue
            item_match = re.fullmatch(r"([A-Za-z]+)_(Manual|Auto)_(\d{3})\s+(https?://\S+)", line)
            if not item_match:
                raise ValueError(f"Invalid TXT video list line {line_number}: {line}")
            item_category, source, item_number, url = item_match.groups()
            list_id = f"{item_category}_{source}_{item_number}"
            if current_category and item_category != current_category:
                raise ValueError(
                    f"Line {line_number} list id does not match [{current_category}]: {list_id}"
                )
            if list_id in seen_list_ids:
                raise ValueError(f"Duplicate list id on line {line_number}: {list_id}")
            seen_list_ids.add(list_id)
            items.append(
                VideoListItem(
                    category=item_category,
                    source=source,
                    number=int(item_number),
                    list_id=list_id,
                    url=url,
                )
            )
    return items


def merge_video_lists(
    manual_list_path: str | Path,
    searched_list_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    manual_items = read_video_list(manual_list_path)
    searched_items = read_video_list(searched_list_path)
    merged_items = merge_video_list_items(manual_items, searched_items)
    write_video_list(merged_items, output_path)
    return {
        "manual_count": len(manual_items),
        "searched_count": len(searched_items),
        "searched_added": len(merged_items) - len(manual_items),
        "merged_count": len(merged_items),
        "output_path": str(output_path),
    }


def accepted_candidate_rows(
    candidate_rows: list[dict[str, str]],
    excluded_video_ids: set[str],
    excluded_urls: set[str],
) -> list[dict[str, str]]:
    accepted: list[dict[str, str]] = []
    seen_video_ids = set(excluded_video_ids)
    seen_urls = set(excluded_urls)
    for row in candidate_rows:
        if str(row.get("decision") or "").strip().lower() != "accept":
            continue
        url = str(row.get("url") or row.get("webpage_url") or "").strip()
        video_id = str(row.get("video_id") or video_id_from_url(url) or "").strip()
        if not url or not video_id or video_id in seen_video_ids or url in seen_urls:
            continue
        category = category_for_candidate(row)
        if not category:
            continue
        accepted.append({**row, "url": url, "video_id": video_id, "category": category})
        seen_video_ids.add(video_id)
        seen_urls.add(url)
    return accepted


def write_searched_video_list(
    candidate_rows: list[dict[str, str]],
    output_path: str | Path,
) -> dict[str, object]:
    accepted = accepted_candidate_rows(candidate_rows, excluded_video_ids=set(), excluded_urls=set())
    counts: dict[str, int] = {}
    items: list[VideoListItem] = []
    for row in accepted:
        category = str(row["category"])
        number = counts.get(category, 0) + 1
        counts[category] = number
        list_id = f"{category}_Auto_{number:03d}"
        items.append(
            VideoListItem(
                category=category,
                source="Auto",
                number=number,
                list_id=list_id,
                url=str(row["url"]),
            )
        )
    write_video_list(items, output_path)
    return {"searched_count": len(items), "output_path": str(output_path)}


def merge_video_list_items(
    manual_items: list[VideoListItem],
    searched_items: list[VideoListItem],
) -> list[VideoListItem]:
    merged: list[VideoListItem] = []
    seen_video_ids: set[str] = set()
    seen_urls: set[str] = set()
    category_order: list[str] = []
    items_by_category: dict[str, list[VideoListItem]] = {}

    for item in [*manual_items, *searched_items]:
        if item.category not in category_order:
            category_order.append(item.category)
        items_by_category.setdefault(item.category, []).append(item)

    for category in category_order:
        category_items = items_by_category.get(category, [])
        for source in ("Manual", "Auto"):
            for item in category_items:
                if item.source != source:
                    continue
                video_id = video_id_from_url(item.url)
                if video_id and video_id in seen_video_ids:
                    continue
                if item.url in seen_urls:
                    continue
                merged.append(item)
                if video_id:
                    seen_video_ids.add(video_id)
                seen_urls.add(item.url)
    return merged


def write_video_list(items: list[VideoListItem], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    category_order: list[str] = []
    for item in items:
        if item.category not in category_order:
            category_order.append(item.category)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for category_index, category in enumerate(category_order):
            if category_index:
                f.write("\n")
            f.write(f"[{category}]\n")
            for item in items:
                if item.category == category:
                    f.write(f"{item.list_id} {item.url}\n")


def category_for_candidate(row: dict[str, str]) -> str:
    for key in ("category", "seed_category"):
        value = first_pipe_value(str(row.get(key) or ""))
        if value:
            return normalize_category(value)
    group = first_pipe_value(str(row.get("seed_group") or ""))
    return CATEGORY_BY_GROUP.get(group, normalize_category(group))


def first_pipe_value(value: str) -> str:
    for part in value.split("|"):
        stripped = part.strip()
        if stripped:
            return stripped
    return ""


def normalize_category(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", value)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def int_or_zero(value: object) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return 0


def normalize_language(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return ""
    return text.split("-")[0]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"

