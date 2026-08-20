from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.common.manifest import CANONICAL_MANIFEST_PATH, read_manifest_rows
from src.common.output_paths import custom_output_root

from .bot_check import is_youtube_bot_check_error
from .candidate_review import (
    DEFAULT_ALLOWED_LANGUAGES,
    DEFAULT_MAX_DURATION_SEC,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MIN_SCRIPT_CHARS,
    DEFAULT_REVIEW_SLEEP_SEC,
    enrich_candidates,
    ensure_review_csv_schema,
    merge_video_lists,
    read_dict_csv,
    write_searched_video_list,
    write_dict_csv,
)
from .config import load_config
from .raw_pipeline import (
    build_content_paths,
    collect_video_metadata as collect_url_metadata,
    is_nonempty_file,
    load_processing_config,
    manifest_row,
    normalize_content_id,
    process_one,
    ref_jsonl_relative_path,
    remove_legacy_raw_artifacts,
    row_to_url,
    row_to_video_id,
    resized_keyframe_resolution,
    resized_keyframes_relative_path,
    resized_keyframes_are_complete,
    selected_filtered_timestamp_json,
    shot_interval_from_config,
    video_id_from_url,
    write_manifest,
    write_metadata_json,
)
from .youtube_api import build_youtube_client, collect_video_candidates


VIDEO_CONFIG_PATH = "config/video_data_collection.json"
SEARCH_CONFIG_PATH = "config/yt_video_search_queries.yaml"
DEFAULT_MANIFEST_PATH = CANONICAL_MANIFEST_PATH
BOT_CHECK_DEFERRED_COOLDOWN_SEC = 30 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect YouTube videos and extract reference multimodal data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("process-one", help="Download one YouTube video and extract reference JSONL data.")
    one.add_argument("--name", required=True)
    one.add_argument("--url", required=True)
    one.add_argument("--lang", choices=["ko", "en"], default=None)

    batch = subparsers.add_parser("process-batch", help="Process all videos from a CSV manifest.")
    batch.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Input/output manifest CSV. Defaults to contracts/manifest.csv.",
    )
    batch.add_argument("--lang", choices=["ko", "en"], default=None)
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--force", action="store_true", help="Reprocess rows even when outputs already exist.")
    batch.add_argument("--fail-fast", action="store_true", help="Stop the batch on the first per-video processing error.")

    create_manifest = subparsers.add_parser("create-manifest", help="Create a manifest from a TXT video list.")
    create_manifest.add_argument("--list", dest="list_file_path", required=True, help="Create a manifest from a manual TXT video list.")
    create_manifest.add_argument(
        "--output",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Output manifest path. Defaults to contracts/manifest.csv.",
    )

    metadata = subparsers.add_parser("get-metadata", help="Download metadata JSON files for rows in a manifest.")
    metadata.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Input manifest CSV. Defaults to contracts/manifest.csv.",
    )
    metadata.add_argument("--force", action="store_true", help="Overwrite existing metadata.json files.")

    subparsers.add_parser("inspect-output", help="Inspect metadata, Ref JSONL, and resized keyframe consistency.")

    search_candidates = subparsers.add_parser("search-candidates", help="Search YouTube and write raw candidate rows to CSV.")
    search_candidates.add_argument("--output", default="output/video_candidates.csv")

    review_candidates = subparsers.add_parser("review-candidates", help="Download candidate metadata/scripts and write review decisions.")
    review_candidates.add_argument("--candidates", default="output/video_candidates.csv")
    review_candidates.add_argument("--output", default="output/video_candidates_reviewed.csv")
    review_candidates.add_argument("--searched-list-output", default="video_list_searched.txt")
    review_candidates.add_argument("--script-dir", default="output/candidate_scripts")
    review_candidates.add_argument("--allowed-languages", default=",".join(DEFAULT_ALLOWED_LANGUAGES))
    review_candidates.add_argument("--min-duration-sec", type=int, default=DEFAULT_MIN_DURATION_SEC)
    review_candidates.add_argument("--max-duration-sec", type=int, default=DEFAULT_MAX_DURATION_SEC)
    review_candidates.add_argument("--min-script-chars", type=int, default=DEFAULT_MIN_SCRIPT_CHARS)
    review_candidates.add_argument("--sleep-sec", type=float, default=DEFAULT_REVIEW_SLEEP_SEC, help="Seconds to sleep between candidate metadata/script requests.")

    merge_list = subparsers.add_parser("merge-video-list", help="Merge manual and searched TXT video lists.")
    merge_list.add_argument("--manual", default="video_list_manual.txt")
    merge_list.add_argument("--searched", default="video_list_searched.txt")
    merge_list.add_argument("--output", default="video_list_merged.txt")

    microlens = subparsers.add_parser(
        "import-microlens",
        help="Inventory and process caller-owned MicroLens-100K MP4 files.",
    )
    microlens.add_argument(
        "--config",
        default="config/microlens_config.json",
        help="MicroLens importer config JSON.",
    )
    microlens.add_argument("--scope", choices=["smoke", "pilot"], required=True)
    microlens.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected local-video processing artifacts.",
    )
    microlens.add_argument(
        "--rebuild-selection",
        action="store_true",
        help="Replace the frozen selection after source/config changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "process-one":
        processing_config = load_processing_config(VIDEO_CONFIG_PATH)
        output_root = resolve_output_root()
        manifest_output = manifest_path()
        row = process_one(
            name=args.name,
            url=args.url,
            data_root=resolve_assets_root(),
            lang=args.lang,
            config_path=VIDEO_CONFIG_PATH,
            output_root=output_root,
        )
        rows = read_existing_manifest(manifest_output)
        seen_content_ids = {
            str(existing.get("content_id") or "")
            for existing in rows
        }
        append_manifest_row(row, manifest_output, rows, seen_content_ids)
        print(f"Wrote manifest to {manifest_output}")
    elif args.command == "process-batch":
        rows, manifest_output = run_process_batch(args)
        print(f"Wrote {len(rows)} manifest rows to {manifest_output}")
    elif args.command == "create-manifest":
        rows, manifest_output = run_create_manifest(args)
        print(f"Wrote {len(rows)} manifest rows to {manifest_output}")
    elif args.command == "get-metadata":
        result = run_get_metadata(args)
        print(
            f"Wrote {result['written']} metadata files; "
            f"skipped {result['skipped']} existing files."
        )
    elif args.command == "inspect-output":
        run_inspect_output()
    elif args.command == "search-candidates":
        rows, output_path = run_search_candidates(args)
        print(f"Wrote {len(rows)} candidate rows to {output_path}")
    elif args.command == "review-candidates":
        rows, output_path, searched_result = run_review_candidates(args)
        print(f"Wrote {len(rows)} reviewed candidate rows to {output_path}")
        print(
            f"Wrote {searched_result['searched_count']} searched video rows to "
            f"{searched_result['output_path']}"
        )
    elif args.command == "merge-video-list":
        result = run_merge_video_list(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "import-microlens":
        from .microlens_config import load_microlens_config
        from .microlens_importer import run_import

        result = run_import(
            load_microlens_config(args.config),
            scope=args.scope,
            force=args.force,
            rebuild_selection=args.rebuild_selection,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["failed"]:
            raise RuntimeError(
                f"MicroLens import completed with {result['failed']} processing failures"
            )


def run_process_batch(args: argparse.Namespace) -> tuple[list[dict[str, str]], Path]:
    processing_config = load_processing_config(VIDEO_CONFIG_PATH)
    data_root = resolve_assets_root()
    output_root = resolve_output_root()
    manifest_input = Path(args.manifest)
    manifest_output = Path(args.manifest)
    rows = read_existing_manifest(manifest_output)
    seen_content_ids = {str(row.get("content_id") or "") for row in rows}
    with manifest_input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    deferred_rows: list[tuple[int, str, str]] = []
    for index, row in enumerate(input_rows, start=1):
        try:
            content_id = indexed_content_id(row, index)
            url = row_to_url(row)
        except ValueError as exc:
            raise ValueError(f"Invalid batch row {index}: {exc}") from exc
        paths = build_content_paths(
            data_root=data_root,
            content_id=content_id,
            url=url,
            output_root=output_root,
            shot_interval=shot_interval_from_config(processing_config),
        )
        if not args.force and content_is_complete(paths, processing_config):
            source_video = Path(paths.video_path)
            if source_video.exists():
                source_video.unlink()
            remove_legacy_raw_artifacts(paths)
            print(f"[{index}/{len(input_rows)}] Skipping completed {content_id}: {url}")
            row_to_write = manifest_row(paths)
            append_manifest_row(row_to_write, manifest_output, rows, seen_content_ids)
            continue
        print(f"[{index}/{len(input_rows)}] Processing {content_id}: {url}")
        try:
            row_to_write = process_one(
                name=content_id,
                url=url,
                data_root=data_root,
                lang=args.lang,
                config_path=VIDEO_CONFIG_PATH,
                output_root=output_root,
                download_metadata=False,
                write_missing_metadata=False,
                bot_check_max_retries=0,
                bot_check_retry_delay_sec=0,
            )
            append_manifest_row(row_to_write, manifest_output, rows, seen_content_ids)
        except Exception as exc:
            if args.fail_fast:
                raise
            if is_youtube_bot_check_error(exc):
                deferred_rows.append((index, content_id, url))
                print(
                    f"[{index}/{len(input_rows)}] Deferred bot-check failure {content_id}: {exc}",
                    file=sys.stderr,
                )
                print(
                    f"[{index}/{len(input_rows)}] Sleeping {BOT_CHECK_DEFERRED_COOLDOWN_SEC // 60} minutes "
                    "before continuing batch.",
                    file=sys.stderr,
                )
                time.sleep(BOT_CHECK_DEFERRED_COOLDOWN_SEC)
            elif isinstance(exc, FileNotFoundError):
                print(f"[{index}/{len(input_rows)}] Skipping failed {content_id}: {exc}", file=sys.stderr)
            else:
                raise
    if deferred_rows:
        print(f"Retrying {len(deferred_rows)} deferred bot-check videos.")
    for retry_index, (original_index, content_id, url) in enumerate(deferred_rows, start=1):
        print(f"[deferred {retry_index}/{len(deferred_rows)}] Retrying {content_id}: {url}")
        try:
            row_to_write = process_one(
                name=content_id,
                url=url,
                data_root=data_root,
                lang=args.lang,
                config_path=VIDEO_CONFIG_PATH,
                output_root=output_root,
                download_metadata=False,
                write_missing_metadata=False,
                bot_check_max_retries=0,
                bot_check_retry_delay_sec=0,
            )
            append_manifest_row(row_to_write, manifest_output, rows, seen_content_ids)
        except Exception as exc:
            if args.fail_fast:
                raise
            if is_youtube_bot_check_error(exc) or isinstance(exc, FileNotFoundError):
                print(
                    f"[deferred {retry_index}/{len(deferred_rows)}] "
                    f"Skipping failed {content_id} from row {original_index}: {exc}",
                    file=sys.stderr,
                )
            else:
                raise
    return rows, manifest_output


def run_create_manifest(args: argparse.Namespace) -> tuple[list[dict[str, str]], Path]:
    processing_config = load_processing_config(VIDEO_CONFIG_PATH)
    data_root = resolve_assets_root()
    output_root = resolve_output_root()
    rows = manifest_rows_from_txt(
        txt_file_path=args.list_file_path,
        data_root=data_root,
        config=processing_config,
        output_root=output_root,
    )
    manifest_output = resolve_manifest_output_path(output_root, args.output)
    write_manifest(rows, manifest_output)
    return rows, manifest_output


def manifest_rows_from_txt(
    txt_file_path: str | Path,
    data_root: str | Path,
    config: dict[str, object],
    output_root: str | Path | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for list_id, url in read_txt_video_list(txt_file_path):
        video_id = video_id_from_url(url)
        if not video_id:
            raise ValueError(f"Could not parse YouTube video id from URL for {list_id}: {url}")
        content_id = normalize_content_id(f"{list_id}_{video_id}")
        paths = build_content_paths(
            data_root=data_root,
            content_id=content_id,
            url=url,
            output_root=output_root,
            shot_interval=shot_interval_from_config(config),
        )
        rows.append(
            manifest_row(paths)
        )
    if not rows:
        raise ValueError(f"No videos found in TXT file: {txt_file_path}")
    return rows


def read_txt_video_list(txt_file_path: str | Path) -> list[tuple[str, str]]:
    path = Path(txt_file_path)
    current_category = ""
    rows: list[tuple[str, str]] = []
    seen_list_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
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
            rows.append((list_id, url))
    return rows


def run_get_metadata(args: argparse.Namespace) -> dict[str, int]:
    from . import utils

    utils.proxy_setup()
    config = load_processing_config(VIDEO_CONFIG_PATH)
    output_root = resolve_output_root()
    input_manifest_path = Path(args.manifest)
    rows = read_existing_manifest(input_manifest_path)
    if not rows:
        raise ValueError(f"No manifest rows found: {input_manifest_path}")

    written = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        content_id = indexed_content_id(row, index)
        url = row_to_url(row)
        metadata_path = metadata_json_path_for_row(
            row=row,
            content_id=content_id,
            url=url,
            config=config,
            output_root=output_root,
        )
        if metadata_path.exists() and not args.force:
            skipped += 1
            continue
        metadata = collect_url_metadata(url=url, fallback_name=content_id)
        write_metadata_json(metadata, metadata_path)
        written += 1
    return {"written": written, "skipped": skipped}


def run_inspect_output(output_root: str | Path | None = None) -> dict[str, object]:
    root = Path(output_root) if output_root is not None else resolve_output_root()
    config = load_processing_config(VIDEO_CONFIG_PATH)
    shot_interval = shot_interval_from_config(config)
    report = inspect_output(root, shot_interval)
    print_output_inspection(report)

    cleanup_candidates = report["cleanup_candidates"]
    if cleanup_candidates:
        while True:
            answer = input(
                f"Delete metadata and Ref JSONL for {len(cleanup_candidates)} "
                "content(s) without resized keyframes? [Y/N]: "
            ).strip().lower()
            if answer in {"y", "n"}:
                break
            print("Please enter Y or N.")
        if answer == "y":
            for content_id in cleanup_candidates:
                (root / "asset" / "metadata" / f"{content_id}.json").unlink(
                    missing_ok=True
                )
                (
                    root
                    / ref_jsonl_relative_path(shot_interval)
                    / f"{content_id}_ref.jsonl"
                ).unlink(missing_ok=True)
            report["deleted"] = list(cleanup_candidates)
            print(f"Deleted metadata/Ref JSONL for {len(cleanup_candidates)} content(s).")
        else:
            print("Cleanup cancelled.")
    return report


def inspect_output(output_root: str | Path, shot_interval: str = "fixed_30s") -> dict[str, object]:
    root = Path(output_root)
    metadata_dir = root / "asset" / "metadata"
    resized_dir = root / resized_keyframes_relative_path(shot_interval)
    ref_dir = root / ref_jsonl_relative_path(shot_interval)

    metadata_ids = {path.stem for path in metadata_dir.glob("*.json") if path.is_file()}
    resized_ids = {path.name for path in resized_dir.iterdir() if path.is_dir()} if resized_dir.is_dir() else set()
    ref_ids = (
        {path.name.removesuffix("_ref.jsonl") for path in ref_dir.glob("*_ref.jsonl") if path.is_file()}
        if ref_dir.is_dir()
        else set()
    )

    missing: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    cleanup_candidates: list[str] = []
    for content_id in sorted(metadata_ids):
        keyframe_dir = resized_dir / content_id
        actual_png = resized_png_names(keyframe_dir)
        ref_path = ref_dir / f"{content_id}_ref.jsonl"
        keyframes_missing = not keyframe_dir.is_dir() or not actual_png
        ref_missing = not ref_path.is_file() or ref_path.stat().st_size == 0

        if keyframes_missing or ref_missing:
            missing.append(
                {
                    "content_id": content_id,
                    "keyframes_missing": keyframes_missing,
                    "ref_missing": ref_missing,
                }
            )
        if keyframes_missing:
            cleanup_candidates.append(content_id)
        if keyframes_missing or ref_missing:
            continue

        try:
            expected_png = expected_ref_keyframe_names(ref_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            mismatches.append({"content_id": content_id, "missing_png": [], "extra_png": [], "error": str(exc)})
            continue

        missing_png = sorted(expected_png - actual_png)
        extra_png = sorted(actual_png - expected_png)
        if missing_png or extra_png:
            mismatches.append(
                {
                    "content_id": content_id,
                    "missing_png": missing_png,
                    "extra_png": extra_png,
                    "error": "",
                }
            )

    return {
        "output_root": root,
        "metadata_count": len(metadata_ids),
        "resized_count": len(resized_ids),
        "ref_count": len(ref_ids),
        "missing": missing,
        "mismatches": mismatches,
        "orphan_resized": sorted(resized_ids - metadata_ids),
        "orphan_ref": sorted(ref_ids - metadata_ids),
        "cleanup_candidates": cleanup_candidates,
        "deleted": [],
    }


def resized_png_names(keyframe_dir: Path) -> set[str]:
    if not keyframe_dir.is_dir():
        return set()
    return {path.name for path in keyframe_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"}


def expected_ref_keyframe_names(ref_path: Path) -> set[str]:
    expected: set[str] = set()
    with ref_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{ref_path}: line {line_number} must be a JSON object")
            if record.get("_type") == "video_metadata":
                continue
            timeline = record.get("timeline")
            if not isinstance(timeline, list):
                raise ValueError(f"{ref_path}: line {line_number} timeline must be a list")
            for shot_index, shot in enumerate(timeline):
                if not isinstance(shot, dict) or shot.get("timestamp") is None:
                    raise ValueError(
                        f"{ref_path}: line {line_number} timeline[{shot_index}] has no valid timestamp"
                    )
                try:
                    timestamp = int(round(float(shot["timestamp"])))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{ref_path}: line {line_number} timeline[{shot_index}] has an invalid timestamp"
                    ) from exc
                expected.add(f"{timestamp:04d}.png")
    return expected


def print_output_inspection(report: dict[str, object]) -> None:
    missing = report["missing"]
    print("[DATA MISSING]")
    if not missing:
        print("  (none)")
    for item in missing:
        statuses = []
        if item["keyframes_missing"]:
            statuses.append("resized_keyframes=missing/empty")
        if item["ref_missing"]:
            statuses.append("ref_jsonl=missing/empty")
        print(f"  {item['content_id']}: {', '.join(statuses)}")

    mismatches = report["mismatches"]
    print("[KEYFRAME MISMATCH]")
    if not mismatches:
        print("  (none)")
    for item in mismatches:
        if item["error"]:
            print(f"  {item['content_id']}: error={item['error']}")
            continue
        print(f"  {item['content_id']}:")
        print(f"    missing_png={item['missing_png'] or []}")
        print(f"    extra_png={item['extra_png'] or []}")

    print("[ORPHAN OUTPUT]")
    print(f"  resized_keyframes={report['orphan_resized'] or []}")
    print(f"  ref_jsonl={report['orphan_ref'] or []}")

    error_count = sum(1 for item in mismatches if item["error"])
    mismatch_count = len(mismatches) - error_count
    print("[SUMMARY]")
    print(
        f"  metadata={report['metadata_count']}, resized_keyframes={report['resized_count']}, "
        f"ref_jsonl={report['ref_count']}, data_missing={len(missing)}, "
        f"keyframe_mismatch={mismatch_count}, inspection_errors={error_count}"
    )


def run_search_candidates(args: argparse.Namespace) -> tuple[list[dict[str, str]], Path]:
    from . import utils

    utils.proxy_setup()
    search_config = load_config(SEARCH_CONFIG_PATH)
    youtube = build_youtube_client(config_path=VIDEO_CONFIG_PATH)
    rows = collect_video_candidates(youtube, search_config)
    output_path = Path(args.output)
    write_dict_csv(
        rows,
        output_path,
        fieldnames=["video_id", "url", "seed_query", "seed_group", "seed_category", "search_language"],
    )
    return rows, output_path


def run_review_candidates(args: argparse.Namespace) -> tuple[list[dict[str, object]], Path, dict[str, object]]:
    allowed_languages = tuple(
        language.strip()
        for language in str(args.allowed_languages).split(",")
        if language.strip()
    )
    if not allowed_languages:
        raise ValueError("--allowed-languages must contain at least one language code")
    candidates = read_dict_csv(args.candidates)
    output_path = Path(args.output)
    ensure_review_csv_schema(output_path)
    existing_rows = read_dict_csv(output_path) if output_path.exists() and output_path.stat().st_size > 0 else []
    rows = enrich_candidates(
        candidates,
        script_dir=args.script_dir,
        allowed_languages=allowed_languages,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        min_script_chars=args.min_script_chars,
        show_progress=True,
        sleep_sec=args.sleep_sec,
        existing_rows=existing_rows,
        output_path=output_path,
    )
    if not output_path.exists():
        write_dict_csv(rows, output_path)
    searched_result = write_searched_video_list(rows, args.searched_list_output)
    return rows, output_path, searched_result


def run_merge_video_list(args: argparse.Namespace) -> dict[str, object]:
    return merge_video_lists(
        manual_list_path=args.manual,
        searched_list_path=args.searched,
        output_path=args.output,
    )


def metadata_json_path_for_row(
    row: dict[str, object],
    content_id: str,
    url: str,
    config: dict[str, object],
    output_root: str | Path | None = None,
) -> Path:
    root = Path(output_root) if output_root else resolve_output_root()
    return root / "asset" / "metadata" / f"{content_id}.json"


def read_existing_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return []
    return read_manifest_rows(manifest_path)


def append_manifest_row(
    row: dict[str, str],
    output_path: str | Path,
    rows: list[dict[str, str]],
    seen_content_ids: set[str],
) -> None:
    content_id = str(row.get("content_id") or "")
    if content_id in seen_content_ids:
        for existing in rows:
            if str(existing.get("content_id") or "") == content_id:
                existing.clear()
                existing.update(row)
                break
    else:
        rows.append(row)
        seen_content_ids.add(content_id)
    write_manifest(rows, output_path)


def load_pipeline_env() -> None:
    load_dotenv("config/.env")


def resolve_output_root() -> Path:
    load_pipeline_env()
    value = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not value:
        raise ValueError("OUTPUT_SAVE_PATH is required in config/.env or the environment")
    return custom_output_root(value)


def manifest_path() -> Path:
    return DEFAULT_MANIFEST_PATH


def resolve_manifest_output_path(output_root: str | Path, output: str | Path) -> Path:
    output_path = Path(output)
    if output_path.is_absolute():
        return output_path
    return Path(output_root) / output_path


def resolve_assets_root() -> str:
    load_pipeline_env()
    value = os.getenv("LINUX_ASSETS_SAVE_PATH", "").strip()
    if not value:
        raise ValueError("LINUX_ASSETS_SAVE_PATH is required in config/.env or the environment")
    return value


def indexed_content_id(row: dict[str, object], index: int) -> str:
    explicit_content_id = row_to_explicit_content_id(row)
    if explicit_content_id:
        return explicit_content_id
    video_id = row_to_video_id(row)
    if not video_id:
        raise ValueError("batch row must contain content_id, name, video_id, or URL column")
    return normalize_content_id(f"{index:03d}_{video_id}")


def row_to_explicit_content_id(row: dict[str, object]) -> str:
    for key in ("content_id", "content id", "name"):
        value = row_value(row, key)
        if value:
            return normalize_content_id(value)
    return ""


def row_value(row: dict[str, object], target_key: str) -> str:
    normalized_target = normalize_row_key(target_key)
    for key, value in row.items():
        if normalize_row_key(str(key)) == normalized_target and str(value or "").strip():
            return str(value).strip()
    return ""


def normalize_row_key(key: str) -> str:
    return key.strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_")


def content_is_complete(paths, config: dict[str, object] | None = None) -> bool:
    ref_path = Path(paths.ref_jsonl)
    all_frames_dir = Path(paths.all_frames_dir)
    resized_keyframes_dir = Path(paths.resized_keyframes_dir)
    outputs_exist = (
        ref_path.exists()
        and ref_path.stat().st_size > 0
        and Path(selected_filtered_timestamp_json(paths)).exists()
        and Path(paths.video_480p_path).exists()
    )
    if not outputs_exist:
        return False
    processing_config = config or load_processing_config(VIDEO_CONFIG_PATH)
    asr_config = processing_config.get("asr_config", {})
    if isinstance(asr_config, dict) and asr_config.get("enabled") is True:
        asr_ref_path = Path(paths.assets_path) / "ASR_Ref.json"
        if not is_nonempty_file(asr_ref_path):
            return False
    if not resized_keyframes_are_complete(
        selected_filtered_timestamp_json(paths),
        all_frames_dir,
        resized_keyframes_dir,
        resized_keyframe_resolution(processing_config),
    ):
        return False

    try:
        from .video_processor import extracted_frames_are_current, validate_480p_video

        validate_480p_video(paths.video_480p_path)
        if not extracted_frames_are_current(paths.video_480p_path, paths.all_frames_dir):
            return False
    except Exception:
        return False
    return True


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
