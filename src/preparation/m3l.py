"""Stream original MovieLens ratings into a source cohort of available M3L trailers."""

from collections import Counter
import csv
import json
import math
import shutil
import subprocess

from tqdm import tqdm

from artifact_io import atomic_write_jsonl, file_fingerprint, write_json, write_jsonl
from preparation.catalog import content_id_for_item, normalize_item_id


SCHEMA = "m3l-full-source/v1"
INTERPRETATION = "평가한 영화 이력의 다음 영화 예측"


def load_mapping(path):
    movies, remapped = set(), set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["newID", "oldID"]:
            raise ValueError("M3L mapping header must be newID,oldID (TSV)")
        for row in reader:
            if set(row) != {"newID", "oldID"} or any(v is None for v in row.values()):
                raise ValueError("malformed M3L mapping row")
            movie = normalize_item_id(row["oldID"])
            new = row["newID"]
            if not new.isdecimal() or movie in movies or int(new) in remapped:
                raise ValueError("invalid or duplicate M3L movie mapping")
            movies.add(movie)
            remapped.add(int(new))
    if not movies or remapped != set(range(len(movies))):
        raise ValueError("M3L newID must be contiguous and zero-based")
    return movies


def load_movie_titles(path):
    titles = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("::")
            if len(fields) != 3:
                raise ValueError(f"invalid movies.dat row {number}")
            movie = normalize_item_id(fields[0])
            if movie in titles:
                raise ValueError(f"duplicate movies.dat ID: {movie}")
            titles[movie] = fields[1].strip()
    return titles


def iter_ratings(path, movie_ids):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("::")
            if len(fields) != 4:
                raise ValueError(f"invalid ratings.dat row {number}")
            user, movie = (normalize_item_id(v) for v in fields[:2])
            rating = float(fields[2])
            if not math.isfinite(rating) or not 0.5 <= rating <= 5 or rating * 2 != int(rating * 2):
                raise ValueError(f"invalid rating at row {number}")
            timestamp = fields[3]
            if not timestamp.isdecimal() or int(timestamp) > (2**63 - 1) // 1000:
                raise ValueError(f"invalid timestamp at row {number}")
            if movie in movie_ids:
                yield {"user_id": user, "item_id": movie, "timestamp": int(timestamp) * 1000}


def probe_video(path):
    """Only media/decode errors are exclusions; process startup errors propagate."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode < 0 or result.returncode in (126, 127):
        raise RuntimeError(f"ffprobe runtime failed ({result.returncode}): {result.stderr.strip()}")
    if result.returncode:
        raise ValueError(f"ffprobe failed: {result.stderr.strip()}")
    doc = json.loads(result.stdout)
    duration = float(doc.get("format", {}).get("duration", "nan"))
    if (
        not any(s.get("codec_type") == "video" for s in doc.get("streams", []))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("requires a video stream and positive finite duration")
    return duration


def video_inventory(movie_ids, videos_dir):
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required; install ffmpeg before prepare-cohort")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"M3L videos directory does not exist: {videos_dir}")
    paths = {}
    for path in videos_dir.glob("*.mp4"):
        if not path.stem.isdecimal() or int(path.stem) <= 0:
            continue
        movie = str(int(path.stem))
        if movie in movie_ids:
            if movie in paths:
                raise ValueError(f"multiple video files normalize to movie {movie}")
            paths[movie] = path
    inventory, excluded = [], []
    for movie in tqdm(sorted(movie_ids, key=int), desc="Check M3L trailers", unit="video"):
        path = paths.get(movie)
        if path is None:
            excluded.append({"item_id": movie, "reason": "missing_video"})
            continue
        stat = path.stat()
        try:
            if not path.is_file() or stat.st_size == 0:
                raise ValueError("video must be a non-empty file")
            duration = probe_video(path)
        except (ValueError, subprocess.TimeoutExpired) as exc:
            excluded.append({"item_id": movie, "reason": "invalid_video", "error": str(exc)})
            continue
        current = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise RuntimeError(f"source video changed during probing: {path}")
        inventory.append(
            {
                "item_id": movie,
                "content_id": content_id_for_item(movie, "m3l_10m"),
                "source_video_path": str(path.resolve()),
                "source_file_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "duration_seconds": duration,
                "eligible": True,
                "exclusion_reasons": [],
            }
        )
    return inventory, excluded


def prepare_m3l_cohort(context, *, plan_only=False):
    root = context.path("data", "root")
    sources = {
        name: root / "metadata" / name
        for name in ("ratings.dat", "movies.dat", "id_mmrec_to_grouplens.tsv")
    }
    identities = {
        name: {"path": str(path), "sha256": file_fingerprint(path)}
        for name, path in sources.items()
    }
    movies = load_mapping(sources["id_mmrec_to_grouplens.tsv"])
    if len(movies) != context.source_counts["item_count"]:
        raise ValueError("M3L mapping cardinality mismatch")
    titles = load_movie_titles(sources["movies.dat"])
    if movies - titles.keys():
        raise ValueError("movies.dat lacks mapped M3L titles")
    inventory, excluded = ([], []) if plan_only else video_inventory(movies, root / "videos")
    available = movies if plan_only else {row["item_id"] for row in inventory}
    directory = context.cohort_dir
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "eligibility.json", {"schema_version": SCHEMA, "status": "blocked"})
    raw_users, raw_items, users, items = set(), set(), set(), set()
    raw_count = kept_count = 0
    excluded_counts = Counter()

    def rows():
        nonlocal raw_count, kept_count
        for row in tqdm(
            iter_ratings(sources["ratings.dat"], movies),
            total=context.source_counts["interaction_count"],
            desc="Read M3L ratings",
            unit="rating",
        ):
            raw_count += 1
            raw_users.add(row["user_id"])
            raw_items.add(row["item_id"])
            if row["item_id"] not in available:
                excluded_counts[row["item_id"]] += 1
                continue
            users.add(row["user_id"])
            items.add(row["item_id"])
            yield {"event_id": kept_count, **row}
            kept_count += 1
        observed = {
            "user_count": len(raw_users),
            "interaction_count": raw_count,
            "item_count": len(raw_items),
        }
        if observed != context.source_counts:
            raise ValueError(f"M3L source cardinality mismatch: {observed}")

    atomic_write_jsonl(directory / "events.jsonl", rows(), durable=True)
    if any(file_fingerprint(path) != identities[name]["sha256"] for name, path in sources.items()):
        raise RuntimeError("M3L metadata changed during preparation")
    inventory = [row for row in inventory if row["item_id"] in items]
    required = [
        {"item_id": item, "content_id": content_id_for_item(item, "m3l_10m")}
        for item in sorted(items, key=int)
    ]
    for row in excluded:
        row["interaction_count"] = excluded_counts[row["item_id"]]
    plan = {
        "schema_version": SCHEMA,
        "dataset": context.dataset,
        "experiment_config_version": context.config["experiment_config_version"],
        "metadata_missing_policy": context.config["validation"]["cohort"][
            "metadata_missing_policy"
        ],
        "source_counts": context.source_counts,
        "sources": identities,
        "rating_policy": "all_ratings",
        "missing_video_policy": "exclude_item_and_interactions",
        "interpretation": INTERPRETATION,
        "timestamp_unit": "milliseconds",
        "user_count": len(users),
        "interaction_count": kept_count,
        "item_count": len(items),
        "no_history_count": len(users),
        "excluded_item_count": len(excluded),
        "excluded_interaction_count": raw_count - kept_count,
        "events_sha256": file_fingerprint(directory / "events.jsonl"),
    }
    write_json(directory / "cohort_plan.json", plan)
    write_jsonl(directory / "required_items.jsonl", required)
    write_jsonl(directory / "excluded_items.jsonl", excluded)
    if not kept_count:
        raise ValueError(
            f"no M3L interactions with valid videos remain; see {directory / 'excluded_items.jsonl'}"
        )
    result = {
        "stage": "prepare-cohort",
        "dataset": context.dataset,
        "user_count": len(users),
        "catalog_size": len(items),
        "content_count": len(items),
        "catalog_scope": "full_source_catalog",
        "status": "planned",
    }
    if plan_only:
        print(f"[COHORT] Plan only; videos not checked: {directory}", flush=True)
        return result
    write_jsonl(directory / "item_inventory.jsonl", inventory)
    write_jsonl(
        directory / "catalog.jsonl",
        [
            {
                key: row[key]
                for key in ("item_id", "content_id", "source_video_path", "duration_seconds")
            }
            for row in inventory
        ],
    )
    write_jsonl(
        directory / "metadata_titles.jsonl",
        [{**r, "title": titles[r["item_id"]]} for r in required],
    )
    write_json(directory / "eligibility.json", {"schema_version": SCHEMA, "status": "ready"})
    print(
        f"[COHORT] {INTERPRETATION}; users={len(users)} ratings={kept_count} movies={len(items)} "
        f"excluded_movies={len(excluded)} excluded_ratings={raw_count - kept_count}",
        flush=True,
    )
    return {**result, "status": "ready"}
