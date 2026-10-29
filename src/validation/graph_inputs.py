"""Frozen BGE features and ragged, scene-local role graphs; no learned item vectors."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile

import numpy as np

from arm_registry import RECOMMENDATION_ARM_CONTRACT, feature_source, select_arms
from artifact_io import fingerprint
from extraction.structured_output import validate_graph_structure, OutputValidationError
from extraction.scene_storage import read_scene_records, validate_scene_payload, recover_scene_graph
from model_provenance import local_model_identity
from artifact_io import read_json, write_json
from validation.shared_cache import SharedCache, checksum

VERSION = "scene-bge-inputs/v1"
ROLES = ("actor", "target", "tool", "receiver", "place")
# Keep the receiver tensor slot for model/checkpoint shape compatibility.
# Current receiver-free input always supplies the absent state.
ARRAYS = (
    "features",
    "nodes",
    "types",
    "scene_offsets",
    "video_offsets",
    "contexts",
    "edges",
    "edge_offsets",
    "missing",
    "missing_offsets",
    "titles",
    "scene_keys",
    "node_keys",
)


def source_identity(context, cohort, arm):
    catalog, metadata = cohort["catalog"], cohort["metadata_titles"]
    if len(catalog) != len(metadata) or any(
        str(item["content_id"]) != str(title["content_id"])
        or str(item["item_id"]) != str(title["item_id"])
        for item, title in zip(catalog, metadata, strict=True)
    ):
        raise ValueError("graph metadata/catalog order mismatch")
    sources = []
    generation_identity = None
    if arm.model:
        directory = context.scene_arm_dir(arm.scene_arm)
        identity_path = directory / "generation_identity.json"
        generation_identity = read_json(identity_path) if identity_path.exists() else None
        for item in cohort["catalog"]:
            path = directory / f"{item['content_id']}.jsonl"
            if path.is_file():
                read_scene_records(path)
            sources.append([item["content_id"], checksum(path) if path.is_file() else None])
    titles = [r["title"].strip() for r in cohort["metadata_titles"]] if arm.uses_title else []
    return fingerprint(
        {
            "version": VERSION,
            "arm_contract": RECOMMENDATION_ARM_CONTRACT,
            "source": arm.scene_arm,
            "files": sources,
            "generation_identity": generation_identity,
            "catalog": cohort["catalog"],
            "titles": titles,
            "encoder": context.config["validation"]["encoder"],
            "model": local_model_identity(context.path("models", "bge")),
        }
    )


def build_input(context, cohort, arm):
    texts, vocabulary = [""], {"": 0}

    def text_id(text):
        text = text.strip()
        if text not in vocabulary:
            vocabulary[text] = len(texts)
            texts.append(text)
        return vocabulary[text]

    data = {name: [] for name in ARRAYS if name != "features"}
    for name in ("scene_offsets", "video_offsets", "edge_offsets", "missing_offsets"):
        data[name].append(0)
    stats = Counter(
        dict.fromkeys(
            (
                "scene_count",
                "valid_scenes",
                "actionless",
                "raw",
                "warning",
                "failed",
                "unsupported_format",
                "invalid_structure",
                "missing_file",
                "title_used",
                "zero_scene_video",
                "empty_video",
            ),
            0,
        )
    )
    videos, excluded = [], []
    titles = cohort["metadata_titles"]
    for item_index, item in enumerate(cohort["catalog"]):
        cid = item["content_id"]
        data["titles"].append(text_id(titles[item_index]["title"]) if arm.uses_title else 0)
        report = Counter()
        path = context.scene_arm_dir(arm.scene_arm) / f"{cid}.jsonl" if arm.model else None
        rows, seen = [], set()
        if path is not None and path.is_file():
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or str(row.get("content_id")) != str(cid):
                    raise ValueError(f"Graph content ID mismatch: {path}")
                index = row.get("scene_idx")
                if type(index) is not int or index < 0 or index in seen:
                    raise ValueError(f"invalid/duplicate scene index: {path}")
                seen.add(index)
                rows.append(row)
        elif path is not None:
            report["missing_file"] += 1
        for row in sorted(rows, key=lambda r: r["scene_idx"]):
            report["scene_count"] += 1

            def reject(reason, detail=None):
                report[reason] += 1
                excluded.append(
                    {
                        "content_id": cid,
                        "scene_idx": row["scene_idx"],
                        "reason": reason,
                        **({"detail": detail} if detail else {}),
                    }
                )

            validate_scene_payload(path, row)
            if "error" in row:
                reject("failed", row["error"])
                continue
            expected = "description" if arm.representation == "description" else "scene_graph"
            if expected not in row:
                raise ValueError(f"scene representation mismatch: expected {expected}: {path}")
            if arm.representation == "description":
                data["contexts"].append(text_id(row["description"]))
                data["scene_keys"].append(json.dumps([str(cid), row["scene_idx"]]))
                for name, values in (("scene_offsets", "nodes"),
                                     ("edge_offsets", "edges"),
                                     ("missing_offsets", "missing")):
                    data[name].append(len(data[values]))
                report["valid_scenes"] += 1
                continue
            g, warnings = recover_scene_graph(row["scene_graph"], row["warning"])
            if not isinstance(g, dict):
                reject("raw")
                continue
            if warnings:
                reject("warning", str(warnings))
                continue
            try:
                validate_graph_structure(g)
            except OutputValidationError as exc:
                reject("invalid_structure", str(exc))
                continue
            report["valid_scenes"] += 1
            report["actionless"] += not bool(g["actions"])
            # Canonical order preserves IDs only as topology; IDs never enter BGE.
            entities = sorted(g["entities"], key=lambda e: e["id"])
            lookup = {}
            for entity in entities:
                lookup[entity["id"]] = len(data["nodes"])
                text = (
                    f"Name: {entity['name']}\n"
                    f"Attributes: {', '.join(sorted(entity['attributes']))}"
                )
                data["nodes"].append(text_id(text))
                data["types"].append(0)
                data["node_keys"].append(entity["id"])
            for action_index, action in enumerate(g["actions"]):
                node = len(data["nodes"])
                data["nodes"].append(text_id(action["action"]))
                data["types"].append(1)
                data["node_keys"].append(f"action_{action_index}")
                states = [node]
                for role, name in enumerate(ROLES):
                    value = None if name == "receiver" else action[name]
                    if value in lookup:
                        data["edges"].extend(
                            [[lookup[value], node, role], [node, lookup[value], role + 5]]
                        )
                        states.append(-1)
                    else:
                        states.append(1 if value == "unknown" else 0)
                data["missing"].append(states)
            context_text = (
                f"Format: {g['format']}\n"
                f"Medium: {g['medium']}\n"
                f"Topics: {', '.join(sorted(g['topics']))}"
            )
            data["contexts"].append(text_id(context_text))
            data["scene_keys"].append(json.dumps([str(cid), row["scene_idx"]]))
            data["scene_offsets"].append(len(data["nodes"]))
            data["edge_offsets"].append(len(data["edges"]))
            data["missing_offsets"].append(len(data["missing"]))
        data["video_offsets"].append(len(data["contexts"]))
        report["title_used"] = int(data["titles"][-1] != 0)
        report["zero_scene_video"] = int(arm.model is not None and not report["valid_scenes"])
        report["empty_video"] = int(not report["valid_scenes"] and not report["title_used"])
        stats.update(report)
        videos.append({"content_id": cid, **dict(report)})
    arrays = {
        name: np.asarray(value, dtype=np.str_ if name.endswith("_keys") else np.int64)
        for name, value in data.items()
    }
    arrays["edges"] = arrays["edges"].reshape(-1, 3)
    arrays["missing"] = arrays["missing"].reshape(-1, 6)
    return texts, arrays, {
        "counts": dict(stats), "videos": videos, "excluded_scenes": excluded,
        "scene_coverage": stats["valid_scenes"] / stats["scene_count"] if stats["scene_count"] else None,
    }


def _encode_shard(settings, texts, device, destination):
    from validation.features import BGETextEncoder

    encoder = BGETextEncoder(settings, device=device)
    np.save(destination, encoder.encode(texts))
    return encoder.last_truncation


def encode_texts(settings, texts):
    """Independent BGE replicas; deterministic partition and merge, no model downloads."""
    from validation.model import torch
    from validation.features import BGETextEncoder

    count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
    if count <= 1 or len(texts) < count:
        encoder = BGETextEncoder(settings)
        return encoder.encode(texts)
    shards = np.array_split(np.arange(len(texts)), count)
    with (
        tempfile.TemporaryDirectory() as temp,
        ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("spawn")) as pool,
    ):
        futures = [
            pool.submit(
                _encode_shard,
                settings,
                [texts[i] for i in shard],
                f"cuda:{gpu}",
                str(Path(temp) / f"{gpu}.npy"),
            )
            for gpu, shard in enumerate(shards)
        ]
        for future in futures:
            future.result()
        return np.concatenate([np.load(Path(temp) / f"{i}.npy") for i in range(count)])


def bundle_valid(directory, signature, embedding_dim):
    try:
        manifest = read_json(directory / "manifest.json")
        features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        return (
            manifest["input_hash"] == signature
            and manifest["version"] == VERSION
            and manifest.get("arm_contract") == RECOMMENDATION_ARM_CONTRACT
            and features.shape == (manifest["feature_count"], embedding_dim)
            and features.dtype == np.float32
            and set(manifest["checksums"]) == {f"{n}.npy" for n in ARRAYS} | {"statistics.json"}
            and all(
                checksum(directory / n) == digest for n, digest in manifest["checksums"].items()
            )
        )
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return False


def prepare(context, *, target, force=False):
    from validation.steps import validation_config
    from validation.selection import prepare_validation_cohort

    context.initialize()
    cohort = prepare_validation_cohort(context)
    settings = validation_config(context).encoder
    pending, reused = [], []
    arms = select_arms(context.config, target)
    sources = {feature_source(arm): arm for arm in arms.values()}
    for name, arm in sources.items():
        signature = source_identity(context, cohort, arm)
        destination = context.representations_dir / name
        cache = SharedCache(context, "scene_inputs", signature)
        if not force and bundle_valid(destination, signature, settings.embedding_dim):
            reused.append(name)
            continue
        if not force:
            with tempfile.TemporaryDirectory() as temp:
                temp = Path(temp)
                if cache.restore(temp) and bundle_valid(temp, signature, settings.embedding_dim):
                    destination.mkdir(parents=True, exist_ok=True)
                    for p in temp.iterdir():
                        if p.name != "cache.json":
                            shutil.copy2(p, destination / p.name)
                    reused.append(name)
                    continue
        texts, arrays, stats = build_input(context, cohort, arm)
        pending.append((name, signature, texts, arrays, stats))
    unique = sorted({t for _, _, texts, _, _ in pending for t in texts if t})
    encoded = (
        encode_texts(settings, unique)
        if unique
        else np.empty((0, settings.embedding_dim), np.float32)
    )
    if encoded.shape != (len(unique), settings.embedding_dim) or not np.isfinite(encoded).all():
        raise ValueError("invalid graph BGE features")
    lookup = {t: i for i, t in enumerate(unique)}
    for name, signature, texts, arrays, stats in pending:
        arm = sources[name]
        if source_identity(context, cohort, arm) != signature:
            raise ValueError(f"Graph inputs changed during preparation: {name}; rerun preparation")
        destination = context.representations_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as temp:
            temp = Path(temp)
            values = np.lib.format.open_memmap(
                temp / "features.npy",
                mode="w+",
                dtype="float32",
                shape=(len(texts), settings.embedding_dim),
            )
            values[0] = 0
            for i, t in enumerate(texts[1:], 1):
                values[i] = encoded[lookup[t]]
            values.flush()
            del values
            for key, array in arrays.items():
                np.save(temp / f"{key}.npy", array)
            write_json(temp / "statistics.json", stats)
            manifest = {
                "version": VERSION,
                "arm_contract": RECOMMENDATION_ARM_CONTRACT,
                "input_hash": signature,
                "source": name,
                "catalog": cohort["catalog"],
                "feature_count": len(texts),
                "checksums": {p.name: checksum(p) for p in temp.iterdir()},
            }
            write_json(temp / "manifest.json", manifest)
            for p in temp.iterdir():
                if p.name != "manifest.json":
                    p.replace(destination / p.name)
            (temp / "manifest.json").replace(destination / "manifest.json")
        SharedCache(context, "scene_inputs", signature).publish(
            destination,
            [f"{n}.npy" for n in ARRAYS] + ["manifest.json", "statistics.json"],
            origin={"run_id": context.run_id},
            replace_corrupt=not force,
        )
    return {
        "stage": "embed-representations",
        "arm_contract": RECOMMENDATION_ARM_CONTRACT,
        "selected_arms": list(arms),
        "generated_sources": [p[0] for p in pending],
        "reused_sources": reused,
    }


def verify(context, cohort, arms):
    selected = select_arms(context.config, list(arms))
    for name, arm in {feature_source(a): a for a in selected.values()}.items():
        signature = source_identity(context, cohort, arm)
        path = context.representations_dir / name
        if not bundle_valid(path, signature, context.config["validation"]["encoder"]["embedding_dim"]):
            raise ValueError(
                f"missing/stale/corrupt graph inputs: {name}; rerun embed-representations"
            )
        if read_json(path / "manifest.json")["catalog"] != cohort["catalog"]:
            raise ValueError("graph catalog mismatch")



def scene_segments_paths(directory, variant="true"):
    """(array, metadata) paths of a scene_segments sidecar variant ("true" or "shuffled")."""
    if variant not in ("true", "shuffled"):
        raise ValueError("scene segment variant must be true or shuffled")
    stem = "scene_segments" if variant == "true" else f"scene_segments_{variant}"
    directory = Path(directory)
    return directory / f"{stem}.npy", directory / f"{stem}.json"


def write_scene_segments(directory, pairs, statistics=None, variant="true"):
    """Attach a scene-segment incidence, [bundle scene row, segment id] pairs, to a valid bundle.

    Pairs are stored sorted by scene row; a segment holds >= 1 scenes of one video, each scene at most once.
    """
    from validation.graph_context import SCENE_SEGMENTS_VERSION

    directory = Path(directory)
    array_path, meta_path = scene_segments_paths(directory, variant)
    manifest = read_json(directory / "manifest.json")
    video_offsets = np.load(directory / "video_offsets.npy", allow_pickle=False)
    scene_count = int(video_offsets[-1])
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if (pairs < 0).any() or (pairs[:, 0] >= scene_count).any():
        raise ValueError("scene segments must pair existing bundle scene rows with segment ids >= 0")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("a scene may belong to a segment only once")
    scene_videos = np.repeat(np.arange(len(video_offsets) - 1), np.diff(video_offsets))
    ids, where = np.unique(pairs[:, 1], return_inverse=True)
    lows = np.full(len(ids), np.iinfo(np.int64).max)
    highs = np.full(len(ids), -1)
    np.minimum.at(lows, where, scene_videos[pairs[:, 0]])
    np.maximum.at(highs, where, scene_videos[pairs[:, 0]])
    if (lows != highs).any():
        raise ValueError("a segment must stay within one video")
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    np.save(array_path, pairs)
    write_json(
        meta_path,
        {
            "version": SCENE_SEGMENTS_VERSION,
            "variant": variant,
            "input_hash": manifest["input_hash"],
            "pairs": int(len(pairs)),
            "segments": int(len(ids)),
            "statistics": statistics,
            "checksum": checksum(array_path),
        },
    )


def scene_segments_identity(directory, variant="true"):
    """Validated sidecar metadata; raises if missing, stale or corrupt."""
    from validation.graph_context import SCENE_SEGMENTS_VERSION

    directory = Path(directory)
    array_path, meta_path = scene_segments_paths(directory, variant)
    if not meta_path.is_file():
        raise ValueError(f"missing scene segments ({variant}): {directory}; run python -m validation prepare-module")
    meta = read_json(meta_path)
    manifest = read_json(directory / "manifest.json")
    if (
        meta.get("version") != SCENE_SEGMENTS_VERSION
        or meta.get("input_hash") != manifest["input_hash"]
        or checksum(array_path) != meta.get("checksum")
    ):
        raise ValueError(f"stale scene segments ({variant}): {directory}; run python -m validation prepare-module")
    return meta


def load_scene_segments(directory, variant="true"):
    """The validated [scene row, segment] pairs of a bundle, sorted by scene row."""
    scene_segments_identity(directory, variant)
    pairs = np.load(scene_segments_paths(directory, variant)[0], allow_pickle=False)
    contexts = np.load(Path(directory) / "contexts.npy", mmap_mode="r", allow_pickle=False)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or (len(pairs) and pairs[:, 0].max() >= len(contexts)):
        raise ValueError(f"scene segments do not match bundle scenes: {directory}")
    return pairs

class GraphStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.arrays = {
            n: np.load(self.directory / f"{n}.npy", mmap_mode="r", allow_pickle=False)
            for n in ARRAYS
        }

    def __getitem__(self, name):
        return self.arrays[name]

    def __len__(self):
        return len(self["titles"])
