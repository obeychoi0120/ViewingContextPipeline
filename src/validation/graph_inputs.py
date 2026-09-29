"""Frozen BGE features and ragged, scene-local role graphs; no learned item vectors."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile

import numpy as np

from arm_registry import select_arms
from artifact_io import fingerprint
from extraction.structured_output import validate_graph_structure, OutputValidationError, without_receiver
from model_provenance import local_model_identity
from artifact_io import read_json, write_json
from validation.graph_context import GRAPH_ARMS
from validation.shared_cache import SharedCache, checksum

VERSION = "role-graph-inputs/v2"
ROLES = ("actor", "target", "tool", "receiver", "location")
# Keep the receiver tensor slot for model/checkpoint shape compatibility; input
# normalization disables its edges and always supplies the absent state.
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
    if arm.model:
        directory = context.scene_arm_dir(arm.scene_arm)
        for item in cohort["catalog"]:
            path = directory / f"{item['content_id']}.jsonl"
            sources.append([item["content_id"], checksum(path) if path.is_file() else None])
    titles = [r["title"].strip() for r in cohort["metadata_titles"]] if arm.uses_title else []
    return fingerprint(
        {
            "version": VERSION,
            "source": arm.scene_arm,
            "files": sources,
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

            g = without_receiver(row.get("scene_graph"))
            if not isinstance(g, dict):
                reject("raw")
                continue
            if row.get("warning"):
                reject("warning", str(row["warning"]))
                continue
            if "actions" not in g:
                reject("unsupported_format")
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
                text = json.dumps(
                    {"kind": entity["name"], "attributes": sorted(entity["attributes"])},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                data["nodes"].append(text_id(text))
                data["types"].append(0)
            for action in g["actions"]:
                node = len(data["nodes"])
                data["nodes"].append(text_id(action["action"]))
                data["types"].append(1)
                states = [node]
                for role, name in enumerate(ROLES):
                    value = action.get(name, "unknown" if name == "location" else None)
                    if value in lookup:
                        data["edges"].extend(
                            [[lookup[value], node, role], [node, lookup[value], role + 5]]
                        )
                        states.append(-1)
                    else:
                        states.append(1 if value == "unknown" else 0)
                data["missing"].append(states)
            context_text = json.dumps(
                {
                    k: sorted(g[k]) if k == "topics" else g[k]
                    for k in ("medium", "format", "topics")
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            data["contexts"].append(text_id(context_text))
            data["scene_offsets"].append(len(data["nodes"]))
            data["edge_offsets"].append(len(data["edges"]))
            data["missing_offsets"].append(len(data["missing"]))
        data["video_offsets"].append(len(data["contexts"]))
        report["title_used"] = int(data["titles"][-1] != 0)
        report["zero_scene_video"] = int(arm.model is not None and not report["valid_scenes"])
        report["empty_video"] = int(not report["valid_scenes"] and not report["title_used"])
        stats.update(report)
        videos.append({"content_id": cid, **dict(report)})
    arrays = {name: np.asarray(value, dtype=np.int64) for name, value in data.items()}
    arrays["edges"] = arrays["edges"].reshape(-1, 3)
    arrays["missing"] = arrays["missing"].reshape(-1, 6)
    return texts, arrays, {"counts": dict(stats), "videos": videos, "excluded_scenes": excluded}


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


def bundle_valid(directory, signature):
    try:
        manifest = read_json(directory / "manifest.json")
        return (
            manifest["input_hash"] == signature
            and manifest["version"] == VERSION
            and set(manifest["checksums"]) == {f"{n}.npy" for n in ARRAYS} | {"statistics.json"}
            and all(
                checksum(directory / n) == digest for n, digest in manifest["checksums"].items()
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(context, *, target, force=False):
    from validation.steps import validation_config
    from validation.selection import prepare_validation_cohort

    if set(target) - GRAPH_ARMS:
        raise ValueError("unsupported graph arm")
    context.initialize()
    cohort = prepare_validation_cohort(context)
    settings = validation_config(context).encoder
    pending, reused = [], []
    for name, arm in select_arms(context.config, target).items():
        signature = source_identity(context, cohort, arm)
        destination = context.representations_dir / f"{name}_embeddings"
        cache = SharedCache(context, "graph_inputs", signature)
        if not force and bundle_valid(destination, signature):
            reused.append(name)
            continue
        if not force:
            with tempfile.TemporaryDirectory() as temp:
                temp = Path(temp)
                if cache.restore(temp) and bundle_valid(temp, signature):
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
        arm = select_arms(context.config, [name])[name]
        if source_identity(context, cohort, arm) != signature:
            raise ValueError(f"Graph inputs changed during preparation: {name}; rerun preparation")
        destination = context.representations_dir / f"{name}_embeddings"
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
                "input_hash": signature,
                "arm": name,
                "catalog": cohort["catalog"],
                "feature_count": len(texts),
                "checksums": {p.name: checksum(p) for p in temp.iterdir()},
            }
            write_json(temp / "manifest.json", manifest)
            for p in temp.iterdir():
                if p.name != "manifest.json":
                    p.replace(destination / p.name)
            (temp / "manifest.json").replace(destination / "manifest.json")
        SharedCache(context, "graph_inputs", signature).publish(
            destination,
            [f"{n}.npy" for n in ARRAYS] + ["manifest.json", "statistics.json"],
            origin={"run_id": context.run_id},
            replace_corrupt=not force,
        )
    return {
        "stage": "embed-representations",
        "representation_mode": "graph",
        "generated_arms": [p[0] for p in pending],
        "reused_arms": reused,
    }


def verify(context, cohort, arms):
    for name, arm in select_arms(context.config, list(arms)).items():
        signature = source_identity(context, cohort, arm)
        path = context.representations_dir / f"{name}_embeddings"
        if not bundle_valid(path, signature):
            raise ValueError(
                f"missing/stale/corrupt graph inputs: {name}; rerun embed-representations"
            )
        if read_json(path / "manifest.json")["catalog"] != cohort["catalog"]:
            raise ValueError("graph catalog mismatch")


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
