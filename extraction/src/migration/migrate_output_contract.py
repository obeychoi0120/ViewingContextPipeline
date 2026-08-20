from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MODES = ("fixed_15s", "fixed_30s", "shot_wise")


@dataclass(frozen=True)
class MigrationItem:
    source: Path
    target: Path
    transform: Callable[[bytes], bytes] | None = None

    def target_bytes(self) -> bytes:
        payload = self.source.read_bytes()
        return self.transform(payload) if self.transform else payload

    def hashes_and_size(self) -> tuple[str, str, int]:
        source_hash = sha256_file(self.source)
        if self.transform is None:
            return source_hash, source_hash, self.source.stat().st_size
        payload = self.target_bytes()
        return source_hash, sha256_bytes(payload), len(payload)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy output paths to the asset/viewing_context/video_profile contract."
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the validated migration. The default is a read-only dry run.",
    )
    return parser.parse_args(argv)


def graph_document_transform(source_scene_context_path: str) -> Callable[[bytes], bytes]:
    def transform(payload: bytes) -> bytes:
        document = json.loads(payload.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("video context document must be an object")
        if "context" not in document:
            if "profile" not in document:
                raise ValueError("video context document has neither profile nor context")
            document["context"] = document.pop("profile")
        document["source_scene_context_path"] = source_scene_context_path
        required = {
            "content_id",
            "source_scene_context_path",
            "context",
            "aggregation_warnings",
        }
        if set(document) != required:
            raise ValueError(f"invalid video context fields: {sorted(document)}")
        return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    return transform


def profile_transform(payload: bytes) -> bytes:
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, dict) or "profile" not in document:
        raise ValueError("video profile document must contain profile")
    return payload


def content_id_from_name(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise ValueError(f"unexpected filename {name!r}; expected suffix {suffix!r}")
    content_id = name[: -len(suffix)]
    if not content_id:
        raise ValueError(f"empty content ID in {name!r}")
    return content_id


def file_items(source_dir: Path, target_dir: Path) -> list[MigrationItem]:
    if not source_dir.is_dir():
        return []
    return [
        MigrationItem(path, target_dir / path.name)
        for path in sorted(source_dir.iterdir())
        if path.is_file()
    ]


def graph_items(
    source_dir: Path,
    target_dir: Path,
    *,
    source_suffixes: tuple[str, ...],
    target_suffix: str,
    scene_dir: Path,
    reference: bool,
) -> list[MigrationItem]:
    if not source_dir.is_dir():
        return []
    items: list[MigrationItem] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            continue
        matched_suffix = next(
            (suffix for suffix in source_suffixes if path.name.endswith(suffix)),
            None,
        )
        if matched_suffix is None:
            raise ValueError(f"unexpected graph filename: {path}")
        content_id = content_id_from_name(path.name, matched_suffix)
        scene_name = (
            f"{content_id}_scene_context_ref.jsonl"
            if reference
            else f"{content_id}_scene_context.jsonl"
        )
        source_scene_path = str(scene_dir / scene_name)
        items.append(
            MigrationItem(
                source=path,
                target=target_dir / f"{content_id}{target_suffix}",
                transform=graph_document_transform(source_scene_path),
            )
        )
    return items


def profile_items(source_dir: Path, target_dir: Path) -> list[MigrationItem]:
    if not source_dir.is_dir():
        return []
    items: list[MigrationItem] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = next(
            (
                candidate
                for candidate in ("_profile_gt.json", "_profile_ref.json", "_profile.json")
                if path.name.endswith(candidate)
            ),
            None,
        )
        if suffix is None:
            raise ValueError(f"unexpected video profile filename: {path}")
        content_id = content_id_from_name(path.name, suffix)
        items.append(
            MigrationItem(
                path,
                target_dir / f"{content_id}_profile.json",
                profile_transform,
            )
        )
    return items


def build_items(output_root: Path) -> list[MigrationItem]:
    items = file_items(
        output_root / "metadata",
        output_root / "asset" / "metadata",
    )
    legacy_root = output_root / "viewing_context"
    for mode in MODES:
        legacy_mode = legacy_root / mode
        asset_mode = output_root / "asset" / mode
        canonical_mode = output_root / "viewing_context" / "img_only" / mode
        items.extend(file_items(legacy_mode / "ref_jsonl", asset_mode / "ref_jsonl"))
        if (legacy_mode / "resized_keyframes").is_dir():
            for content_dir in sorted((legacy_mode / "resized_keyframes").iterdir()):
                if content_dir.is_dir():
                    items.extend(
                        file_items(
                            content_dir,
                            asset_mode / "resized_keyframes" / content_dir.name,
                        )
                    )

        scene_sources = {
            "ref": "scene_context_graph_ref",
            "qwen": "scene_context_graph_qwen",
            "mistral": "scene_context_graph_mistral",
            "gaussa_gemma4_e2b_v0_3": "scene_context_graph_gauss_gemma4_e2b",
        }
        for target_postfix, source_name in scene_sources.items():
            items.extend(
                file_items(
                    legacy_mode / source_name,
                    canonical_mode / f"scene_context_graph_{target_postfix}",
                )
            )

        graph_sources = {
            "ref": (
                ("video_profile_graph_ref", "video_profile_graph_ref_canonical"),
                ("_profile_graph_ref.json", "_profile_ref.json", "_context_graph_ref.json"),
                "_context_graph_ref.json",
                True,
            ),
            "qwen": (
                ("video_profile_graph_qwen",),
                ("_profile_graph_ond.json", "_profile_graph.json", "_context_graph_ond.json"),
                "_context_graph_ond.json",
                False,
            ),
            "mistral": (
                ("video_profile_graph_mistral",),
                ("_profile_graph_ond.json", "_profile_graph.json", "_context_graph_ond.json"),
                "_context_graph_ond.json",
                False,
            ),
            "gaussa_gemma4_e2b_v0_3": (
                ("video_profile_graph_gauss_gemma4_e2b",),
                ("_profile_graph_ond.json", "_profile_graph.json", "_context_graph_ond.json"),
                "_context_graph_ond.json",
                False,
            ),
        }
        for postfix, (source_names, suffixes, target_suffix, reference) in graph_sources.items():
            scene_dir = canonical_mode / f"scene_context_graph_{postfix}"
            for source_name in source_names:
                items.extend(
                    graph_items(
                        legacy_mode / source_name,
                        canonical_mode / f"video_context_graph_{postfix}",
                        source_suffixes=suffixes,
                        target_suffix=target_suffix,
                        scene_dir=scene_dir,
                        reference=reference,
                    )
                )

        for source_name in ("video_profile_gt", "video_profile_ref"):
            items.extend(
                profile_items(
                    legacy_mode / source_name,
                    output_root / "video_profile" / mode,
                )
            )

        failure_source = legacy_mode / "scene_context_graph_gauss_gemma4_e2b_failures"
        items.extend(
            file_items(
                failure_source,
                output_root
                / "failures"
                / "viewing_context"
                / "img_only"
                / mode
                / "scene_context_graph_gaussa_gemma4_e2b_v0_3",
            )
        )
    return items


def preflight(items: list[MigrationItem]) -> dict[str, object]:
    by_target: dict[Path, list[tuple[MigrationItem, str]]] = {}
    errors: list[str] = []
    mappings: list[dict[str, object]] = []
    for item in items:
        try:
            source_hash, target_hash, size = item.hashes_and_size()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{item.source}: {exc}")
            continue
        by_target.setdefault(item.target, []).append((item, target_hash))
        mappings.append(
            {
                "source": str(item.source),
                "target": str(item.target),
                "source_sha256": source_hash,
                "target_sha256": target_hash,
                "size": size,
            }
        )
    for target, candidates in by_target.items():
        hashes = {target_hash for _, target_hash in candidates}
        if len(hashes) != 1:
            errors.append(
                f"conflicting duplicate target {target}: "
                + ", ".join(str(item.source) for item, _ in candidates)
            )
        if target.exists() and sha256_file(target) not in hashes:
            errors.append(f"existing target differs: {target}")
    return {
        "schema_version": "output-contract-migration/v1",
        "mapping_count": len(mappings),
        "target_count": len(by_target),
        "errors": errors,
        "mappings": mappings,
    }


def apply_migration(items: list[MigrationItem], report: dict[str, object]) -> None:
    if report["errors"]:
        raise ValueError("migration preflight failed")
    expected_by_target = {
        Path(mapping["target"]): mapping["target_sha256"]
        for mapping in report["mappings"]
    }
    written: set[Path] = set()
    for item in items:
        if item.target in written:
            continue
        expected_hash = expected_by_target[item.target]
        if item.target.is_file() and sha256_file(item.target) == expected_hash:
            written.add(item.target)
            continue
        item.target.parent.mkdir(parents=True, exist_ok=True)
        temporary = item.target.with_name(f".{item.target.name}.migration.tmp")
        if item.transform is None:
            temporary.unlink(missing_ok=True)
            try:
                os.link(item.source, temporary)
            except OSError:
                shutil.copyfile(item.source, temporary)
        else:
            temporary.write_bytes(item.target_bytes())
        temporary.replace(item.target)
        if sha256_file(item.target) != expected_hash:
            raise OSError(f"target verification failed: {item.target}")
        written.add(item.target)

    for item in items:
        item.source.unlink(missing_ok=True)
    _remove_empty_legacy_dirs({item.source.parent for item in items})


def _remove_empty_legacy_dirs(directories: set[Path]) -> None:
    pending = sorted(directories, key=lambda path: len(path.parts), reverse=True)
    for directory in pending:
        current = directory
        while current.name and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.name != "output":
        raise ValueError(f"output root must end in 'output': {output_root}")
    items = build_items(output_root)
    report = preflight(items)
    print(
        json.dumps(
            {
                "mapping_count": report["mapping_count"],
                "target_count": report["target_count"],
                "error_count": len(report["errors"]),
                "errors": report["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["errors"]:
        return 1
    if not args.apply:
        return 0

    apply_migration(items, report)
    report_path = output_root / "reports" / "migration" / "output_contract_v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
