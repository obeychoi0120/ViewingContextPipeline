"""Move legacy MicroLens artifacts without rewriting feature/checkpoint identities."""

import argparse
import fcntl
from pathlib import Path
import sys

from artifact_io import atomic_write_json, read_json
from pipeline_runtime import DATASETS


DATASET = "microlens_100k"
JOURNAL = "dataset_layout_migration.json"
SCHEMA = "dataset-layout-migration/v1"


def _exists(path):
    return path.exists() or path.is_symlink()


def _identity(path):
    stat = path.lstat()
    return [stat.st_dev, stat.st_ino]


def _path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid migration path: {relative}")
    return root / path


def _plan(root):
    moves = []

    def group(parent):
        if not parent.is_dir():
            return
        children = sorted(p for p in parent.iterdir() if p.name not in DATASETS)
        if not children:
            return
        namespace = parent / DATASET
        if _exists(namespace):
            raise FileExistsError(f"migration destination already exists: {namespace}")
        for source in children:
            if source.is_symlink():
                raise ValueError(f"top-level symlink requires manual relocation: {source}")
            destination = namespace / source.name
            moves.append(
                {
                    "source": str(source.relative_to(root)),
                    "destination": str(destination.relative_to(root)),
                    "identity": _identity(source),
                    "status": "pending",
                }
            )

    group(root / "preparation")
    runs = root / "runs"
    if runs.is_dir():
        for run in sorted(p for p in runs.iterdir() if p.is_dir()):
            group(run)
    group(root / "shared_cache")
    return {"schema_version": SCHEMA, "dataset": DATASET, "moves": moves}


def _load_or_plan(root):
    journal = root / JOURNAL
    if not journal.exists():
        return _plan(root)
    plan = read_json(journal)
    if plan.get("schema_version") != SCHEMA or plan.get("dataset") != DATASET:
        raise ValueError(f"unsupported migration journal: {journal}")
    return plan


def _check(root, plan):
    for move in plan["moves"]:
        source, destination = (_path(root, move[key]) for key in ("source", "destination"))
        if _exists(source):
            if move["status"] == "moved" or _identity(source) != move["identity"]:
                raise ValueError(f"migration source changed: {source}")
            if _exists(destination):
                raise FileExistsError(f"migration conflict: {destination}")
        elif not (
            _exists(destination)
            and move["status"] in {"moving", "moved"}
            and (
                _identity(destination) == move["identity"]
                or any(
                    # Older journals may describe an atomically rewritten contract.
                    # Recognize that completed move without updating the file again.
                    contract["path"] == move["destination"]
                    and contract["compatible"]
                    and isinstance(contract.get("original"), dict)
                    and read_json(destination) == {**contract["original"], "dataset": DATASET}
                    for contract in plan.get("contracts", [])
                )
            )
        ):
            raise ValueError(f"missing or replaced migration artifact: {source}")


def migrate(artifacts_root, *, apply=False):
    root = Path(artifacts_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifacts root does not exist: {root}")

    def execute():
        plan = _load_or_plan(root)
        _check(root, plan)
        for move in plan["moves"]:
            print(f"[{move['status']}] {move['source']} -> {move['destination']}", flush=True)
        if not apply:
            print("[DRY RUN] No artifacts changed. Add --apply to move.", flush=True)
            return plan
        if not plan["moves"]:
            print("[MIGRATION] No legacy artifacts to move.", flush=True)
            return plan

        def save():
            atomic_write_json(root / JOURNAL, plan, durable=True)

        save()
        for move in plan["moves"]:
            source, destination = (_path(root, move[key]) for key in ("source", "destination"))
            if _exists(source):
                move["status"] = "moving"
                save()
                destination.parent.mkdir(parents=True, exist_ok=True)
                if _exists(destination):
                    raise FileExistsError(f"migration conflict: {destination}")
                # Rename only: never copy or delete a tree, and never cross filesystems.
                source.rename(destination)
            move["status"] = "moved"
            save()
        print(f"[MIGRATION] Complete. Journal: {root / JOURNAL}", flush=True)
        return plan

    if not apply:
        return execute()
    with (root / ".dataset_layout_migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return execute()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--apply", action="store_true", help="Apply the previewed directory moves.")
    args = parser.parse_args(argv)
    try:
        migrate(args.artifacts_root, apply=args.apply)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] dataset layout migration: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
