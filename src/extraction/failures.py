"""Terminal scene failures per content, and summary failures in one JSONL file."""

import json
from pathlib import Path
import shutil

from artifact_io import atomic_write_jsonl
from pipeline_runtime import read_jsonl


class FailureLog:
    def __init__(self, directory, *, scenes=False):
        directory = Path(directory)
        self.scenes = scenes
        self.path = directory / ("failures" if scenes else "failures.jsonl")
        self.rows = {}
        self.by_content = {}
        # Read the current layout last so an interrupted migration cannot restore stale rows.
        per_content = sorted((directory / "failures").glob("*.jsonl"))
        sources = [directory / "failure.jsonl"]
        sources.extend([directory / "failures.jsonl", *per_content] if scenes
                       else [*per_content, directory / "failures.jsonl"])
        existing = {}
        for path in sources:
            if not path.is_file():
                continue
            existing[path] = read_jsonl(path)
            for row in existing[path]:
                self._remember(row.get("content_id", path.stem), row.get("scene_idx"),
                               row.get("error") or "generation failed",
                               row.get("raw_output", row.get("raw_response", "")))
        targets = ({self.path_for(cid): list(rows.values()) for cid, rows in self.by_content.items()}
                   if scenes else {self.path: list(self.rows.values())})
        for path, rows in targets.items():
            if existing.get(path, []) != rows:
                self._write(path, rows)
        # Remove old paths only after all migrated records have been published.
        for path in existing:
            if path not in targets:
                path.unlink()
        if not scenes and (directory / "failures").is_dir():
            shutil.rmtree(directory / "failures")
        # The one-attempt policy deliberately discards unfinished generation/repair state.
        for name in (".recovery", ".pending", ".checkpoints"):
            path = directory / name
            if path.is_dir():
                shutil.rmtree(path)
        for name in (".pending-contents.json", ".completed-contents.json"):
            (directory / name).unlink(missing_ok=True)

    def path_for(self, content_id):
        return self.path / f"{content_id}.jsonl" if self.scenes else self.path

    def _remember(self, content_id, scene_idx, error, raw_output):
        cid = str(content_id)
        row = {"content_id": cid}
        if self.scenes:
            if type(scene_idx) is not int or scene_idx < 0:
                raise ValueError(f"invalid failed scene index for {cid}: {scene_idx}")
            row["scene_idx"] = scene_idx
        else:
            scene_idx = None
        row.update(error=str(error), raw_output=raw_output if raw_output is not None else "")
        self.rows[(cid, scene_idx)] = row
        self.by_content.setdefault(cid, {})[scene_idx] = row
        return row

    @staticmethod
    def _write(path, rows):
        if rows:
            atomic_write_jsonl(path, rows, durable=False)
        else:
            path.unlink(missing_ok=True)

    def contains(self, content_id, scene_idx):
        return (str(content_id), scene_idx) in self.rows

    def remove(self, content_id, scene_idx):
        cid = str(content_id)
        key = (cid, scene_idx)
        if key not in self.rows:
            return
        rows = (self.by_content[cid].values() if self.scenes else self.rows.values())
        remaining = [row for row in rows
                     if (row["content_id"], row.get("scene_idx")) != key]
        self._write(self.path_for(cid), remaining)
        del self.rows[key]
        del self.by_content[cid][scene_idx]
        if not self.by_content[cid]:
            del self.by_content[cid]

    def record(self, content_id, scene_idx, error, raw_output=""):
        cid = str(content_id)
        previous = self.rows.get((cid, scene_idx))
        row = self._remember(cid, scene_idx, error, raw_output)
        if previous == row:
            return
        path = self.path_for(cid)
        if previous is not None:
            rows = self.by_content[cid].values() if self.scenes else self.rows.values()
            self._write(path, list(rows))
            return
        # New failures append in O(1), including runs with many failed scenes.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def clear_contents(self, content_ids):
        selected = {str(cid) for cid in content_ids} & self.by_content.keys()
        if not selected:
            return
        for cid in selected:
            for scene_idx in self.by_content.pop(cid):
                del self.rows[(cid, scene_idx)]
            if self.scenes:
                self.path_for(cid).unlink(missing_ok=True)
        if not self.scenes:
            self._write(self.path, list(self.rows.values()))

    def count(self, content_ids):
        return sum(len(self.by_content.get(str(cid), {})) for cid in set(content_ids))
