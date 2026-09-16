"""One compact, terminal failure record per scene or summary."""

import json
from pathlib import Path
import shutil

from artifact_io import atomic_write_jsonl
from pipeline_runtime import read_jsonl


class FailureLog:
    def __init__(self, directory):
        directory = Path(directory)
        self.path = directory / "failure.jsonl"
        self.rows = {}
        saved = read_jsonl(self.path) if self.path.is_file() else []
        for row in saved:
            self._remember(row["content_id"], row.get("scene_idx"), row["error"])
        legacy = directory / "failures"
        if legacy.is_dir():
            for path in sorted(legacy.glob("*.jsonl")):
                for row in read_jsonl(path):
                    self._remember(row.get("content_id", path.stem), row.get("scene_idx"),
                                   row.get("error") or "generation failed")
            self._rewrite()
            shutil.rmtree(legacy)
        elif saved != list(self.rows.values()):
            self._rewrite()
        # The new policy deliberately discards unfinished generation/repair state.
        for name in (".recovery", ".pending", ".checkpoints"):
            path = directory / name
            if path.is_dir():
                shutil.rmtree(path)
        for name in (".pending-contents.json", ".completed-contents.json"):
            (directory / name).unlink(missing_ok=True)

    def _remember(self, content_id, scene_idx, error):
        row = {"content_id": str(content_id), "error": str(error)}
        if scene_idx is not None:
            row["scene_idx"] = scene_idx
        self.rows[(row["content_id"], scene_idx)] = row
        return row

    def _rewrite(self):
        if self.rows:
            atomic_write_jsonl(self.path, self.rows.values(), durable=False)
        else:
            self.path.unlink(missing_ok=True)

    def contains(self, content_id, scene_idx):
        return (str(content_id), scene_idx) in self.rows

    def record(self, content_id, scene_idx, error):
        key = (str(content_id), scene_idx)
        previous = self.rows.get(key)
        row = self._remember(content_id, scene_idx, error)
        if previous == row:
            return
        if previous is not None:
            self._rewrite()
            return
        # Append new failures in O(1); do not rewrite a growing run-wide file per scene.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def clear_contents(self, content_ids):
        selected = {str(cid) for cid in content_ids}
        remaining = {key: row for key, row in self.rows.items() if key[0] not in selected}
        if remaining != self.rows:
            self.rows = remaining
            self._rewrite()

    def count(self, content_ids):
        selected = {str(cid) for cid in content_ids}
        return sum(cid in selected for cid, _ in self.rows)
