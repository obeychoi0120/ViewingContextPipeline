"""Stable per-user event sequences and explicit leave-one-out partitions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


DAY = 86_400_000
SCHEMA = "microlens-leave-one-out/v1"
PROTOCOL = "leave_one_out"


class EventTable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.users = sorted({r["user_id"] for r in self.rows})
        self.items = sorted({r["item_id"] for r in self.rows}, key=int)
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.by_user = defaultdict(list)
        for i, row in enumerate(self.rows):
            if row["event_id"] != i:
                raise ValueError("event IDs must preserve source row order")
            self.by_user[row["user_id"]].append(i)
        self.timestamps = np.asarray([r["timestamp"] for r in self.rows], dtype=np.int64)
        self.targets = np.asarray([self.item_index[r["item_id"]] + 1 for r in self.rows])
        self.history_ends = np.zeros(len(self.rows), dtype=np.int64)
        for ids in self.by_user.values():
            ids.sort(key=lambda i: (self.timestamps[i], self.rows[i].get("source_event_id", i)))
            self.history_ends[ids] = np.arange(len(ids))

    def history(self, event, limit=None):
        ids = self.by_user[self.rows[event]["user_id"]]
        end = int(self.history_ends[event])
        start = max(0, end - limit) if limit else 0
        return self.targets[ids[start:end]].tolist()

    def select(self, *, eligible=True):
        mask = np.ones(len(self.rows), dtype=bool)
        if eligible:
            mask &= self.history_ends > 0
        ids = np.flatnonzero(mask)
        return ids[np.lexsort((ids, self.timestamps[ids]))]

    def splits(self):
        from artifact_io import fingerprint

        phases = {name: [] for name in ("train", "validation", "test")}
        users = []
        for user in self.users:
            ids = self.by_user[user]
            if len(ids) < 5 or len(ids) > 13:
                raise ValueError("leave-one-out sequences must contain 5..13 events")
            train, validation, test = ids[:-2], ids[-2], ids[-1]
            phases["train"].extend(train)
            phases["validation"].append(validation)
            phases["test"].append(test)
            users.append({"user_id": user, "train": train, "validation": validation, "test": test})
        manifest = {
            "protocol": PROTOCOL,
            "users": users,
            "phases": {
                name: {
                    "event_ids": ids,
                    "raw_count": len(ids),
                    "eligible_count": sum(self.history_ends[i] > 0 for i in ids),
                    "no_history_count": sum(self.history_ends[i] == 0 for i in ids),
                }
                for name, ids in phases.items()
            },
        }
        # Convert NumPy counts to ordinary JSON integers.
        for phase in manifest["phases"].values():
            phase["eligible_count"] = int(phase["eligible_count"])
            phase["no_history_count"] = int(phase["no_history_count"])
        return [{**manifest, "split_hash": fingerprint(manifest)}]
