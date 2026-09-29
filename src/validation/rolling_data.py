"""Lossless source events and strictly causal UTC rolling partitions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np


DAY = 86_400_000
SCHEMA = "microlens-full-rolling/v1"


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
            ids.sort(key=lambda i: (self.timestamps[i], i))
            times = self.timestamps[ids]
            self.history_ends[ids] = np.searchsorted(times, times, side="left")

    def history(self, event, limit=None):
        ids = self.by_user[self.rows[event]["user_id"]]
        end = int(self.history_ends[event])
        start = max(0, end - limit) if limit else 0
        return self.targets[ids[start:end]].tolist()

    def select(self, start=None, end=None, *, eligible=True):
        mask = np.ones(len(self.rows), dtype=bool)
        if start is not None:
            mask &= self.timestamps >= start
        if end is not None:
            mask &= self.timestamps < end
        if eligible:
            mask &= self.history_ends > 0
        ids = np.flatnonzero(mask)
        return ids[np.lexsort((ids, self.timestamps[ids]))]

    def splits(self, days=7):
        final_day = int(self.timestamps.max()) // DAY * DAY
        result = []
        for start in range(final_day - days * DAY, final_day, DAY):
            ranges = {
                "selection": (None, start - DAY),
                "validation": (start - DAY, start),
                "refit": (None, start),
                "test": (start, start + DAY),
            }
            phases = {}
            for name, (lower, upper) in ranges.items():
                raw = len(self.select(lower, upper, eligible=False))
                count = len(self.select(lower, upper))
                phases[name] = {
                    "start_ms": lower,
                    "end_ms": upper,
                    "raw_count": raw,
                    "eligible_count": count,
                    "no_history_count": raw - count,
                }
            result.append(
                {
                    "evaluation_date": datetime.fromtimestamp(start / 1000, timezone.utc)
                    .date()
                    .isoformat(),
                    "phases": phases,
                }
            )
        return result
