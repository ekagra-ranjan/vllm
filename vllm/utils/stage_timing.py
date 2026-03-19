from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class StageTimingStat:
    total_secs: float = 0.0
    count: int = 0
    max_secs: float = 0.0

    def add(self, elapsed_secs: float, count: int = 1) -> None:
        self.total_secs += elapsed_secs
        self.count += count
        self.max_secs = max(self.max_secs, elapsed_secs)

    def to_dict(self) -> dict[str, float | int]:
        avg_secs = self.total_secs / self.count if self.count else 0.0
        return {
            "total_secs": self.total_secs,
            "count": self.count,
            "avg_secs": avg_secs,
            "max_secs": self.max_secs,
        }


class StageTimingRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, StageTimingStat] = defaultdict(StageTimingStat)

    @contextmanager
    def record(self, stage: str, *, count: int = 1):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.add(stage, elapsed, count=count)

    def add(self, stage: str, elapsed_secs: float, *, count: int = 1) -> None:
        with self._lock:
            self._stats[stage].add(elapsed_secs, count=count)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {stage: stat.to_dict() for stage, stat in self._stats.items()}

    def snapshot_and_reset(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            snapshot = {stage: stat.to_dict() for stage, stat in self._stats.items()}
            self._stats = defaultdict(StageTimingStat)
            return snapshot

    def reset(self) -> None:
        with self._lock:
            self._stats = defaultdict(StageTimingStat)
