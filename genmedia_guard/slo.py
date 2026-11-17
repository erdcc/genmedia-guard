"""SLO definitions and a sliding window of request samples."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SLO:
    """Service level objectives for one model API."""

    latency_p95_ms: float = 2500.0
    max_error_rate: float = 0.02


@dataclass(frozen=True)
class Sample:
    """One observed request."""

    ts: float
    latency_ms: float
    ok: bool
    quality_ok: bool = True
    feature: Optional[float] = None  # output feature used for drift detection


@dataclass(frozen=True)
class WindowStats:
    count: int
    error_rate: float
    p95_ms: float
    quality_fail_rate: float
    features: Tuple[float, ...]


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; returns 0.0 for an empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return float(ordered[rank - 1])


def burn_rate(error_rate: float, slo: SLO) -> float:
    """How fast the error budget is being spent (1.0 = exactly on budget)."""
    if slo.max_error_rate <= 0:
        return math.inf if error_rate > 0 else 0.0
    return error_rate / slo.max_error_rate


class SlidingWindow:
    """Keeps samples from the last ``window_s`` seconds."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._samples: Deque[Sample] = deque()

    def add(self, sample: Sample) -> None:
        self._samples.append(sample)

    def clear(self) -> None:
        self._samples.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()

    def stats(self, now: float) -> WindowStats:
        self._prune(now)
        samples: List[Sample] = list(self._samples)
        count = len(samples)
        if count == 0:
            return WindowStats(0, 0.0, 0.0, 0.0, ())
        ok_samples = [s for s in samples if s.ok]
        errors = count - len(ok_samples)
        quality_fails = sum(1 for s in ok_samples if not s.quality_ok)
        return WindowStats(
            count=count,
            error_rate=errors / count,
            p95_ms=percentile([s.latency_ms for s in ok_samples], 0.95),
            quality_fail_rate=(quality_fails / len(ok_samples)) if ok_samples else 0.0,
            features=tuple(s.feature for s in ok_samples if s.feature is not None),
        )
