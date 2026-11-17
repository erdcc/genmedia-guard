"""Minimal Prometheus text-format metrics (no external dependency).

Only the pieces this project needs: Counter, Gauge, Histogram and a Registry
that renders the Prometheus exposition format (text/plain; version=0.0.4).
"""

from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0,
)


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names: Sequence[str], values: Sequence[str], extra: Iterable[Tuple[str, str]] = ()) -> str:
    pairs = list(zip(names, values)) + list(extra)
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in pairs) + "}"


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: Dict[str, object]) -> Tuple[str, ...]:
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"{self.name}: expected labels {sorted(self.labelnames)}, got {sorted(labels)}"
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def _header(self) -> List[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]

    def render(self) -> List[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: Dict[Tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        if amount < 0:
            raise ValueError("counters can only increase")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: object) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> List[str]:
        lines = self._header()
        with self._lock:
            for key, val in sorted(self._values.items()):
                lines.append(f"{self.name}{_labels(self.labelnames, key)} {val}")
        return lines


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: Dict[Tuple[str, ...], float] = {}

    def set(self, value: float, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def value(self, **labels: object) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> List[str]:
        lines = self._header()
        with self._lock:
            for key, val in sorted(self._values.items()):
                lines.append(f"{self.name}{_labels(self.labelnames, key)} {val}")
        return lines


class Histogram(_Metric):
    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, labelnames)
        self.buckets = tuple(sorted(buckets))
        # key -> (per-bucket counts, sum, count)
        self._data: Dict[Tuple[str, ...], List[float]] = {}

    def observe(self, value: float, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            row = self._data.setdefault(key, [0.0] * len(self.buckets) + [0.0, 0.0])
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    row[i] += 1
            row[-2] += value
            row[-1] += 1

    def count(self, **labels: object) -> float:
        row = self._data.get(self._key(labels))
        return row[-1] if row else 0.0

    def render(self) -> List[str]:
        lines = self._header()
        with self._lock:
            for key, row in sorted(self._data.items()):
                for i, upper in enumerate(self.buckets):
                    lab = _labels(self.labelnames, key, [("le", repr(float(upper)))])
                    lines.append(f"{self.name}_bucket{lab} {row[i]}")
                lab_inf = _labels(self.labelnames, key, [("le", "+Inf")])
                lines.append(f"{self.name}_bucket{lab_inf} {row[-1]}")
                base = _labels(self.labelnames, key)
                lines.append(f"{self.name}_sum{base} {row[-2]}")
                lines.append(f"{self.name}_count{base} {row[-1]}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._metrics: List[_Metric] = []

    def register(self, metric):
        self._metrics.append(metric)
        return metric

    def render(self) -> str:
        lines: List[str] = []
        for metric in self._metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"
