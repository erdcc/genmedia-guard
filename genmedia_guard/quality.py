"""Output-quality gate and drift detection for generated images.

Generative models fail in ways that uptime and latency never show: they keep
returning HTTP 200 while the output quietly turns blank, saturated or blurry.
This module turns that into measurable signals:

* ``OutputValidator`` - per-request checks (shape, blank, saturated).
* ``sharpness``       - a cheap scalar feature of each output.
* ``psi``             - Population Stability Index between a baseline of that
                         feature and a recent window, to catch gradual drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def sharpness(img: np.ndarray) -> float:
    """Mean absolute horizontal gradient of the grayscale image."""
    gray = img.astype(np.float32).mean(axis=2)
    return float(np.abs(np.diff(gray, axis=1)).mean())


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    reasons: Tuple[str, ...]
    sharpness: Optional[float]


class OutputValidator:
    """Cheap, model-agnostic sanity checks on a generated RGB image."""

    def __init__(
        self,
        expected_shape: Tuple[int, int, int] = (64, 64, 3),
        min_contrast: float = 3.0,
        max_saturated_fraction: float = 0.98,
    ) -> None:
        self.expected_shape = expected_shape
        self.min_contrast = min_contrast
        self.max_saturated_fraction = max_saturated_fraction

    def validate(self, img: object) -> QualityReport:
        if (
            not isinstance(img, np.ndarray)
            or img.shape != self.expected_shape
            or img.dtype != np.uint8
        ):
            return QualityReport(False, ("bad_shape_or_dtype",), None)

        reasons = []
        if float(img.std()) < self.min_contrast:
            reasons.append("blank_output")
        saturated = float(((img == 0) | (img == 255)).all(axis=2).mean())
        if saturated > self.max_saturated_fraction:
            reasons.append("saturated_output")
        return QualityReport(not reasons, tuple(reasons), sharpness(img))


def psi(
    baseline: Sequence[float],
    current: Sequence[float],
    bins: int = 5,
    eps: float = 1e-4,
) -> float:
    """Population Stability Index of ``current`` against ``baseline``.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    Bin edges are baseline quantiles, so each baseline bin holds ~1/bins of
    the mass. Note PSI is biased upwards for small samples (roughly
    ``(bins - 1) * (1/n_baseline + 1/n_current)``); keep ``bins`` small and
    ``current`` large enough when using it as a gate.
    """
    base = np.asarray(baseline, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if base.size == 0 or cur.size == 0:
        return 0.0

    edges = np.unique(np.quantile(base, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        # Degenerate baseline (almost constant): any movement away is drift.
        return 0.0 if np.allclose(cur, base.mean(), atol=1e-6) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf

    expected = np.histogram(base, edges)[0] / base.size
    actual = np.histogram(cur, edges)[0] / cur.size
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
