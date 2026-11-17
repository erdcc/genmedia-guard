"""A deterministic stand-in for a diffusion model API.

It lets the reliability tooling be developed, tested and demoed without a GPU.
Each ``ModelVersion`` can be given the kinds of defects a real model rollout
introduces: higher latency, errors, blurrier output, or silent collapse into
blank images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class ModelError(Exception):
    """Raised when the model fails to produce an output."""

    def __init__(self, message: str, latency_ms: float) -> None:
        super().__init__(message)
        self.latency_ms = latency_ms


@dataclass(frozen=True)
class ModelVersion:
    name: str
    base_latency_ms: float = 1200.0
    jitter_ms: float = 250.0
    error_rate: float = 0.005
    blur: float = 0.0  # 0 = sharp, 1 = very blurry (gradual quality drift)
    collapse_rate: float = 0.0  # fraction of requests that return a blank image


class FakeDiffusionModel:
    def __init__(
        self,
        version: ModelVersion,
        shape: Tuple[int, int, int] = (64, 64, 3),
        seed: int = 0,
    ) -> None:
        self.version = version
        self.shape = shape
        self._rng = np.random.default_rng(seed)

    def generate(self, seed: int) -> Tuple[np.ndarray, float]:
        """Return ``(image, latency_ms)`` or raise ``ModelError``."""
        v = self.version
        latency_ms = max(50.0, v.base_latency_ms + self._rng.normal(0.0, v.jitter_ms))

        if self._rng.random() < v.error_rate:
            raise ModelError(f"{v.name}: inference failed", latency_ms * 0.2)

        if v.collapse_rate and self._rng.random() < v.collapse_rate:
            blank = np.full(self.shape, 127, dtype=np.uint8)
            return blank, latency_ms

        return self._render(seed), latency_ms

    def _render(self, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        h, w, _ = self.shape
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros(self.shape, dtype=np.float32)
        for ch in range(3):
            fx, fy, phase = r.uniform(0.5, 3.0), r.uniform(0.5, 3.0), r.uniform(0, 6.28)
            img[..., ch] = 128 + 80 * np.sin(fx * xx / w * 6.28 + phase) * np.cos(fy * yy / h * 6.28)
        for _ in range(4):  # hard edges, so sharpness is meaningful
            size = int(r.integers(8, 16))
            x0, y0 = int(r.integers(0, w - size)), int(r.integers(0, h - size))
            img[y0 : y0 + size, x0 : x0 + size] = r.integers(0, 255, 3)
        img += r.normal(0.0, 4.0, self.shape)

        for _ in range(int(round(self.version.blur * 6))):
            img = (
                img
                + np.roll(img, 1, 0)
                + np.roll(img, -1, 0)
                + np.roll(img, 1, 1)
                + np.roll(img, -1, 1)
            ) / 5.0
        return np.clip(img, 0, 255).astype(np.uint8)


STABLE = ModelVersion("v1-stable")

# Candidate versions used by the simulator and the demo endpoint.
SCENARIOS = {
    "healthy": ModelVersion("v2-healthy"),
    "slightly-slower": ModelVersion("v2-slightly-slower", base_latency_ms=1320.0),
    "latency-regression": ModelVersion("v2-latency-regression", base_latency_ms=2100.0),
    "error-spike": ModelVersion("v2-error-spike", error_rate=0.08),
    "quality-drift": ModelVersion("v2-quality-drift", blur=0.7),
    "silent-collapse": ModelVersion("v2-silent-collapse", collapse_rate=0.06),
}
