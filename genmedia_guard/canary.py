"""SLO- and quality-gated canary controller.

The controller owns the traffic split between a stable model version and a
candidate. It never talks to a model itself: callers feed it request samples
(``record``) and ask for a verdict (``evaluate``). That keeps it deterministic
and testable with a fake clock.

At every step the candidate must pass all gates over a full observation
window, otherwise traffic is rolled back to the stable version:

1. absolute SLOs (p95 latency, error rate),
2. relative gates against the stable version running at the same time,
3. output-quality failure rate (blank / saturated images),
4. output drift (PSI of a sharpness feature vs. a stable baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .quality import psi
from .slo import SLO, Sample, SlidingWindow


class CanaryState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class Action(str, Enum):
    HOLD = "hold"
    ADVANCE = "advance"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class Verdict:
    action: Action
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CanaryConfig:
    steps: Tuple[float, ...] = (0.05, 0.25, 0.5)  # canary traffic fractions
    step_duration_s: float = 60.0
    min_requests: int = 100  # canary requests needed before gates are trusted
    slo: SLO = field(default_factory=SLO)
    max_latency_ratio: float = 1.25  # canary p95 / stable p95
    max_error_delta: float = 0.02  # canary error rate - stable error rate
    max_quality_fail_rate: float = 0.02
    max_psi: float = 0.25
    psi_bins: int = 5
    fast_fail_requests: int = 20
    fast_fail_error_rate: float = 0.25  # catastrophic failure: roll back early


class CanaryController:
    def __init__(
        self,
        config: CanaryConfig,
        baseline_features: Sequence[float],
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.baseline_features = tuple(baseline_features)
        self._clock = clock
        window = config.step_duration_s * 2
        self._stable = SlidingWindow(window)
        self._canary = SlidingWindow(window)
        self.state = CanaryState.IDLE
        self.step = -1
        self._step_started = 0.0
        self.reasons: Tuple[str, ...] = ()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.state = CanaryState.RUNNING
        self.reasons = ()
        self._stable.clear()
        self._begin_step(0)

    def _begin_step(self, index: int) -> None:
        self.step = index
        self._step_started = self._clock()
        self._canary.clear()  # judge each step only on its own traffic

    def weight(self) -> float:
        """Fraction of traffic that should go to the candidate right now."""
        if self.state == CanaryState.RUNNING:
            return self.config.steps[self.step]
        if self.state == CanaryState.PROMOTED:
            return 1.0
        return 0.0

    # -- data in -------------------------------------------------------------
    def record(self, role: str, sample: Sample) -> None:
        (self._canary if role == "canary" else self._stable).add(sample)

    # -- decisions -----------------------------------------------------------
    def evaluate(self) -> Verdict:
        if self.state != CanaryState.RUNNING:
            return Verdict(Action.HOLD, (f"state={self.state.value}",))

        cfg = self.config
        now = self._clock()
        canary = self._canary.stats(now)
        stable = self._stable.stats(now)

        if (
            canary.count >= cfg.fast_fail_requests
            and canary.error_rate >= cfg.fast_fail_error_rate
        ):
            return self._rollback(
                [f"canary error rate {canary.error_rate:.0%} is catastrophic "
                 f"(>= {cfg.fast_fail_error_rate:.0%} over {canary.count} requests)"]
            )

        if canary.count < cfg.min_requests:
            return Verdict(
                Action.HOLD,
                (f"waiting for evidence: {canary.count}/{cfg.min_requests} canary requests",),
            )

        failures: List[str] = []
        if canary.p95_ms > cfg.slo.latency_p95_ms:
            failures.append(
                f"canary p95 {canary.p95_ms:.0f}ms exceeds SLO {cfg.slo.latency_p95_ms:.0f}ms"
            )
        if canary.error_rate > cfg.slo.max_error_rate:
            failures.append(
                f"canary error rate {canary.error_rate:.1%} exceeds SLO {cfg.slo.max_error_rate:.1%}"
            )

        if stable.count >= cfg.fast_fail_requests and stable.p95_ms > 0:
            ratio = canary.p95_ms / stable.p95_ms
            if ratio > cfg.max_latency_ratio:
                failures.append(
                    f"canary p95 is {ratio:.2f}x stable (limit {cfg.max_latency_ratio:.2f}x)"
                )
            delta = canary.error_rate - stable.error_rate
            if delta > cfg.max_error_delta:
                failures.append(
                    f"canary error rate is {delta:+.1%} vs stable (limit {cfg.max_error_delta:+.1%})"
                )

        if canary.quality_fail_rate > cfg.max_quality_fail_rate:
            failures.append(
                f"{canary.quality_fail_rate:.1%} of canary outputs fail quality checks "
                f"(limit {cfg.max_quality_fail_rate:.1%})"
            )

        if self.baseline_features and len(canary.features) >= cfg.min_requests:
            drift = psi(self.baseline_features, canary.features, bins=cfg.psi_bins)
            if drift > cfg.max_psi:
                failures.append(f"output drift PSI {drift:.2f} exceeds limit {cfg.max_psi:.2f}")

        if failures:
            return self._rollback(failures)

        if now - self._step_started >= cfg.step_duration_s:
            if self.step + 1 < len(cfg.steps):
                self._begin_step(self.step + 1)
                return Verdict(
                    Action.ADVANCE, (f"all gates passed; canary weight now {self.weight():.0%}",)
                )
            self.state = CanaryState.PROMOTED
            return Verdict(Action.PROMOTE, ("all gates passed at every step",))

        return Verdict(Action.HOLD, ("gates passing; step in progress",))

    def _rollback(self, reasons: List[str]) -> Verdict:
        self.state = CanaryState.ROLLED_BACK
        self.reasons = tuple(reasons)
        return Verdict(Action.ROLLBACK, self.reasons)

    # -- introspection -------------------------------------------------------
    def status(self) -> Dict[str, object]:
        now = self._clock()
        c = self._canary.stats(now)
        s = self._stable.stats(now)
        return {
            "state": self.state.value,
            "weight": self.weight(),
            "step": self.step,
            "reasons": list(self.reasons),
            "canary": {"requests": c.count, "p95_ms": round(c.p95_ms, 1),
                       "error_rate": round(c.error_rate, 4),
                       "quality_fail_rate": round(c.quality_fail_rate, 4)},
            "stable": {"requests": s.count, "p95_ms": round(s.p95_ms, 1),
                       "error_rate": round(s.error_rate, 4)},
        }
