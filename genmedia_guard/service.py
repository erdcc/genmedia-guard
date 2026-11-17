"""Request path: routing, model call, quality gate, metrics, canary feed.

``GuardService`` is transport-agnostic (the ASGI app and the simulator both
use it), and takes an injectable clock so it can run on virtual time.
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Dict, Optional

import numpy as np

from .canary import Action, CanaryConfig, CanaryController, CanaryState, Verdict
from .fake_model import SCENARIOS, STABLE, FakeDiffusionModel, ModelError
from .metrics import Counter, Gauge, Histogram, Registry
from .quality import OutputValidator, sharpness
from .slo import Sample


class GuardService:
    def __init__(
        self,
        config: Optional[CanaryConfig] = None,
        clock: Callable[[], float] = time.monotonic,
        seed: int = 0,
        baseline_size: int = 500,
    ) -> None:
        self.config = config or CanaryConfig()
        self._clock = clock
        self._route_rng = np.random.default_rng(seed + 1)
        self.validator = OutputValidator()
        self.stable_model = FakeDiffusionModel(STABLE, seed=seed + 2)
        self.canary_model: Optional[FakeDiffusionModel] = None
        self._seed = seed

        self.registry = Registry()
        self.m_requests = self.registry.register(Counter(
            "genmedia_requests_total", "Requests by model version and outcome.",
            ("version", "status")))
        self.m_latency = self.registry.register(Histogram(
            "genmedia_request_duration_seconds", "Model latency in seconds.", ("version",)))
        self.m_quality = self.registry.register(Counter(
            "genmedia_quality_failures_total", "Outputs rejected by the quality gate.",
            ("version", "reason")))
        self.m_fallback = self.registry.register(Counter(
            "genmedia_fallback_total", "Requests re-served by the stable version.", ()))
        self.m_weight = self.registry.register(Gauge(
            "genmedia_canary_weight", "Fraction of traffic sent to the canary.", ()))
        self.m_state = self.registry.register(Gauge(
            "genmedia_canary_state", "1 for the current canary state.", ("state",)))
        self.m_rollbacks = self.registry.register(Counter(
            "genmedia_canary_rollbacks_total", "Automated canary rollbacks.", ()))
        self.m_promotions = self.registry.register(Counter(
            "genmedia_canary_promotions_total", "Canaries promoted to stable.", ()))

        baseline = [
            sharpness(self.stable_model._render(1_000_000 + i)) for i in range(baseline_size)
        ]
        self.controller = CanaryController(self.config, baseline, clock)
        self._publish_state()

    # -- canary management ---------------------------------------------------
    def start_canary(self, scenario: str) -> Dict[str, object]:
        if scenario not in SCENARIOS:
            raise KeyError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        self.canary_model = FakeDiffusionModel(SCENARIOS[scenario], seed=self._seed + 3)
        self.controller.start()
        self._publish_state()
        return self.controller.status()

    def evaluate(self) -> Verdict:
        verdict = self.controller.evaluate()
        if verdict.action == Action.ROLLBACK:
            self.m_rollbacks.inc()
        elif verdict.action == Action.PROMOTE:
            self.m_promotions.inc()
            self.stable_model = self.canary_model or self.stable_model
        self._publish_state()
        return verdict

    def _publish_state(self) -> None:
        self.m_weight.set(self.controller.weight())
        for state in CanaryState:
            self.m_state.set(1.0 if self.controller.state == state else 0.0, state=state.value)

    # -- request path --------------------------------------------------------
    def pick_role(self) -> str:
        w = self.controller.weight()
        if self.canary_model is not None and w > 0 and self._route_rng.random() < w:
            return "canary"
        return "stable"

    def _serve(self, role: str, seed: int):
        model = self.canary_model if role == "canary" else self.stable_model
        assert model is not None
        try:
            img, latency_ms = model.generate(seed)
        except ModelError as exc:
            return model.version.name, None, exc.latency_ms, None, str(exc)
        report = self.validator.validate(img)
        return model.version.name, img, latency_ms, report, None

    def _observe(self, role: str, version: str, latency_ms: float, ok: bool, report) -> None:
        quality_ok = report.ok if report else True
        feature = report.sharpness if report else None
        self.controller.record(
            role,
            Sample(self._clock(), latency_ms, ok, quality_ok, feature),
        )
        self.m_latency.observe(latency_ms / 1000.0, version=version)
        status = "error" if not ok else ("ok" if quality_ok else "quality_fail")
        self.m_requests.inc(version=version, status=status)
        if report and not report.ok:
            for reason in report.reasons:
                self.m_quality.inc(version=version, reason=reason)

    def generate(self, prompt: str, seed: int) -> Dict[str, object]:
        """Serve one request. Returns a response dict (never raises for model faults)."""
        role = self.pick_role()
        version, img, latency_ms, report, error = self._serve(role, seed)
        ok = error is None
        self._observe(role, version, latency_ms, ok, report)

        failed = (not ok) or (report is not None and not report.ok)
        if failed and role == "canary":
            # Request-level mitigation: the user still gets an answer from stable.
            self.m_fallback.inc()
            version2, img2, lat2, report2, error2 = self._serve("stable", seed)
            self._observe("stable", version2, lat2, error2 is None, report2)
            if error2 is None and report2 is not None and report2.ok:
                return self._response(version2, img2, lat2 + latency_ms, report2, fallback=True)
            return {"status": 502, "error": error2 or "quality gate rejected output",
                    "version": version2}

        if not ok:
            return {"status": 502, "error": error, "version": version}
        return self._response(version, img, latency_ms, report, fallback=False)

    @staticmethod
    def _response(version, img, latency_ms, report, fallback: bool) -> Dict[str, object]:
        return {
            "status": 200,
            "version": version,
            "latency_ms": round(latency_ms, 1),
            "fallback": fallback,
            "image_sha1": hashlib.sha1(img.tobytes()).hexdigest()[:16],
            "quality": {"ok": report.ok, "reasons": list(report.reasons),
                        "sharpness": round(report.sharpness or 0.0, 3)},
        }
