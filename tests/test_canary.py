import pytest

from genmedia_guard.canary import Action, CanaryConfig, CanaryController, CanaryState
from genmedia_guard.slo import SLO, Sample


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


CFG = CanaryConfig(steps=(0.1, 0.5), step_duration_s=60, min_requests=50,
                   slo=SLO(latency_p95_ms=2000, max_error_rate=0.02))


@pytest.fixture
def setup():
    clock = Clock()
    ctrl = CanaryController(CFG, baseline_features=[], clock=clock)
    ctrl.start()
    return ctrl, clock


def feed(ctrl, clock, role, n, latency=1000.0, ok=True, quality_ok=True):
    for _ in range(n):
        ctrl.record(role, Sample(clock.t, latency, ok, quality_ok))


def test_weight_follows_state(setup):
    ctrl, clock = setup
    assert ctrl.weight() == 0.1


def test_holds_until_enough_evidence(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "stable", 200)
    feed(ctrl, clock, "canary", 10)
    clock.t = 120
    assert ctrl.evaluate().action == Action.HOLD


def test_advances_then_promotes_when_healthy(setup):
    ctrl, clock = setup
    for step_end in (60, 120):
        feed(ctrl, clock, "stable", 200)
        feed(ctrl, clock, "canary", 60)
        clock.t = step_end
        verdict = ctrl.evaluate()
    assert verdict.action == Action.PROMOTE
    assert ctrl.state == CanaryState.PROMOTED and ctrl.weight() == 1.0


def test_step_waits_for_full_duration(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "stable", 200)
    feed(ctrl, clock, "canary", 60)
    clock.t = 30
    v = ctrl.evaluate()
    assert v.action == Action.HOLD and ctrl.weight() == 0.1


def test_rolls_back_on_absolute_latency_slo(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "stable", 200, latency=2500)  # stable is slow too
    feed(ctrl, clock, "canary", 60, latency=2500)
    clock.t = 10
    v = ctrl.evaluate()
    assert v.action == Action.ROLLBACK
    assert any("exceeds SLO" in r for r in v.reasons)
    assert ctrl.weight() == 0.0 and ctrl.state == CanaryState.ROLLED_BACK


def test_rolls_back_on_relative_latency_even_within_slo(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "stable", 200, latency=1000)
    feed(ctrl, clock, "canary", 60, latency=1500)  # under 2000ms SLO, 1.5x stable
    clock.t = 10
    v = ctrl.evaluate()
    assert v.action == Action.ROLLBACK
    assert any("x stable" in r for r in v.reasons)


def test_rolls_back_on_quality_failures(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "stable", 200)
    feed(ctrl, clock, "canary", 50)
    feed(ctrl, clock, "canary", 5, quality_ok=False)  # ~9% > 2%
    clock.t = 10
    v = ctrl.evaluate()
    assert v.action == Action.ROLLBACK
    assert any("quality" in r for r in v.reasons)


def test_fast_fail_on_catastrophic_error_rate(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "canary", 25, ok=False)  # only 25 requests < min_requests
    clock.t = 5
    v = ctrl.evaluate()
    assert v.action == Action.ROLLBACK
    assert "catastrophic" in v.reasons[0]


def test_drift_gate_triggers_rollback():
    clock = Clock()
    baseline = [10.0 + 0.01 * i for i in range(200)]
    ctrl = CanaryController(CFG, baseline_features=baseline, clock=clock)
    ctrl.start()
    feed(ctrl, clock, "stable", 200)
    for _ in range(60):
        ctrl.record("canary", Sample(clock.t, 1000, True, True, feature=2.0))  # far from baseline
    clock.t = 10
    v = ctrl.evaluate()
    assert v.action == Action.ROLLBACK
    assert any("drift" in r for r in v.reasons)


def test_terminal_state_ignores_evaluation(setup):
    ctrl, clock = setup
    feed(ctrl, clock, "canary", 25, ok=False)
    ctrl.evaluate()
    assert ctrl.evaluate().action == Action.HOLD
