import pytest

from genmedia_guard.canary import CanaryConfig
from genmedia_guard.simulate import run_scenario

FAST = CanaryConfig(step_duration_s=30, min_requests=100)


@pytest.mark.parametrize("scenario", ["healthy", "slightly-slower"])
def test_good_candidates_are_promoted(scenario):
    result = run_scenario(scenario, rps=60, config=FAST)
    assert result.outcome == "promoted"
    assert result.final_weight == 1.0


@pytest.mark.parametrize(
    "scenario, expected_reason",
    [
        ("latency-regression", "stable"),
        ("error-spike", "error rate"),
        ("quality-drift", "drift"),
        ("silent-collapse", "quality checks"),
    ],
)
def test_bad_candidates_are_rolled_back_at_the_first_step(scenario, expected_reason):
    result = run_scenario(scenario, rps=60, config=FAST)
    assert result.outcome == "rolled_back"
    assert result.final_weight == 0.0
    assert any(expected_reason in r for r in result.reasons)
    # Blast radius: rolled back while still on the smallest traffic slice.
    assert all(weight <= FAST.steps[0] or action == "rollback"
               for _, action, weight in result.timeline)
