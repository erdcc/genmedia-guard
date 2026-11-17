"""Deterministic canary-rollout simulator on virtual time.

    python -m genmedia_guard.simulate --scenario quality-drift
    python -m genmedia_guard.simulate --all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .canary import Action, CanaryConfig, CanaryState
from .fake_model import SCENARIOS
from .service import GuardService


class VirtualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@dataclass(frozen=True)
class SimulationResult:
    scenario: str
    outcome: str  # "promoted" | "rolled_back" | "timeout"
    elapsed_s: float
    final_weight: float
    canary_requests: int
    reasons: Tuple[str, ...]
    timeline: Tuple[Tuple[float, str, float], ...]  # (t, action, weight)


def run_scenario(
    name: str,
    seed: int = 7,
    rps: int = 50,
    eval_every_s: int = 5,
    max_s: int = 1800,
    config: Optional[CanaryConfig] = None,
) -> SimulationResult:
    clock = VirtualClock()
    svc = GuardService(config=config, clock=clock, seed=seed)
    svc.start_canary(name)

    timeline: List[Tuple[float, str, float]] = []
    next_seed = 0
    reasons: Tuple[str, ...] = ()
    outcome = "timeout"

    for second in range(1, max_s + 1):
        for i in range(rps):
            clock.t = second - 1 + i / rps
            svc.generate("a cat", next_seed)
            next_seed += 1
        clock.t = float(second)

        if second % eval_every_s == 0:
            verdict = svc.evaluate()
            if verdict.action != Action.HOLD:
                timeline.append((clock.t, verdict.action.value, svc.controller.weight()))
            if verdict.action == Action.ROLLBACK:
                outcome, reasons = "rolled_back", verdict.reasons
                break
            if verdict.action == Action.PROMOTE:
                outcome, reasons = "promoted", verdict.reasons
                break

    return SimulationResult(
        scenario=name,
        outcome=outcome,
        elapsed_s=clock.t,
        final_weight=svc.controller.weight(),
        canary_requests=int(sum(
            v for (ver, _), v in svc.m_requests._values.items() if ver != "v1-stable"
        )),
        reasons=reasons,
        timeline=tuple(timeline),
    )


def _print(result: SimulationResult) -> None:
    print(f"\n== {result.scenario}: {result.outcome.upper()} after {result.elapsed_s:.0f}s "
          f"(final canary weight {result.final_weight:.0%}, "
          f"{result.canary_requests} canary requests)")
    for t, action, weight in result.timeline:
        print(f"   t={t:>5.0f}s  {action:<9} -> canary weight {weight:.0%}")
    for reason in result.reasons:
        print(f"   - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--all", action="store_true", help="run every scenario")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rps", type=int, default=50)
    args = parser.parse_args()

    names = sorted(SCENARIOS) if args.all or not args.scenario else [args.scenario]
    for name in names:
        _print(run_scenario(name, seed=args.seed, rps=args.rps))


if __name__ == "__main__":
    main()
