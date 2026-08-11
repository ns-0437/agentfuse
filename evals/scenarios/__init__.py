"""Scenario registry for the AgentFuse eval suite.

Positives are genuine failures the breaker must catch; negatives are healthy runs
it must leave alone. Both halves matter — recall without a low false-positive
rate is a guardrail nobody keeps switched on.
"""

from __future__ import annotations

from ..schema import Scenario
from .positives import SCENARIOS as _POSITIVES
from .negatives import SCENARIOS as _NEGATIVES

ALL_SCENARIOS: list[Scenario] = [*_POSITIVES, *_NEGATIVES]

POSITIVES = _POSITIVES
NEGATIVES = _NEGATIVES


def by_id(scenario_id: str) -> Scenario:
    for s in ALL_SCENARIOS:
        if s.id == scenario_id:
            return s
    raise KeyError(f"unknown scenario: {scenario_id}")


def families() -> list[str]:
    return sorted({s.family for s in ALL_SCENARIOS})


__all__ = ["ALL_SCENARIOS", "POSITIVES", "NEGATIVES", "by_id", "families"]
