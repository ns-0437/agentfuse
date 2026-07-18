"""Token / cost budget detector.

Long-running autonomy fails economically before it fails logically: a stuck
agent burns tokens indefinitely. This tracks cumulative spend and, separately,
the *burn rate* over a recent window. A hard ceiling trips CRITICAL (escalate);
a runaway burn rate trips a normal steering event so a recovery can cut the
waste before the ceiling is hit.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..events import AgentEvent
from .base import Detector, Trip, Severity


class SpendDetector(Detector):
    name = "spend"

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        burst_window: int = 6,
        burst_tokens: Optional[int] = None,
    ):
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.burst_window = burst_window
        self.burst_tokens = burst_tokens
        self._total_tokens = 0
        self._total_cost = 0.0
        self._recent: deque[int] = deque(maxlen=burst_window)

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        step_tokens = event.tokens_in + event.tokens_out
        self._total_tokens += step_tokens
        self._total_cost += event.cost_usd
        if step_tokens:
            self._recent.append(step_tokens)

        # Hard ceilings -> escalate, don't just steer.
        if self.max_tokens is not None and self._total_tokens >= self.max_tokens:
            return Trip(
                detector=self.name,
                severity=Severity.CRITICAL,
                reason=f"Token budget exhausted: {self._total_tokens} >= {self.max_tokens}.",
                evidence={"total_tokens": self._total_tokens, "limit": self.max_tokens},
            )
        if self.max_cost_usd is not None and self._total_cost >= self.max_cost_usd:
            return Trip(
                detector=self.name,
                severity=Severity.CRITICAL,
                reason=f"Cost budget exhausted: ${self._total_cost:.4f} >= ${self.max_cost_usd:.4f}.",
                evidence={"total_cost_usd": round(self._total_cost, 4), "limit": self.max_cost_usd},
            )

        # Burn-rate guard -> steer before the ceiling.
        if self.burst_tokens is not None and len(self._recent) == self.burst_window:
            burned = sum(self._recent)
            if burned >= self.burst_tokens:
                return Trip(
                    detector=self.name,
                    severity=Severity.TRIP,
                    reason=(
                        f"Token burn rate spiking: {burned} tokens over the last "
                        f"{self.burst_window} steps (>= {self.burst_tokens})."
                    ),
                    evidence={"window_tokens": burned, "window": self.burst_window},
                )
        return None

    def reset(self) -> None:
        self._recent.clear()

    @property
    def totals(self) -> dict:
        return {"tokens": self._total_tokens, "cost_usd": round(self._total_cost, 4)}
