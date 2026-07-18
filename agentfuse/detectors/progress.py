"""No-progress (stall) detector.

Distinct from the loop detector: an agent can churn through *different* tool
calls and reasoning turns yet never actually move the working state — the
hallmark of a logical trap where it reasons busily from a false premise. If the
state hash is unchanged across ``patience`` steps despite ongoing activity, we
trip.
"""

from __future__ import annotations

from typing import Optional

from ..events import AgentEvent, EventType
from .base import Detector, Trip, Severity


class NoProgressDetector(Detector):
    name = "progress"

    def __init__(self, patience: int = 6):
        self.patience = patience
        self._last_state_hash: Optional[str] = None
        self._stall_steps = 0
        self._activity_since_change = 0

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        if event.type in (EventType.TOOL_CALL, EventType.LLM_CALL):
            self._activity_since_change += 1

        h = event.state_hash
        if h is not None:
            if h == self._last_state_hash:
                self._stall_steps += 1
            else:
                self._last_state_hash = h
                self._stall_steps = 0
                self._activity_since_change = 0

        if self._stall_steps >= self.patience and self._activity_since_change >= self.patience:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"No state progress for {self._stall_steps} steps despite "
                    f"{self._activity_since_change} actions — likely a logical trap."
                ),
                evidence={
                    "stall_steps": self._stall_steps,
                    "actions_without_progress": self._activity_since_change,
                    "state_hash": self._last_state_hash,
                },
            )
        return None

    def reset(self) -> None:
        self._stall_steps = 0
        self._activity_since_change = 0
