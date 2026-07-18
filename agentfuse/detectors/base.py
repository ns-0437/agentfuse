"""Detector contract.

A detector is a small, stateful sensor. It sees the stream of ``AgentEvent``s
and, when a dangerous pattern crosses a threshold, returns a ``Trip``. The
monitor owns the policy of what to do with a trip; detectors only diagnose.
Keeping them independent means you can add a new failure-mode sensor without
touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..events import AgentEvent


class Severity(str, Enum):
    WARN = "warn"        # log it, keep going
    TRIP = "trip"        # pause + steer
    CRITICAL = "critical"  # pause + escalate to human / abort


@dataclass
class Trip:
    """A detector's diagnosis that something is wrong."""

    detector: str
    reason: str
    severity: Severity = Severity.TRIP
    evidence: dict = field(default_factory=dict)


class Detector:
    """Base class. Subclasses implement :meth:`inspect`."""

    name: str = "detector"

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        """Return a ``Trip`` if this event pushes the run over a threshold."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal counters, e.g. after a successful steering recovery."""
