"""The circuit-breaker engine.

``CircuitBreakerMonitor`` is the supervisor that sits *above* the agent's
execution graph. Adapters push normalized ``AgentEvent``s into :meth:`observe`;
the monitor runs every detector, and on a trip it:

  1. freezes an ``ExecutionSnapshot``,
  2. asks the (separate) ``RecoveryEngine`` for a ``SteeringPath``,
  3. returns a ``Directive`` telling the adapter what to do next —
     inject the steering text and resume, pause/escalate, or abort.

The monitor never runs the agent itself; it only observes and directs. That
separation is the whole point: the thing judging the run is not the thing
performing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .events import AgentEvent, EventType, ExecutionSnapshot
from .detectors import Detector, LoopDetector, DriftDetector, SpendDetector, NoProgressDetector
from .detectors.base import Severity
from .recovery import RecoveryEngine, SteeringPath, RecoveryAction
from .tracer import Tracer


class DirectiveKind(str, Enum):
    CONTINUE = "continue"
    INJECT = "inject"      # feed steering_text back into the agent, then resume
    PAUSE = "pause"        # halt and escalate to a human
    ABORT = "abort"        # stop the run entirely


@dataclass
class Directive:
    kind: DirectiveKind = DirectiveKind.CONTINUE
    steering_text: Optional[str] = None
    path: Optional[SteeringPath] = None


@dataclass
class MonitorConfig:
    original_goal: str
    max_recoveries: int = 3          # after this many trips, escalate instead of steer
    loop_threshold: int = 3
    drift_threshold: float = 0.45
    stall_patience: int = 6
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    burst_window: int = 6
    burst_tokens: Optional[int] = None
    jsonl_path: Optional[str] = None
    echo: bool = True


class CircuitBreakerMonitor:
    def __init__(self, config: MonitorConfig, detectors: Optional[list[Detector]] = None,
                 recovery: Optional[RecoveryEngine] = None, tracer: Optional[Tracer] = None):
        self.config = config
        self.detectors: list[Detector] = detectors or [
            LoopDetector(threshold=config.loop_threshold),
            DriftDetector(original_goal=config.original_goal, threshold=config.drift_threshold),
            NoProgressDetector(patience=config.stall_patience),
            SpendDetector(
                max_tokens=config.max_tokens,
                max_cost_usd=config.max_cost_usd,
                burst_window=config.burst_window,
                burst_tokens=config.burst_tokens,
            ),
        ]
        self.recovery = recovery or RecoveryEngine()
        self.tracer = tracer or Tracer(jsonl_path=config.jsonl_path, echo=config.echo)
        self.tracer.meta({
            "original_goal": config.original_goal,
            "recovery_backend": self.recovery.backend,
            "config": {
                "loop_threshold": config.loop_threshold,
                "drift_threshold": config.drift_threshold,
                "stall_patience": config.stall_patience,
                "max_tokens": config.max_tokens,
                "max_cost_usd": config.max_cost_usd,
                "max_recoveries": config.max_recoveries,
            },
        })
        self.history: list[AgentEvent] = []
        self.route_history: list[str] = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.current_goal: Optional[str] = None
        self.recovery_count = 0

    # ------------------------------------------------------------------
    def observe(self, event: AgentEvent) -> Directive:
        """Ingest one event; return what the agent should do next."""
        self.history.append(event)
        self.total_tokens += event.tokens_in + event.tokens_out
        self.total_cost += event.cost_usd
        if event.node:
            if not self.route_history or self.route_history[-1] != event.node:
                self.route_history.append(event.node)
        if event.goal:
            self.current_goal = event.goal
        self.tracer.event(event)

        if event.type in (EventType.COMPLETE, EventType.ABORT):
            return Directive(DirectiveKind.CONTINUE)

        for detector in self.detectors:
            trip = detector.inspect(event, self.history)
            if trip is None:
                continue
            return self._handle_trip(event, detector, trip)

        return Directive(DirectiveKind.CONTINUE)

    # ------------------------------------------------------------------
    def _handle_trip(self, event: AgentEvent, detector, trip) -> Directive:
        self.tracer.trip(event, trip)

        snapshot = ExecutionSnapshot(
            step=event.step,
            original_goal=self.config.original_goal,
            current_goal=self.current_goal,
            total_tokens=self.total_tokens,
            total_cost_usd=self.total_cost,
            route_history=self.route_history,
            recent_events=[e.to_dict() for e in self.history[-10:]],
            trip_reason=trip.reason,
            trip_detector=trip.detector,
            trip_evidence=trip.evidence,
        )

        # Critical severity or too many recoveries -> stop steering, get a human.
        if trip.severity == Severity.CRITICAL or self.recovery_count >= self.config.max_recoveries:
            path = self.recovery.recover(snapshot)
            path.action = RecoveryAction.ESCALATE if trip.severity != Severity.CRITICAL else path.action
            self.tracer.recovery(path)
            kind = DirectiveKind.ABORT if path.action == RecoveryAction.ABORT else DirectiveKind.PAUSE
            return Directive(kind, steering_text=path.instruction, path=path)

        path = self.recovery.recover(snapshot)
        self.recovery_count += 1
        self.tracer.recovery(path)

        # A successful steer clears detector counters so the same near-miss
        # pattern doesn't instantly re-trip on the next step.
        for d in self.detectors:
            d.reset()

        if path.action == RecoveryAction.ESCALATE:
            return Directive(DirectiveKind.PAUSE, steering_text=path.instruction, path=path)
        if path.action == RecoveryAction.ABORT:
            return Directive(DirectiveKind.ABORT, steering_text=path.instruction, path=path)
        return Directive(DirectiveKind.INJECT, steering_text=path.instruction, path=path)

    # ------------------------------------------------------------------
    def finish(self, status: str = "complete") -> dict:
        totals = {
            "status": status,
            "steps": self.history[-1].step if self.history else 0,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "trips": self.tracer.trips,
            "recoveries": self.tracer.recoveries,
            "route": " -> ".join(self.route_history[-12:]),
        }
        self.tracer.summary(totals)
        self.tracer.close()
        return totals
