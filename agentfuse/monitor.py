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
from .detectors import (Detector, LoopDetector, DriftDetector, SpendDetector,
                        NoProgressDetector, RateOfProgressDetector)
from .calibration import AdaptiveCalibrator
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
    # Must be >= len(strategies.STEERABLE), or the steering ladder is truncated
    # and its upper rungs can never be reached. At 3 against a 4-rung ladder the
    # measured recovery rate was 55.4%; at 5 it is 75.2%, then it plateaus.
    max_recoveries: int = 5
    loop_threshold: int = 3
    # None = let DriftDetector choose per mode. Embedding and lexical similarity
    # live on different scales (~0.62 vs ~0.20), so one constant cannot serve
    # both; hard-coding the lexical value would cripple the embedding path.
    # Set an explicit float to override. See evals/results/REPORT.md.
    drift_threshold: Optional[float] = None
    stall_patience: int = 6
    # Consecutive formally-identical state advances tolerated before a run that
    # advances every step but converges on nothing is called out. Set to None to
    # disable the rate detector entirely.
    rate_patience: Optional[int] = 8
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    burst_window: int = 6
    burst_tokens: Optional[int] = None
    jsonl_path: Optional[str] = None
    echo: bool = True
    # How many steps a steer gets to demonstrate it worked before it is judged
    # ineffective and the ladder climbs past it.
    verify_window: int = 4
    # Learn per-run baselines from demonstrably healthy stretches and widen
    # thresholds to match this workload. Never tightens them.
    adaptive: bool = True


class CircuitBreakerMonitor:
    def __init__(self, config: MonitorConfig, detectors: Optional[list[Detector]] = None,
                 recovery: Optional[RecoveryEngine] = None, tracer: Optional[Tracer] = None):
        self.config = config
        self.calibrator = AdaptiveCalibrator(enabled=config.adaptive)
        self.detectors: list[Detector] = detectors or [
            LoopDetector(threshold=config.loop_threshold, calibrator=self.calibrator),
            DriftDetector(original_goal=config.original_goal, threshold=config.drift_threshold),
            NoProgressDetector(patience=config.stall_patience, calibrator=self.calibrator),
            SpendDetector(
                max_tokens=config.max_tokens,
                max_cost_usd=config.max_cost_usd,
                burst_window=config.burst_window,
                burst_tokens=config.burst_tokens,
            ),
        ]
        # Last in the default list deliberately: it and SpendDetector can both
        # have a claim on a run that advances every step while burning tokens,
        # and on a tie the budget breach is the more actionable diagnosis.
        # Only added when the caller took the defaults — appending a sensor to a
        # hand-picked detector list would override an explicit choice, which is
        # different from wiring up a detector the caller did ask for.
        if detectors is None and config.rate_patience is not None:
            self.detectors.append(RateOfProgressDetector(patience=config.rate_patience))
        # Calibration is owned by the monitor and injected into whatever
        # detectors it ends up with — including a caller-supplied list. Wiring it
        # only into the default-constructed set meant any custom detector list
        # silently ran uncalibrated, which is exactly what the eval does.
        for _d in self.detectors:
            if hasattr(_d, "calibrator") and getattr(_d, "calibrator", None) is None:
                _d.calibrator = self.calibrator

        self.recovery = recovery or RecoveryEngine()
        self.tracer = tracer or Tracer(jsonl_path=config.jsonl_path, echo=config.echo)
        self.tracer.meta({
            "original_goal": config.original_goal,
            "recovery_backend": self.recovery.backend,
            "config": {
                "loop_threshold": config.loop_threshold,
                "drift_threshold": config.drift_threshold,
                "stall_patience": config.stall_patience,
                "rate_patience": config.rate_patience,
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
        # The steer awaiting a verdict: (path, step it was injected at).
        self._pending_steer = None
        self.steers_that_worked = 0
        self.steers_that_failed = 0

    # ------------------------------------------------------------------
    def _verify_pending(self, event: AgentEvent, new_trip: bool = False) -> None:
        """Decide whether the last steer worked, once there is evidence either way.

        A steer counts as having worked if the agent makes genuine state progress
        within ``verify_window`` steps of receiving it. It counts as failed if the
        breaker trips again first, or if the window closes with nothing advanced.

        Until this existed the memory only knew what had been *tried*, which is
        not enough to stop the engine repeating a correction that already failed.
        """
        if self._pending_steer is None:
            return
        path, injected_at = self._pending_steer

        if event.state is not None and not new_trip:
            self.recovery.verify(path, worked=True)
            self._pending_steer = None
            self.steers_that_worked += 1
            return

        if new_trip or (event.step - injected_at) > self.config.verify_window:
            self.recovery.verify(path, worked=False)
            self._pending_steer = None
            self.steers_that_failed += 1

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
        self.calibrator.observe(event)

        if event.type in (EventType.COMPLETE, EventType.ABORT):
            self._verify_pending(event)
            return Directive(DirectiveKind.CONTINUE)

        self._verify_pending(event)

        for detector in self.detectors:
            trip = detector.inspect(event, self.history)
            if trip is None:
                continue
            return self._handle_trip(event, detector, trip)

        return Directive(DirectiveKind.CONTINUE)

    # ------------------------------------------------------------------
    def _handle_trip(self, event: AgentEvent, detector, trip) -> Directive:
        # Tripping again is the clearest possible evidence the last steer failed.
        self._verify_pending(event, new_trip=True)
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

        self._pending_steer = (path, event.step)
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
            "steers_verified_working": self.steers_that_worked,
            "steers_verified_failed": self.steers_that_failed,
            "route": " -> ".join(self.route_history[-12:]),
        }
        self.tracer.summary(totals)
        self.tracer.close()
        return totals
