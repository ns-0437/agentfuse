"""Replay engine — drives a scenario through a real CircuitBreakerMonitor.

This is deliberately *replay*, not live inference: the trajectories are scripted,
so the suite is deterministic, free, and runnable in CI. We instantiate the real
monitor and the real detectors, so what we measure is the production code path,
not a simulation of it.

Semantics that matter for honest scoring:

  * **Stop on first trip.** In production the breaker interrupts the run, so the
    scenario halts there. That makes "tokens saved" exact rather than notional.
  * **Recovery is mocked.** We are measuring *detection* quality here, not
    steering quality, and the deterministic mock keeps the suite hermetic.
    Supervision cost is still charged via the CostModel.
  * **No console noise.** The tracer records to memory instead of printing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentfuse import (  # noqa: E402
    AgentEvent, EventType, CircuitBreakerMonitor, MonitorConfig, Tracer,
)
from agentfuse.detectors import (  # noqa: E402
    Detector, LoopDetector, DriftDetector, NoProgressDetector, SpendDetector,
)
from agentfuse.recovery import RecoveryEngine  # noqa: E402

from .schema import Scenario, ScenarioResult, CostModel, DEFAULT_COST  # noqa: E402


# Shared baseline configuration. Scenarios may override individual keys, but the
# defaults are held constant across the suite so results aren't tuned per-case.
DEFAULT_CONFIG = {
    "loop_threshold": 3,
    "drift_threshold": 0.45,
    "stall_patience": 6,
    "max_tokens": 100_000,
    "max_cost_usd": None,
    "burst_window": 6,
    "burst_tokens": None,
    "max_recoveries": 3,
}


class RecordingTracer(Tracer):
    """Captures trips/recoveries in memory and prints nothing."""

    def __init__(self) -> None:
        super().__init__(jsonl_path=None, echo=False)
        self.trip_log: list[dict] = []
        self.recovery_log: list[dict] = []

    def trip(self, event: AgentEvent, trip) -> None:  # type: ignore[override]
        self.trips += 1
        self.trip_log.append({
            "step": event.step,
            "detector": trip.detector,
            "severity": trip.severity.value,
            "reason": trip.reason,
        })

    def recovery(self, path) -> None:  # type: ignore[override]
        self.recoveries += 1
        self.recovery_log.append({"action": path.action.value, "backend": path.backend})

    def summary(self, totals: dict) -> None:  # type: ignore[override]
        pass


def build_detectors(scenario: Scenario, cfg: dict,
                    disabled: Optional[set[str]] = None,
                    extra: Optional[list[Detector]] = None) -> list[Detector]:
    """Construct the detector set, optionally ablating some by name."""
    disabled = disabled or set()
    built: list[Detector] = []
    if "loop" not in disabled:
        built.append(LoopDetector(threshold=cfg["loop_threshold"]))
    if "drift" not in disabled:
        built.append(DriftDetector(original_goal=scenario.goal,
                                   threshold=cfg["drift_threshold"]))
    if "progress" not in disabled:
        built.append(NoProgressDetector(patience=cfg["stall_patience"]))
    if "spend" not in disabled:
        built.append(SpendDetector(max_tokens=cfg["max_tokens"],
                                   max_cost_usd=cfg["max_cost_usd"],
                                   burst_window=cfg["burst_window"],
                                   burst_tokens=cfg["burst_tokens"]))
    if extra:
        built.extend(extra)
    return built


def _events_for_step(step, step_no: int) -> list[AgentEvent]:
    """Expand one scripted step into the events an adapter would emit."""
    events: list[AgentEvent] = []
    if step.kind == "tool":
        events.append(AgentEvent(
            type=EventType.TOOL_CALL, step=step_no, node=step.node,
            tool_name=step.tool_name, tool_args=step.tool_args,
            tokens_in=step.tokens_in, tokens_out=step.tokens_out, goal=step.goal,
        ))
        events.append(AgentEvent(
            type=EventType.TOOL_RESULT, step=step_no, node=step.node,
            tool_name=step.tool_name, text=step.result,
        ))
        if step.progress:
            events.append(AgentEvent(
                type=EventType.STATE_UPDATE, step=step_no, node=step.node,
                state={"advanced_at": step_no, "last": str(step.result)[:80]},
            ))
    else:  # think
        events.append(AgentEvent(
            type=EventType.LLM_CALL, step=step_no, node=step.node,
            text=step.text, goal=step.goal,
            tokens_in=step.tokens_in, tokens_out=step.tokens_out,
        ))
        if step.progress:
            events.append(AgentEvent(
                type=EventType.STATE_UPDATE, step=step_no, node=step.node,
                state={"advanced_at": step_no, "last": str(step.text)[:80]},
            ))
    return events


def run_scenario(scenario: Scenario,
                 disabled: Optional[set[str]] = None,
                 extra_detectors: Optional[list[Detector]] = None,
                 cost: CostModel = DEFAULT_COST,
                 stop_on_first_trip: bool = True) -> ScenarioResult:
    """Replay one scenario and score what the breaker did."""
    cfg = {**DEFAULT_CONFIG, **scenario.config}
    tracer = RecordingTracer()
    monitor = CircuitBreakerMonitor(
        config=MonitorConfig(original_goal=scenario.goal, echo=False,
                             max_recoveries=cfg["max_recoveries"],
                             loop_threshold=cfg["loop_threshold"],
                             drift_threshold=cfg["drift_threshold"],
                             stall_patience=cfg["stall_patience"],
                             max_tokens=cfg["max_tokens"],
                             max_cost_usd=cfg["max_cost_usd"],
                             burst_window=cfg["burst_window"],
                             burst_tokens=cfg["burst_tokens"]),
        detectors=build_detectors(scenario, cfg, disabled, extra_detectors),
        recovery=RecoveryEngine(backend="mock"),
        tracer=tracer,
    )

    trip_step_index: Optional[int] = None
    tokens_spent = 0

    for idx, step in enumerate(scenario.steps):
        tokens_spent += step.tokens
        for ev in _events_for_step(step, idx + 1):
            monitor.observe(ev)
            if tracer.trip_log and trip_step_index is None:
                trip_step_index = idx
        if trip_step_index is not None and stop_on_first_trip:
            break

    tripped = trip_step_index is not None
    first = tracer.trip_log[0] if tracer.trip_log else None

    # -- token economics ------------------------------------------------
    tokens_saved = scenario.tokens_after_index(trip_step_index) if tripped else 0
    supervision = tracer.recoveries * cost.recovery_call_tokens
    if cost.drift_probe_tokens:
        supervision += cost.drift_probe_tokens * sum(
            1 for s in scenario.steps[: (trip_step_index or len(scenario.steps) - 1) + 1]
            if s.goal
        )
    if tripped and not scenario.label.should_trip:
        supervision += cost.false_positive_penalty

    steps_late = None
    if tripped and scenario.label.onset_index is not None:
        steps_late = trip_step_index - scenario.label.onset_index

    return ScenarioResult(
        scenario_id=scenario.id,
        family=scenario.family,
        should_trip=scenario.label.should_trip,
        known_gap=scenario.label.known_gap,
        tripped=tripped,
        trip_detector=first["detector"] if first else None,
        trip_step_index=trip_step_index,
        trip_severity=first["severity"] if first else None,
        all_trips=list(tracer.trip_log),
        recoveries=tracer.recoveries,
        tokens_spent=tokens_spent,
        tokens_saved=tokens_saved,
        supervision_cost=supervision,
        steps_late=steps_late,
        expected_detector=scenario.label.detector,
    )


def run_suite(scenarios: list[Scenario],
              disabled: Optional[set[str]] = None,
              extra_detectors: Optional[list[Detector]] = None,
              cost: CostModel = DEFAULT_COST) -> list[ScenarioResult]:
    return [run_scenario(s, disabled=disabled, extra_detectors=extra_detectors, cost=cost)
            for s in scenarios]
