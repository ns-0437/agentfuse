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

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The benchmark is free and deterministic BY DEFAULT. Without this, simply
# having a key configured would turn an ordinary `pytest` run into thousands of
# billed embedding and reasoning calls as a side effect. Opt in deliberately
# with AGENTFUSE_OFFLINE=0 (or `run_eval.py --live`).
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse import (  # noqa: E402
    AgentEvent, EventType, CircuitBreakerMonitor, MonitorConfig, Tracer,
)
from agentfuse.detectors import (  # noqa: E402
    Detector, LoopDetector, DriftDetector, NoProgressDetector, SpendDetector,
    RateOfProgressDetector,
)
from agentfuse.recovery import RecoveryEngine  # noqa: E402

from agentfuse.recovery import RecoveryAction  # noqa: E402

from .schema import Scenario, ScenarioResult, CostModel, DEFAULT_COST  # noqa: E402
from .steering import RecoveryOutcome, score_steering  # noqa: E402


# Shared baseline configuration. Scenarios may override individual keys, but the
# defaults are held constant across the suite so results aren't tuned per-case.
DEFAULT_CONFIG = {
    "loop_threshold": 3,
    # None = let the detector pick per mode. Hard-coding the LEXICAL value
    # here silently disabled drift detection the moment embeddings were
    # enabled: embedding similarities sit around 0.6-0.8, so a 0.20
    # threshold can never trip. Drift recall went to 0% and looked like a
    # model failure rather than a config bug.
    "drift_threshold": None,
    "stall_patience": 6,
    "rate_patience": 8,
    "max_tokens": 100_000,
    "max_cost_usd": None,
    "burst_window": 6,
    "burst_tokens": None,
    "max_recoveries": 5,
}


class RecordingTracer(Tracer):
    """Captures trips/recoveries in memory and prints nothing."""

    def __init__(self) -> None:
        super().__init__(jsonl_path=None, echo=False)
        self.trip_log: list[dict] = []
        self.recovery_log: list[dict] = []
        self.paths: list = []          # the SteeringPath objects themselves

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
        self.paths.append(path)
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
    # Last, mirroring MonitorConfig's default ordering: on an event both this and
    # the spend detector could claim, the budget breach is the more actionable
    # diagnosis. Ablation keys on the name, so "rate" can be dropped like any other.
    if "rate" not in disabled and cfg.get("rate_patience") is not None:
        built.append(RateOfProgressDetector(patience=cfg["rate_patience"]))
    if extra:
        built.extend(extra)
    return built


def _events_for_step(step, step_no: int) -> list[AgentEvent]:
    """Expand one scripted step into the events an adapter would emit.

    Fidelity matters more than convenience here. Comparing this against a real
    ``openai-agents`` trace (``runs/real_agentkit.jsonl``) showed the benchmark
    was emitting a stream production never produces:

        real       llm_call -> tool_call -> tool_result [-> state_update]
        synthetic              tool_call -> tool_result [-> state_update]

    A model turn *precedes every tool call* in production — that is where the
    tool call comes from — and the token usage is attributed to that turn, not
    to the tool call. Omitting it meant the stall detector's activity counter
    accumulated at roughly two-thirds the real rate, and the drift detector was
    probed far less often than it will be in real use. Thresholds calibrated
    against that stream would not transfer. So tool steps now emit the
    preceding model turn, and carry the tokens on it.
    """
    events: list[AgentEvent] = []
    if step.kind == "tool":
        # The model turn that decided to make this call.
        events.append(AgentEvent(
            type=EventType.LLM_CALL, step=step_no, node=step.node,
            text=step.text, goal=step.goal,
            tokens_in=step.tokens_in, tokens_out=step.tokens_out,
        ))
        events.append(AgentEvent(
            type=EventType.TOOL_CALL, step=step_no, node=step.node,
            tool_name=step.tool_name, tool_args=step.tool_args,
            goal=step.goal,
        ))
        # Progress rides ON the tool result, exactly as the AgentKit hooks do.
        # Emitting a standalone STATE_UPDATE here (which no adapter produces)
        # is what hid the LoopDetector reset bug.
        events.append(AgentEvent(
            type=EventType.TOOL_RESULT, step=step_no, node=step.node,
            tool_name=step.tool_name, text=step.result,
            state=({"advanced_at": step_no, "last": str(step.result)[:80]}
                   if step.progress else None),
        ))
    else:  # think
        events.append(AgentEvent(
            type=EventType.LLM_CALL, step=step_no, node=step.node,
            text=step.text, goal=step.goal,
            tokens_in=step.tokens_in, tokens_out=step.tokens_out,
        ))
        if step.progress:
            events[-1].state = {"advanced_at": step_no, "last": str(step.text)[:80]}
    return events


def _drive_recovery(scenario: Scenario, monitor, tracer, first_trip,
                    trip_step_index, max_extra_steps: int = 40) -> Optional[RecoveryOutcome]:
    """Close the loop: keep steering until a rung works, or escalation happens.

    This replaced a measurement that could not fail. The previous version scored
    the steering with our own rubric and unlocked the recovery branch whenever
    that rubric approved our own instruction — but the rubric and the ladder's
    instruction templates were written together, so it always approved them.
    "Recovery rate 95%" meant "our templates match our rubric", and every rung
    unlocked the same branch, so climbing the ladder scored identically to never
    climbing it. The ladder's entire value was unmeasured.

    Now the scenario declares ``responds_to``: the one rung this agent actually
    obeys, which is ground truth independent of anything we generate. The agent
    keeps failing until the ladder reaches that rung. So the metric answers a
    question that can genuinely come out wrong: **does the ladder find the
    intervention that works, before recoveries run out?**
    """
    if not scenario.label.should_trip or first_trip is None:
        return None
    if not scenario.recovery_branch:
        return None

    out = RecoveryOutcome(attempted=bool(tracer.paths))
    seen: list[str] = []
    tail = scenario.steps[-1]          # the failing behaviour the agent repeats
    step_no = (trip_step_index or 0) + 1

    def judge(path) -> bool:
        s = score_steering(
            path, original_goal=scenario.goal,
            trip_detector=first_trip["detector"],
            trip_severity=first_trip["severity"],
            failing_tool=scenario.failing_tool,
            previous_instructions=list(seen))
        seen.append(path.instruction or "")
        out.steering_scores.append(s)
        return s.usable

    # The trip that brought us here already produced a steer.
    for path in tracer.paths:
        usable = judge(path)
        if path.action is RecoveryAction.ESCALATE:
            out.escalated = True
        elif usable and path.strategy == scenario.responds_to:
            out.recovered = True

    out.usable = any(s.usable for s in out.steering_scores)
    if out.recovered or out.escalated:
        if out.recovered:
            _replay_branch(scenario, monitor, out, step_no)
        return out

    # Not yet: the agent ignores what it was told and carries on failing. Each
    # repetition trips the breaker again, which climbs the ladder.
    seen_paths = len(tracer.paths)
    for _ in range(max_extra_steps):
        step_no += 1
        for ev in _events_for_step(tail, step_no):
            monitor.observe(ev)

        for path in tracer.paths[seen_paths:]:
            seen_paths += 1
            usable = judge(path)
            if path.action is RecoveryAction.ESCALATE:
                out.escalated = True
            elif usable and path.strategy == scenario.responds_to:
                out.recovered = True

        out.usable = any(s.usable for s in out.steering_scores)
        if out.recovered:
            _replay_branch(scenario, monitor, out, step_no)
            return out
        if out.escalated:
            return out
    return out


def _replay_branch(scenario: Scenario, monitor, out: RecoveryOutcome, base: int) -> None:
    """The agent takes the advice and completes the task."""
    for j, step in enumerate(scenario.recovery_branch):
        out.tokens_to_recovery += step.tokens
        for ev in _events_for_step(step, base + j + 1):
            monitor.observe(ev)


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
                             rate_patience=cfg["rate_patience"],
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

    # -- did the steering actually work? --------------------------------
    recovery = _drive_recovery(scenario, monitor, tracer, first, trip_step_index)
    if recovery is not None:
        tokens_spent += recovery.tokens_to_recovery

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
        recovery=recovery,
    )


def run_suite(scenarios: list[Scenario],
              disabled: Optional[set[str]] = None,
              extra_detectors: Optional[list[Detector]] = None,
              cost: CostModel = DEFAULT_COST) -> list[ScenarioResult]:
    return [run_scenario(s, disabled=disabled, extra_detectors=extra_detectors, cost=cost)
            for s in scenarios]
