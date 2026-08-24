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

Concurrency
-----------
``observe`` is serialised by a re-entrant lock. This is not defensive
programming; the unguarded version was measured failing. Driving one monitor
from two threads with the GIL switch interval lowered to expose the window
produced both mis-attributed trips — a trip naming a tool the agent never
called, which then becomes a steering instruction telling it to stop using
that tool — and an outright ``RuntimeError: deque mutated during iteration``
from inside a detector. A supervisor that crashes the run it is supervising is
worse than no supervisor, and this project's own adapters raise into the agent's
call stack.

The lock makes each event atomic. It does **not** make interleaved events from
*different agents* semantically meaningful: detector counters are per-monitor, so
two agents sharing one monitor still blend their histories. **One monitor per
agent run** remains the supported model, and is what every adapter here builds.
"""

from __future__ import annotations

import os
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .checkpoint import SQLiteCheckpointStore, load_state_dict, state_dict
from .events import AgentEvent, EventType, ExecutionSnapshot, SeenStateTracker
from .notify import Notification, Notifier, build_notifier
from .detectors import (Detector, LoopDetector, DriftDetector, SpendDetector,
                        NoProgressDetector, RateOfProgressDetector)
from .calibration import AdaptiveCalibrator
from .detectors.base import Severity
from .recovery import RecoveryEngine, SteeringPath, RecoveryAction
from .memory import JSONMemory
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
    # The agent's model, used to price events that arrive without a cost. Without
    # it `max_cost_usd` cannot be enforced — see agentfuse/pricing.py.
    model: Optional[str] = None
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
    # Durable run state. Without it a restart resets every counter — including
    # the spend ceiling, which means a runaway run comes back with a brand new
    # budget and the guard that exists to bound it has been rearmed rather than
    # enforced. Set a path to survive restarts; None keeps the old behaviour.
    checkpoint_path: Optional[str] = None
    # Events between saves. Bounds how much is lost to a hard kill; the write is
    # a single small sqlite transaction, so this can be low.
    checkpoint_every: int = 20
    # Identifies the run to resume. Generated when not supplied.
    run_id: Optional[str] = None
    # Where an escalation actually goes. Without one, "escalate to a human" means
    # printing to a console that, on an unattended run, nobody is reading.
    escalation_webhook: Optional[str] = None
    # Drop agent-produced text from the escalation payload. The trace is the
    # agent's reasoning and tool output, and posting it to an external URL is
    # data egress; set False where that matters more than the detail.
    escalation_include_agent_text: bool = True
    # Shared secret for HMAC-SHA256 signing of the escalation POST. Without it a
    # receiver cannot distinguish a real "your agent was halted" from anyone who
    # has learned the URL, and webhook URLs leak — into CI logs, screenshots and
    # config repos. Read from AGENTFUSE_ESCALATION_SECRET when not set here, so
    # the secret does not have to live in the same code as the config.
    escalation_secret: Optional[str] = None
    # Allow posting escalations over plaintext http://. Off by default: the
    # payload carries the goal, the failure reason and agent output.
    escalation_allow_insecure: bool = False


class CircuitBreakerMonitor:
    def __init__(self, config: MonitorConfig, detectors: Optional[list[Detector]] = None,
                 recovery: Optional[RecoveryEngine] = None, tracer: Optional[Tracer] = None,
                 notifier: Optional[Notifier] = None):
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
                model=config.model,
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

        # `checkpoint_path` promises durable run state, but a default-constructed
        # RecoveryEngine's memory is in-process only -- the ladder's climb
        # history (which rungs already failed, for which failure signature) was
        # silently lost on every restart, the same failure shape the spend
        # ceiling was already fixed for once (this module's own docstring). Only
        # wired in when the caller took the RecoveryEngine default AND asked for
        # durability; an explicit `recovery=` is a choice this must not override.
        if recovery is None and config.checkpoint_path:
            recovery = RecoveryEngine(memory=JSONMemory(path=config.checkpoint_path + ".memory.jsonl"))
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
        # Re-entrant: _handle_trip runs inside observe, and a tracer or detector
        # callback could re-enter. A plain Lock would deadlock on that path.
        self._lock = threading.RLock()
        self.run_id = config.run_id or f"run-{uuid.uuid4().hex[:12]}"
        self._store = (SQLiteCheckpointStore(config.checkpoint_path)
                       if config.checkpoint_path else None)
        self._events_since_checkpoint = 0
        # Misuse detection only; see _warn_if_shared_across_agents.
        self._agent_id: Optional[str] = None
        self._warned_shared = False
        self.notifier = notifier if notifier is not None else build_notifier(
            webhook_url=config.escalation_webhook, echo=config.echo,
            include_agent_text=config.escalation_include_agent_text,
            secret=(config.escalation_secret
                    or os.getenv("AGENTFUSE_ESCALATION_SECRET")),
            allow_insecure=config.escalation_allow_insecure)
        #: None until an escalation happens, then whether a human was reached.
        #: Kept distinct from "did we escalate" so a failed delivery cannot be
        #: mistaken for a successful one.
        self.escalation_delivered: Optional[bool] = None
        self.escalations = 0
        self._warned_no_channel = False
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
        # Whether a post-steer state is a GENUINE advance, not merely present.
        # See _verify_pending for why this cannot be "event.state is not None".
        self._verify_seen = SeenStateTracker()

    # ------------------------------------------------- durable run state
    def state(self) -> dict:
        """Everything a resumed run needs in order to keep counting."""
        return {
            "run_id": self.run_id,
            "totals": {
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "recovery_count": self.recovery_count,
                "steers_that_worked": self.steers_that_worked,
                "steers_that_failed": self.steers_that_failed,
                "escalations": self.escalations,
                # `False` here means "a human was needed and nobody was told" --
                # a fact this project's own docs call load-bearing (README: "None
                # means never needed, False means needed and nobody was told...
                # those must not look alike"). Losing it on a restart collapses
                # a real notification failure back to "nothing has happened yet".
                "escalation_delivered": self.escalation_delivered,
                "route_history": list(self.route_history),
                "current_goal": self.current_goal,
                "last_step": self.history[-1].step if self.history else 0,
            },
            "calibrator": state_dict(self.calibrator),
            "baseline": state_dict(self.calibrator.baseline),
            "verify_seen": state_dict(self._verify_seen),
            # Keyed by detector name, so reordering or adding a detector cannot
            # silently load one detector's counters into another.
            "detectors": {d.name: state_dict(d) for d in self.detectors},
        }

    def checkpoint(self) -> None:
        """Persist now. Safe to call at any time; a no-op without a store."""
        if self._store is None:
            return
        with self._lock:
            snap = self.state()
            try:
                self._store.save(self.run_id, snap, step=snap["totals"]["last_step"])
                self._events_since_checkpoint = 0
            except Exception:
                # A checkpoint failure must never take down the run being
                # supervised — the same rule the recovery engine follows.
                pass

    def restore(self, run_id: Optional[str] = None) -> bool:
        """Reload a previous run's counters. Returns whether anything was found.

        Detector state is matched **by name**, so a detector that is absent from
        the checkpoint simply starts fresh rather than inheriting a stranger's
        counters.
        """
        if self._store is None:
            return False
        saved = self._store.load(run_id or self.run_id)
        if not saved:
            return False
        with self._lock:
            totals = saved.get("totals", {})
            self.total_tokens = totals.get("total_tokens", 0)
            self.total_cost = totals.get("total_cost", 0.0)
            self.recovery_count = totals.get("recovery_count", 0)
            self.steers_that_worked = totals.get("steers_that_worked", 0)
            self.steers_that_failed = totals.get("steers_that_failed", 0)
            self.escalations = totals.get("escalations", 0)
            self.escalation_delivered = totals.get("escalation_delivered")
            self.route_history = list(totals.get("route_history", []))
            self.current_goal = totals.get("current_goal")
            load_state_dict(self.calibrator, saved.get("calibrator", {}))
            load_state_dict(self.calibrator.baseline, saved.get("baseline", {}))
            load_state_dict(self._verify_seen, saved.get("verify_seen", {}))
            by_name = saved.get("detectors", {})
            for d in self.detectors:
                if d.name in by_name:
                    load_state_dict(d, by_name[d.name])
            self.run_id = saved.get("run_id", self.run_id)
        return True

    # ------------------------------------------------------------------
    def _verify_pending(self, event: AgentEvent, new_trip: bool = False) -> None:
        """Decide whether the last steer worked, once there is evidence either way.

        A steer counts as having worked if the agent makes GENUINE state progress
        within ``verify_window`` steps of receiving it. It counts as failed if the
        breaker trips again first, or if the window closes with nothing advanced.

        Until this existed the memory only knew what had been *tried*, which is
        not enough to stop the engine repeating a correction that already failed.

        "Genuine" used to mean nothing more than ``event.state is not None`` --
        which sounds like it means progress, and does not. The production
        adapter (``adapters/openai_sdk.py``) sets ``state=`` on every single
        ``TOOL_RESULT``, unconditionally, so that check registered a steer as
        successful the instant ANY tool call landed afterward -- including the
        exact same failing call the agent was steered away from. Reproduced
        directly: an agent that repeats a call byte-for-byte after being
        steered off it still counted as a *worked* steer, because a result
        payload existed, which it always does.

        The fix reuses ``SeenStateTracker``, the same bounded-window novelty
        check ``NoProgressDetector`` and ``LoopDetector`` already use to tell a
        genuine advance from a repeat -- ``self._verify_seen`` is a dedicated
        instance so this cannot be silently disabled by ablating either
        detector. Every state is fed to it unconditionally (not only while a
        steer is pending), matching those detectors' own semantics: a state
        the run reached BEFORE the steer must not count as new progress after
        it just because the tracker started empty.
        """
        advanced = False
        if event.state is not None:
            advanced = self._verify_seen.advance(event.state_hash)

        if self._pending_steer is None:
            return
        path, injected_at = self._pending_steer

        if advanced and not new_trip:
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
        with self._lock:
            directive = self._observe_locked(event)
            self._events_since_checkpoint += 1
            # Always save on a trip, whatever the interval: that is the state a
            # crash is most likely to follow and the state most expensive to
            # lose, since it carries the recovery ladder's position.
            if self._store is not None and (
                    directive.kind is not DirectiveKind.CONTINUE
                    or self._events_since_checkpoint >= self.config.checkpoint_every):
                self.checkpoint()
            return directive

    def _warn_if_shared_across_agents(self, event: AgentEvent) -> None:
        """One monitor per agent run is the only supported model. Say so.

        Sharing a monitor between agents does not merely fail to be useful, it
        actively halts healthy runs. Measured with two agents on different goals
        driving one monitor: **7 spurious PAUSE directives across 4 steps**, both
        agents healthy on their own terms. The cause is structural rather than a
        bug to patch -- ``original_goal`` is singular, so every drift probe scores
        agent B's reasoning against agent A's objective, and the token and cost
        ceilings pool into one budget nobody set.

        Supporting it properly would mean per-agent goals, per-agent detector
        state and per-agent budgets, which is one monitor per agent with extra
        indirection. So this warns instead, once, rather than pretending.

        It can only fire when events carry ``meta['agent_id']``. Nothing else in
        an event distinguishes two agents from one agent walking several graph
        nodes, and inventing a signal that is not there would produce false
        alarms on ordinary LangGraph runs.
        """
        agent_id = (event.meta or {}).get("agent_id")
        if agent_id is None:
            return
        if self._agent_id is None:
            self._agent_id = agent_id
        elif agent_id != self._agent_id and not self._warned_shared:
            self._warned_shared = True
            warnings.warn(
                f"this CircuitBreakerMonitor is receiving events from more than "
                f"one agent ({self._agent_id!r} and {agent_id!r}). Detector state, "
                f"the original goal and the spend ceiling are all shared, so "
                f"drift is scored against another agent's objective and healthy "
                f"runs get paused. Use one monitor per agent run.",
                RuntimeWarning, stacklevel=3)

    def _observe_locked(self, event: AgentEvent) -> Directive:
        self._warn_if_shared_across_agents(event)
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
            self._escalate(event, trip, path)
            return Directive(kind, steering_text=path.instruction, path=path)

        path = self.recovery.recover(snapshot)
        self.recovery_count += 1
        self.tracer.recovery(path)

        # A successful steer clears detector counters so the same near-miss
        # pattern doesn't instantly re-trip on the next step.
        for d in self.detectors:
            d.reset()

        if path.action == RecoveryAction.ESCALATE:
            self._escalate(event, trip, path)
            return Directive(DirectiveKind.PAUSE, steering_text=path.instruction, path=path)
        if path.action == RecoveryAction.ABORT:
            self._escalate(event, trip, path)
            return Directive(DirectiveKind.ABORT, steering_text=path.instruction, path=path)

        self._pending_steer = (path, event.step)
        return Directive(DirectiveKind.INJECT, steering_text=path.instruction, path=path)

    # ------------------------------------------------------------------
    def _escalate(self, event: AgentEvent, trip, path) -> None:
        """Tell a human, and record whether one was actually reached.

        Delivery is tracked rather than assumed. A notifier that fails silently
        would recreate, one layer up, exactly the bug this feature fixes: an
        escalation nobody receives that looks identical to one they did.
        """
        self.escalations += 1
        note = Notification(
            run_id=self.run_id, reason=trip.reason, detector=trip.detector,
            step=event.step, goal=self.config.original_goal,
            action=path.action.value, instruction=path.instruction or "",
            evidence=trip.evidence or {}, totals=self.spend_totals,
        )
        if self.notifier is None:
            self.escalation_delivered = False
            # Once per monitor: this is a statement about configuration, not
            # about this particular escalation, and a long run can escalate
            # many times. Repeating it buries the signal it is meant to raise.
            if not self._warned_no_channel:
                self._warned_no_channel = True
                warnings.warn(
                    "the breaker escalated but no escalation channel is "
                    "configured, so no human was told. Set "
                    "MonitorConfig.escalation_webhook.",
                    RuntimeWarning, stacklevel=2)
            return
        try:
            delivered = bool(self.notifier.send(note))
        except Exception:      # a reporting failure must not kill the run
            delivered = False
        # Once any escalation fails to reach anyone, the run's answer to "was a
        # human told?" is no, and a later success does not undo that.
        self.escalation_delivered = delivered and self.escalation_delivered is not False
        if not delivered:
            warnings.warn(
                f"escalation for run {self.run_id} could not be delivered; "
                f"the run has halted and nobody has been told.",
                RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------
    def finish(self, status: str = "complete") -> dict:
        with self._lock:
            self.checkpoint()      # the final totals are worth keeping
            return self._finish_locked(status)

    @property
    def spend_totals(self) -> dict:
        """Cost as the spend detector sees it — the only place it is complete.

        The monitor's own ``total_cost`` only ever sums ``event.cost_usd``, which
        callers rarely populate. The detector additionally prices events from the
        model, so it is the authority; reporting the monitor's figure alongside
        it would print ``$0.00`` next to a real number.
        """
        for d in self.detectors:
            if d.name == "spend" and hasattr(d, "totals"):
                return d.totals
        return {"tokens": self.total_tokens, "cost_usd": round(self.total_cost, 4),
                "unpriced_tokens": 0, "cost_is_complete": False}

    def _finish_locked(self, status: str) -> dict:
        spend = self.spend_totals
        totals = {
            "status": status,
            "steps": self.history[-1].step if self.history else 0,
            "total_tokens": self.total_tokens,
            "total_cost_usd": spend.get("cost_usd", round(self.total_cost, 4)),
            # A non-zero value means the dollar figure above is a FLOOR.
            "unpriced_tokens": spend.get("unpriced_tokens", 0),
            "cost_is_complete": spend.get("cost_is_complete", False),
            "trips": self.tracer.trips,
            "recoveries": self.tracer.recoveries,
            "steers_verified_working": self.steers_that_worked,
            "steers_verified_failed": self.steers_that_failed,
            "escalations": self.escalations,
            # None = never escalated. False = escalated and NOBODY was told.
            "escalation_delivered": self.escalation_delivered,
            "route": " -> ".join(self.route_history[-12:]),
        }
        self.tracer.summary(totals)
        self.tracer.close()
        return totals
