"""No-progress (stall) detector.

Distinct from the loop detector: an agent can churn through *different* tool
calls and reasoning turns yet never move the working state — the hallmark of a
logical trap, where it reasons busily and coherently from a false premise.

A note on the previous implementation, because the bug is instructive. It only
incremented its stall counter when a ``STATE_UPDATE`` event arrived carrying the
*same* hash as before. But during a genuine stall no state updates are emitted
at all — that is precisely what a stall is — so the counter never moved and the
trip condition was unreachable. The detector required evidence of state changes
in order to detect the absence of state changes, and could never fire on the one
failure it existed to catch. The eval measured its causal contribution at exactly
0.0 F1, which is how it was found.

The definition used now is the direct one: **count actions taken since the last
genuine advance, and trip when that count exceeds patience.** Any state update
carrying a new hash resets the count; a repeated hash is a no-op write and counts
as activity rather than progress.

Tool calls are weighted more heavily than reasoning turns. An agent that thinks
for several turns before acting is deliberating, which is normal and healthy;
an agent that *acts* repeatedly and changes nothing is stuck.

Never judge a call before its result arrives
--------------------------------------------
The threshold is tested **only on a tool result**, never on the call or the model
turn that precedes it. Weight still accumulates on those events; what changes is
when the verdict is allowed to happen.

This is not a detail. A tool step emits ``llm_call -> tool_call -> tool_result``,
so evaluating on the call meant the counter could cross patience on the
``tool_call`` and halt the run **one event before the result that would have
reset it**. Measured: every benign retry-then-success scenario tripped on the
final, successful attempt — the agent had already issued the call that fixed
everything, and the supervisor killed it in the gap before the answer came back.
16 false positives, all on runs that were about to succeed.

The loop detector learned this same lesson earlier and independently (see its
module docstring: a pure call-counter "fires on that third call *before its
result arrives*, killing a run that was about to work"). Two detectors, the same
mistake, found twice — so it is written down here as a rule rather than a fix:
**a supervisor must not act on an action whose outcome it has not yet seen.**
"""

from __future__ import annotations

from typing import Optional

from ..events import AgentEvent, EventType
from .base import Detector, Trip, Severity

# A tool call is a full unit of "doing something"; a reasoning turn is partial.
# This keeps ordinary deliberation from being mistaken for a stall.
_TOOL_WEIGHT = 1.0
_THINK_WEIGHT = 0.5


class NoProgressDetector(Detector):
    name = "progress"

    #: How far past patience a run with NO advance yet is allowed to go before
    #: the grace period is overridden. Bounds the blind window.
    GRACE_MULTIPLIER = 2.0

    def __init__(self, patience: int = 6, calibrator=None):
        self.patience = patience
        self.calibrator = calibrator
        self._last_state_hash: Optional[str] = None
        self._work_since_progress = 0.0   # weighted actions since the last advance
        self._actions_since_progress = 0  # raw count, for the evidence payload
        self._tools_since_progress = 0
        self._last_progress_step = 0

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        # A state update carrying a NEW hash is the only thing that counts as
        # progress. A repeated hash is a no-op write, not an advance.
        h = event.state_hash
        if h is not None and h != self._last_state_hash:
            self._last_state_hash = h
            self._work_since_progress = 0.0
            self._actions_since_progress = 0
            self._tools_since_progress = 0
            self._last_progress_step = event.step
            return None

        if event.type is EventType.TOOL_CALL:
            self._work_since_progress += _TOOL_WEIGHT
            self._actions_since_progress += 1
            self._tools_since_progress += 1
            return None          # the outcome of this call is not known yet
        elif event.type is EventType.LLM_CALL:
            self._work_since_progress += _THINK_WEIGHT
            self._actions_since_progress += 1
            return None          # a model turn is a decision, not an outcome
        elif event.type is not EventType.TOOL_RESULT:
            return None

        # Only a tool result is a moment where the outcome is known. Reaching
        # here means the result carried no state advance (that case returned
        # above), so the work counted so far genuinely produced nothing.

        effective = (self.calibrator.stall_patience(self.patience)
                     if self.calibrator else self.patience)

        # Grace period: before the run has demonstrated its own rhythm even once,
        # there is no evidence to distinguish "exploring" from "stuck" — an agent
        # that legitimately makes several calls between milestones looks
        # identical to one going nowhere. So hold fire until the first genuine
        # advance, bounded by a hard ceiling so a run that NEVER advances (the
        # stall we most want to catch) is still caught, just later.
        if self.calibrator is not None and self._last_state_hash is None:
            if self._work_since_progress < effective * self.GRACE_MULTIPLIER:
                return None

        # Require at least one tool call: a purely deliberative stretch is the
        # agent thinking, not the agent stuck in a trap.
        if self._work_since_progress >= effective and self._tools_since_progress >= 1:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"{self._actions_since_progress} actions "
                    f"({self._tools_since_progress} tool calls) since the last state "
                    f"advance at step {self._last_progress_step} — the agent is busy "
                    f"but the task is not moving, which suggests reasoning from a "
                    f"false premise."
                ),
                evidence={
                    "actions_without_progress": self._actions_since_progress,
                    "tool_calls_without_progress": self._tools_since_progress,
                    "weighted_work": round(self._work_since_progress, 1),
                    "last_progress_step": self._last_progress_step,
                    "steps_without_progress": event.step - self._last_progress_step,
                },
            )
        return None

    def reset(self) -> None:
        self._work_since_progress = 0.0
        self._actions_since_progress = 0
        self._tools_since_progress = 0
