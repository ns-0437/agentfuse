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

    def __init__(self, patience: int = 6):
        self.patience = patience
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
        elif event.type is EventType.LLM_CALL:
            self._work_since_progress += _THINK_WEIGHT
            self._actions_since_progress += 1
        else:
            return None

        # Require at least one tool call: a purely deliberative stretch is the
        # agent thinking, not the agent stuck in a trap.
        if self._work_since_progress >= self.patience and self._tools_since_progress >= 1:
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
