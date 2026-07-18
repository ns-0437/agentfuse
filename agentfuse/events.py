"""Event schema for AgentFuse.

Every observable thing an agent does — a tool call, a state mutation, a graph
route, an LLM turn — is normalized into an ``AgentEvent``. Adapters (AgentKit,
plain OpenAI SDK, LangGraph) translate their native callbacks into these events
and feed them to the monitor. Keeping one flat schema is what lets a single
circuit-breaker engine sit under three different frameworks.

Pure stdlib on purpose: the core must run with nothing installed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    """The kinds of things we observe on the execution graph."""

    TOOL_CALL = "tool_call"      # agent invoked a tool (name + args)
    TOOL_RESULT = "tool_result"  # tool returned
    LLM_CALL = "llm_call"        # a reasoning turn (carries token/cost accounting)
    STATE_UPDATE = "state_update"  # working state / scratchpad changed
    ROUTE = "route"              # graph edge fired (node -> node)
    TRIP = "trip"               # a detector tripped the breaker
    RECOVERY = "recovery"        # the steering/recovery engine produced a path
    RESUME = "resume"            # execution resumed after a steering injection
    ABORT = "abort"             # run halted / escalated to human
    COMPLETE = "complete"        # agent reported the objective done


def stable_hash(obj: Any) -> str:
    """Deterministic short hash of arbitrary JSON-able content.

    Used both for state-progress detection (did the working state actually
    change?) and for loop detection (normalized tool signature).
    """
    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        blob = str(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class AgentEvent:
    """A single normalized observation from the agent's execution."""

    type: EventType
    step: int = 0
    node: str = "agent"                       # which graph node / sub-agent
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    text: Optional[str] = None                # reasoning text / tool output snippet
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    state: Optional[dict] = None              # snapshot of working state, if provided
    goal: Optional[str] = None                # the current interpreted objective
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    @property
    def tool_signature(self) -> Optional[str]:
        """Normalized ``(tool, args)`` fingerprint used for loop detection.

        Two calls with identical tool + identical args collapse to the same
        signature even if issued steps apart — that repetition is the smoke
        that precedes an infinite-tool-loop fire.
        """
        if self.tool_name is None:
            return None
        return f"{self.tool_name}:{stable_hash(self.tool_args or {})}"

    @property
    def state_hash(self) -> Optional[str]:
        if self.state is None:
            return None
        return stable_hash(self.state)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class ExecutionSnapshot:
    """Frozen view of the run at the moment the breaker trips.

    This is exactly the packet handed to the (separate) recovery reasoning
    model so it can reason about *why* the agent went off the rails without
    ever being the agent that went off the rails.
    """

    step: int
    original_goal: str
    current_goal: Optional[str]
    total_tokens: int
    total_cost_usd: float
    route_history: list[str]
    recent_events: list[dict]
    trip_reason: str
    trip_detector: str
    trip_evidence: dict

    def to_prompt_context(self) -> str:
        """Render the snapshot as compact context for the reasoning model."""
        lines = [
            f"ORIGINAL OBJECTIVE (system prompt): {self.original_goal}",
            f"AGENT'S CURRENT INTERPRETED GOAL: {self.current_goal or '(unchanged)'}",
            f"STEP: {self.step} | TOKENS: {self.total_tokens} | COST: ${self.total_cost_usd:.4f}",
            f"RECENT GRAPH ROUTE: {' -> '.join(self.route_history[-8:])}",
            f"BREAKER TRIPPED BY: {self.trip_detector}",
            f"REASON: {self.trip_reason}",
            f"EVIDENCE: {json.dumps(self.trip_evidence, default=str)}",
            "RECENT ACTIONS:",
        ]
        for e in self.recent_events[-8:]:
            tn = e.get("tool_name")
            if tn:
                lines.append(f"  - step {e.get('step')}: called {tn}({json.dumps(e.get('tool_args') or {}, default=str)})")
            elif e.get("text"):
                lines.append(f"  - step {e.get('step')}: {str(e.get('text'))[:120]}")
        return "\n".join(lines)
