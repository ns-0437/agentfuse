"""Real OpenAI AgentKit (`openai-agents`) integration via ``RunHooks``.

This is the production wiring — a genuine ``agents.RunHooks`` subclass that
observes the live agent lifecycle (LLM turns, tool calls, handoffs, tool
results) and feeds every event into the AgentFuse circuit breaker. When a
detector trips and the breaker wants to steer or halt, the hook raises
``BreakerInterrupt`` to abort the in-flight ``Runner.run``; the caller catches
it, appends the steering instruction to the conversation, and re-runs — the
standard, transferable way to inject corrective input into an Agents SDK run.

Import is lazy on ``agents`` so the AgentFuse core stays dependency-free: only
users who actually run against AgentKit need the SDK installed.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agents import RunHooks  # requires `pip install openai-agents`

from ..events import AgentEvent, EventType
from ..monitor import (
    CircuitBreakerMonitor,
    MonitorConfig,
    Directive,
    DirectiveKind,
)


class BreakerInterrupt(Exception):
    """Raised from a hook to abort a runaway ``Runner.run``.

    Carries the ``Directive`` the breaker produced so the driver can decide
    whether to inject steering and resume, or stop and escalate.
    """

    def __init__(self, directive: Directive):
        self.directive = directive
        super().__init__(directive.steering_text or "circuit breaker interrupt")


class FuseRunHooks(RunHooks):
    """Drop-in ``RunHooks`` that supervises an AgentKit run with AgentFuse."""

    def __init__(self, original_goal: str, monitor: Optional[CircuitBreakerMonitor] = None,
                 **config_kwargs: Any):
        self.monitor = monitor or CircuitBreakerMonitor(
            MonitorConfig(original_goal=original_goal, **config_kwargs)
        )
        self._step = 0
        self.pending_steering: Optional[str] = None
        self.last_directive: Optional[Directive] = None
        # A trip raised from inside on_tool_end lands in the SDK's tool executor,
        # which wraps it into a tool-execution failure and destroys the clean
        # interrupt. So a trip observed during tool handling is deferred and
        # raised at the next turn boundary instead — which is also the better
        # place to stop, since it halts before the next model call is paid for.
        self._deferred: Optional[Directive] = None

    # -- lifecycle hooks -------------------------------------------------
    async def on_agent_start(self, context, agent) -> None:
        self._observe(AgentEvent(
            type=EventType.ROUTE, step=self._step, node=getattr(agent, "name", "agent"),
            text=f"enter {getattr(agent, 'name', 'agent')}",
        ))

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self._observe(AgentEvent(
            type=EventType.ROUTE, step=self._step, node=getattr(to_agent, "name", "agent"),
            text=f"{getattr(from_agent,'name','?')} -> {getattr(to_agent,'name','?')}",
        ))

    async def on_llm_end(self, context, agent, response) -> None:
        node = getattr(agent, "name", "agent")
        usage = getattr(response, "usage", None)
        tin = int(getattr(usage, "input_tokens", 0) or 0)
        tout = int(getattr(usage, "output_tokens", 0) or 0)

        text_parts: list[str] = []
        tool_calls = []
        for item in getattr(response, "output", []) or []:
            itype = getattr(item, "type", None)
            if itype == "function_call":
                tool_calls.append(item)
            elif itype == "message":
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        text_parts.append(t)
        text = " ".join(text_parts).strip()

        # One LLM_CALL event carries the reasoning text (for drift) + token spend.
        self._step += 1
        self._observe(AgentEvent(
            type=EventType.LLM_CALL, step=self._step, node=node,
            text=text or None, goal=text or None, tokens_in=tin, tokens_out=tout,
        ))

        # One TOOL_CALL event per function call — args live here, on the response.
        for tc in tool_calls:
            try:
                args = json.loads(getattr(tc, "arguments", "") or "{}")
            except Exception:
                args = {"_raw": getattr(tc, "arguments", None)}
            self._step += 1
            self._observe(AgentEvent(
                type=EventType.TOOL_CALL, step=self._step, node=node,
                tool_name=getattr(tc, "name", "tool"), tool_args=args,
            ))

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        """Safe boundary: surface any trip deferred from tool handling."""
        if self._deferred is not None:
            directive, self._deferred = self._deferred, None
            raise BreakerInterrupt(directive)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        # Mark genuine progress only when the result actually advances the task,
        # so failed/empty tool results don't reset the loop/stall detectors.
        text = str(result)
        progressed = any(k in text.lower() for k in ("rotated", "secret-", "token:"))
        state = {"progress": True, "result": text[:120]} if progressed else None
        self._observe(AgentEvent(
            type=EventType.TOOL_RESULT, step=self._step, node=getattr(agent, "name", "agent"),
            tool_name=getattr(tool, "name", None), text=text[:200], state=state,
        ), defer_interrupt=True)

    # -- glue ------------------------------------------------------------
    def _observe(self, event: AgentEvent, defer_interrupt: bool = False) -> None:
        directive = self.monitor.observe(event)
        self.last_directive = directive
        if directive.kind is DirectiveKind.INJECT and directive.steering_text:
            self.pending_steering = directive.steering_text
        elif directive.kind not in (DirectiveKind.PAUSE, DirectiveKind.ABORT):
            return

        if defer_interrupt:
            # Raised at the next turn boundary — see on_llm_start.
            self._deferred = directive
            return
        raise BreakerInterrupt(directive)

    def take_steering(self) -> Optional[str]:
        s, self.pending_steering = self.pending_steering, None
        return s

    def finish(self, status: str = "complete") -> dict:
        return self.monitor.finish(status)
