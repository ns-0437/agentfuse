"""OpenAI AgentKit adapter — first-class integration.

AgentKit runs agents as a graph of nodes with a runner and lifecycle hooks. This
adapter bridges those hooks to the AgentFuse monitor: each tool call, model turn,
and node transition becomes an ``AgentEvent``; when the breaker returns an INJECT
directive, we push the steering instruction back into the agent as a system-role
message before the next turn.

AgentKit's hook surface has evolved across releases, so this adapter is written
against the stable *concepts* (on_tool_call / on_message / on_handoff / on_step)
and exposes them as plain callables. Wire whichever hook names your installed
AgentKit version exposes to the matching ``on_*`` method below — the mapping is
one line each and documented inline.
"""

from __future__ import annotations

from typing import Any, Optional

from ..events import AgentEvent, EventType
from ..monitor import CircuitBreakerMonitor, MonitorConfig, Directive, DirectiveKind


class AgentKitBreaker:
    """Wraps an AgentKit agent/runner with a circuit breaker.

    Usage sketch (names depend on your AgentKit version)::

        from agents import Agent, Runner            # OpenAI AgentKit
        from agentfuse.adapters.agentkit import AgentKitBreaker

        breaker = AgentKitBreaker(original_goal=SYSTEM_PROMPT,
                                  max_tokens=200_000, max_cost_usd=5.0)

        agent = Agent(name="researcher", instructions=SYSTEM_PROMPT, tools=[...])

        # Register hooks (adjust to your AgentKit's hook API):
        hooks = breaker.as_hooks()
        result = Runner.run_sync(agent, input=task, hooks=hooks)
        breaker.finish()
    """

    def __init__(self, original_goal: str, monitor: Optional[CircuitBreakerMonitor] = None,
                 **config_kwargs: Any):
        self.monitor = monitor or CircuitBreakerMonitor(
            MonitorConfig(original_goal=original_goal, **config_kwargs)
        )
        self._step = 0
        self._pending_steering: Optional[str] = None

    # -- hook entry points ------------------------------------------------
    def on_step(self) -> None:
        self._step += 1

    def on_tool_call(self, tool_name: str, tool_args: dict, node: str = "agent") -> Directive:
        return self._observe(AgentEvent(
            type=EventType.TOOL_CALL, step=self._next(), node=node,
            tool_name=tool_name, tool_args=tool_args,
        ))

    def on_tool_result(self, tool_name: str, result: Any, state: Optional[dict] = None,
                       node: str = "agent") -> Directive:
        return self._observe(AgentEvent(
            type=EventType.TOOL_RESULT, step=self._step, node=node,
            tool_name=tool_name, text=str(result)[:200], state=state,
        ))

    def on_message(self, text: str, tokens_in: int = 0, tokens_out: int = 0,
                   cost_usd: float = 0.0, goal: Optional[str] = None,
                   node: str = "agent") -> Directive:
        return self._observe(AgentEvent(
            type=EventType.LLM_CALL, step=self._next(), node=node, text=text,
            tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd, goal=goal,
        ))

    def on_handoff(self, from_node: str, to_node: str) -> Directive:
        return self._observe(AgentEvent(
            type=EventType.ROUTE, step=self._step, node=to_node,
            text=f"{from_node} -> {to_node}",
        ))

    def on_state_update(self, state: dict, node: str = "agent") -> Directive:
        return self._observe(AgentEvent(
            type=EventType.STATE_UPDATE, step=self._step, node=node, state=state,
        ))

    # -- glue -------------------------------------------------------------
    def _next(self) -> int:
        self._step += 1
        return self._step

    def _observe(self, event: AgentEvent) -> Directive:
        directive = self.monitor.observe(event)
        if directive.kind is DirectiveKind.INJECT and directive.steering_text:
            # Stash the steering text; the runner should inject it as a
            # system/developer message before the agent's next turn.
            self._pending_steering = directive.steering_text
        return directive

    def take_steering(self) -> Optional[str]:
        """Pop any pending steering instruction to inject before the next turn."""
        s, self._pending_steering = self._pending_steering, None
        return s

    def as_hooks(self) -> dict:
        """Return a dict you can adapt to AgentKit's ``RunHooks`` / lifecycle API.

        Map these to your installed AgentKit hook names, e.g.::

            class FuseHooks(RunHooks):
                async def on_tool_start(self, ctx, agent, tool):
                    breaker.on_tool_call(tool.name, ctx.tool_arguments, agent.name)
                async def on_tool_end(self, ctx, agent, tool, result):
                    breaker.on_tool_result(tool.name, result, node=agent.name)
                async def on_handoff(self, ctx, from_agent, to_agent):
                    breaker.on_handoff(from_agent.name, to_agent.name)
        """
        return {
            "on_tool_call": self.on_tool_call,
            "on_tool_result": self.on_tool_result,
            "on_message": self.on_message,
            "on_handoff": self.on_handoff,
            "on_state_update": self.on_state_update,
        }

    def finish(self, status: str = "complete") -> dict:
        return self.monitor.finish(status)
