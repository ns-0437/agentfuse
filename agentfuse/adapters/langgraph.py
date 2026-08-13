"""LangGraph adapter.

LangGraph models an agent as a stateful graph; the natural integration point is
a callback handler that fires on tool start/end and LLM end, plus an optional
supervisor node. This adapter exposes a LangChain-compatible callback handler so
the breaker observes the run without you rewriting the graph. When a steering
directive is produced it is stashed on the handler; a thin supervisor node reads
it and appends it to the message state before the next agent node runs.
"""

from __future__ import annotations

from typing import Any, Optional

from ..events import AgentEvent, EventType
from ..monitor import CircuitBreakerMonitor, MonitorConfig, DirectiveKind


class FuseCallbackHandler:
    """LangChain ``BaseCallbackHandler``-compatible breaker.

    Subclass or duck-type into ``langchain_core.callbacks.BaseCallbackHandler``::

        from langchain_core.callbacks import BaseCallbackHandler
        class Handler(BaseCallbackHandler, FuseCallbackHandler): ...

    then pass ``callbacks=[handler]`` to your graph invocation.
    """

    def __init__(self, original_goal: str, monitor: Optional[CircuitBreakerMonitor] = None,
                 **config_kwargs: Any):
        self.monitor = monitor or CircuitBreakerMonitor(
            MonitorConfig(original_goal=original_goal, **config_kwargs)
        )
        self._step = 0
        self.pending_steering: Optional[str] = None

    def _observe(self, event: AgentEvent) -> None:
        d = self.monitor.observe(event)
        if d.kind is DirectiveKind.INJECT and d.steering_text:
            self.pending_steering = d.steering_text
        elif d.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT) and d.steering_text:
            self.pending_steering = f"[HALT] {d.steering_text}"

    # LangChain callback signatures (subset)
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        self._step += 1
        name = (serialized or {}).get("name", "tool")
        # LangChain's on_tool_end does not tell us which tool finished, so the
        # name is carried over from the start callback. Without it the result
        # event had tool_name=None: the loop detector could not match the result
        # to the call that produced it, so it never formed a pair and never
        # fired, and no trip could ever name the offending tool.
        self._inflight_tool = name
        self._inflight_node = kwargs.get("name", "agent")
        self._observe(AgentEvent(
            type=EventType.TOOL_CALL, step=self._step, tool_name=name,
            tool_args={"input": input_str}, node=self._inflight_node,
        ))

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._observe(AgentEvent(
            type=EventType.TOOL_RESULT, step=self._step, text=str(output)[:200],
            tool_name=getattr(self, "_inflight_tool", None),
            node=getattr(self, "_inflight_node", "agent"),
            state={"tool_output": str(output)[:200]},
        ))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._step += 1
        text = ""
        try:
            text = response.generations[0][0].text
        except Exception:
            text = str(response)[:200]
        usage = {}
        try:
            usage = response.llm_output.get("token_usage", {})
        except Exception:
            pass
        self._observe(AgentEvent(
            type=EventType.LLM_CALL, step=self._step, text=text, goal=text or None,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        ))

    def supervisor_node(self, state: dict) -> dict:
        """Optional LangGraph node: inject any pending steering into message state."""
        if self.pending_steering:
            msgs = list(state.get("messages", []))
            msgs.append({"role": "system", "content": f"[CIRCUIT BREAKER STEERING] {self.pending_steering}"})
            self.pending_steering = None
            return {"messages": msgs}
        return {}

    def finish(self, status: str = "complete") -> dict:
        return self.monitor.finish(status)
