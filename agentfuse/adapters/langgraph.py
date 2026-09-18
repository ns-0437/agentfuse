"""LangGraph adapter.

LangGraph models an agent as a stateful graph; the natural integration point is
a callback handler that fires on tool start/end and LLM end, plus an optional
supervisor node. This adapter exposes a LangChain-compatible callback handler so
the breaker observes the run without you rewriting the graph. When a steering
directive is produced it is stashed on the handler; a thin supervisor node reads
it and appends it to the message state before the next agent node runs.

Enforcement, not just a message
--------------------------------
An independent external review (2026-09-17) found that a PAUSE/ABORT directive
used to become nothing more than a ``"[HALT] ..."`` string appended to the
message state by ``supervisor_node`` — the callback itself returned normally,
and nothing stopped the graph from dispatching its next tool call. A model is
free to ignore a string in its own context; that is not enforcement.
``FuseCallbackHandler`` now raises :class:`BreakerInterrupt` the moment a
PAUSE/ABORT directive is produced, and keeps raising on every subsequent
dispatch while halted, relying on ``raise_error = True`` (a documented
LangChain callback-handler attribute) so the callback manager propagates the
exception instead of logging and swallowing it. Verified directly against a
real ``langchain_core`` ``@tool``-decorated function invoked through
``.invoke()`` (the path LangGraph's prebuilt tool nodes use): with
``raise_error = True``, an exception raised from ``on_tool_start`` reaches the
caller and the tool body never runs. A plain ``.run()`` call does NOT route
through the modern callback path and will not be stopped this way — always
drive tools through ``.invoke()`` under a graph, which LangGraph does.
"""

from __future__ import annotations

from typing import Any, Optional

from ..events import AgentEvent, EventType
from ..monitor import CircuitBreakerMonitor, MonitorConfig, Directive, DirectiveKind

try:
    from langchain_core.callbacks import BaseCallbackHandler as _LCBase
except ImportError:      # langchain-core is this adapter's own optional extra
    _LCBase = object


class BreakerInterrupt(Exception):
    """Raised to stop LangGraph dispatch once the breaker has ordered a halt.

    Unlike an ``INJECT`` steer (delivered as a message via ``supervisor_node``
    for the model to read and, hopefully, obey), PAUSE/ABORT are not requests
    a model can choose to honour — see this module's docstring. Carries the
    ``Directive`` so a caller can distinguish "pause, a human should look at
    this" from "abort, stop for good".
    """

    def __init__(self, directive: Directive):
        self.directive = directive
        super().__init__(directive.steering_text or "circuit breaker halt")


class FuseCallbackHandler(_LCBase):
    """LangChain ``BaseCallbackHandler``-compatible breaker.

    When ``langchain_core`` is installed this IS a real ``BaseCallbackHandler``
    subclass — pass an instance straight to ``callbacks=[handler]``, no
    multiple inheritance required::

        handler = FuseCallbackHandler(original_goal="...")

    An earlier version of this docstring recommended
    ``class Handler(BaseCallbackHandler, FuseCallbackHandler): ...``. That is
    a real, reproduced bug, not a style note: Python's MRO resolves
    ``on_tool_start``/``on_tool_end`` to ``BaseCallbackHandler``'s own no-op
    implementations whenever it is listed first, so a handler built that way
    silently observed *zero events*. Direct instantiation removes the failure
    mode entirely rather than asking every integrator to get inheritance order
    right.
    """

    #: A swallowed halt is not a halt. LangChain's callback manager checks this
    #: attribute before deciding whether an exception from a handler propagates
    #: or is only logged — see this module's docstring for the verified proof.
    raise_error = True

    def __init__(self, original_goal: str, monitor: Optional[CircuitBreakerMonitor] = None,
                 **config_kwargs: Any):
        super().__init__()
        self.monitor = monitor or CircuitBreakerMonitor(
            MonitorConfig(original_goal=original_goal, **config_kwargs)
        )
        self._step = 0
        self.pending_steering: Optional[str] = None
        #: Set once a PAUSE/ABORT directive fires; persists across calls so a
        #: later benign event cannot overwrite a stronger pending halt.
        self.halted: Optional[Directive] = None
        # Call id -> (tool name, node). A single slot here (the previous
        # design) meant interleaved calls to different tools attributed a
        # result to whichever call started LAST, not the one it actually
        # belongs to — reproduced directly: start(A), start(B), end(A) labeled
        # A's own result as B's. Keyed by LangChain's own run_id, which is
        # unique per in-flight call and supplied on both on_tool_start and
        # on_tool_end.
        self._inflight: dict[Any, tuple[str, str]] = {}

    def _check_halted(self) -> None:
        """Refuse to let the graph proceed once a halt has been ordered.

        Called before observing a new event, at every dispatch boundary this
        handler is wired into — not only at the moment of the trip — so a
        halt stays enforced for every subsequent tool/LLM call, not just the
        one that caused it.
        """
        if self.halted is not None:
            raise BreakerInterrupt(self.halted)

    def _observe(self, event: AgentEvent) -> None:
        d = self.monitor.observe(event)
        if d.kind is DirectiveKind.INJECT and d.steering_text:
            self.pending_steering = d.steering_text
            return
        if d.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT):
            self.halted = d
            self.pending_steering = f"[HALT] {d.steering_text}" if d.steering_text else None
            raise BreakerInterrupt(d)

    # LangChain callback signatures (subset)
    def on_tool_start(self, serialized: dict, input_str: str, *,
                       run_id: Any = None, **kwargs: Any) -> None:
        self._check_halted()
        self._step += 1
        name = (serialized or {}).get("name", "tool")
        node = kwargs.get("name", "agent")
        # Keyed by run_id so on_tool_end can recover the right (tool, node)
        # even when another call started and finished in between.
        self._inflight[run_id] = (name, node)
        self._observe(AgentEvent(
            type=EventType.TOOL_CALL, step=self._step, tool_name=name,
            tool_args={"input": input_str}, node=node,
            meta={"call_id": str(run_id)} if run_id is not None else {},
        ))

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self._check_halted()
        name, node = self._inflight.pop(run_id, (None, "agent"))
        self._observe(AgentEvent(
            type=EventType.TOOL_RESULT, step=self._step, text=str(output)[:200],
            tool_name=name, node=node,
            state={"tool_output": str(output)[:200]},
            meta={"call_id": str(run_id)} if run_id is not None else {},
        ))

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        """No detection here — only enforce a halt already in force.

        A halt must block the NEXT model call too, not only the next tool
        dispatch; without this, a run halted right after a tool result could
        still take one more (billed) LLM turn before anything stopped it.
        """
        self._check_halted()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._check_halted()
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
        """Optional LangGraph node: inject any pending steering into message state.

        For an INJECT directive this is the actual delivery mechanism. For a
        halt it is best-effort context for whoever reads the state afterward —
        the halt itself is already enforced by ``_check_halted`` raising, not
        by this node running.
        """
        if self.pending_steering:
            msgs = list(state.get("messages", []))
            msgs.append({"role": "system", "content": f"[CIRCUIT BREAKER STEERING] {self.pending_steering}"})
            self.pending_steering = None
            return {"messages": msgs}
        return {}

    def finish(self, status: str = "complete") -> dict:
        return self.monitor.finish(status)
