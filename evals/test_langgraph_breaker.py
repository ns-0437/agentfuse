"""LangGraph adapter — real enforcement, not a callback that logs and continues.

No test drove this adapter against a real LangChain callback dispatch before
this file; ``evals/test_adapters_untested.py`` only ever calls its methods
directly, which cannot catch a bug in HOW the exception propagates through
LangChain's own callback manager. An independent external review
(2026-09-17) found three real defects here — this file pins all three plus
the fix that makes halt an actual enforcement mechanism, not a message.

    pytest evals/test_langgraph_breaker.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

langchain_core = pytest.importorskip("langchain_core", reason="pip install langchain-core")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from agentfuse.monitor import Directive, DirectiveKind  # noqa: E402
from agentfuse.adapters.langgraph import FuseCallbackHandler, BreakerInterrupt  # noqa: E402

GOAL = "Rotate the production database credential."


class Capture:
    """A stand-in monitor that returns a scripted directive on the Nth event."""

    def __init__(self, directive: Directive = None, trip_on: int = None):
        self.events: list = []
        self._directive = directive or Directive()
        self._trip_on = trip_on

    def observe(self, event):
        self.events.append(event)
        if self._trip_on is None or len(self.events) == self._trip_on:
            return self._directive
        return Directive()


# ------------------------------------------------------------- subclassing
def test_fuse_callback_handler_is_a_real_base_callback_handler():
    """The fix for the documented-inheritance bug: no multiple inheritance
    required at all. Direct instantiation must observe real events."""
    assert issubclass(FuseCallbackHandler, BaseCallbackHandler)
    cap = Capture()
    h = FuseCallbackHandler(GOAL, monitor=cap)
    h.on_tool_start({"name": "tool_a"}, "{}", run_id="a")
    assert len(cap.events) == 1, "direct instantiation must observe events"


# ------------------------------------------------------------- concurrency
def test_interleaved_tool_calls_keep_their_own_identity():
    """Reproduces the review's probe: start(A), start(B), end(A) must not
    attribute A's result to B."""
    cap = Capture()
    h = FuseCallbackHandler(GOAL, monitor=cap)
    h.on_tool_start({"name": "tool_a"}, "{}", run_id="a")
    h.on_tool_start({"name": "tool_b"}, "{}", run_id="b")
    h.on_tool_end("result_a", run_id="a")
    assert cap.events[-1].tool_name == "tool_a"

    h.on_tool_end("result_b", run_id="b")
    assert cap.events[-1].tool_name == "tool_b"


# ------------------------------------------------------------- enforcement
def test_a_halt_directive_raises_instead_of_returning_normally():
    """The core enforcement bug: PAUSE/ABORT used to become a string in the
    conversation while the callback returned normally and the graph kept
    going. It must now raise."""
    cap = Capture(Directive(DirectiveKind.ABORT, steering_text="Budget exhausted"))
    h = FuseCallbackHandler(GOAL, monitor=cap)
    with pytest.raises(BreakerInterrupt) as exc:
        h.on_tool_start({"name": "charge"}, "{}", run_id="r1")
    assert exc.value.directive.kind is DirectiveKind.ABORT


def test_a_halt_blocks_every_subsequent_dispatch_not_just_the_first():
    """A late benign event must not un-halt the run — the halt is a
    persistent state checked at every future dispatch, not a one-shot."""
    cap = Capture(Directive(DirectiveKind.PAUSE, steering_text="stop"), trip_on=1)
    h = FuseCallbackHandler(GOAL, monitor=cap)
    with pytest.raises(BreakerInterrupt):
        h.on_tool_start({"name": "first"}, "{}", run_id="r1")
    # A second, unrelated tool must still be blocked, even though the
    # monitor itself would return CONTINUE for it.
    with pytest.raises(BreakerInterrupt):
        h.on_tool_start({"name": "second"}, "{}", run_id="r2")
    with pytest.raises(BreakerInterrupt):
        h.on_llm_start({}, [])


def test_halt_actually_prevents_the_tool_from_running_through_real_langchain_dispatch():
    """The proof that matters: driven through LangChain's real ``.invoke()``
    callback dispatch (what LangGraph's tool nodes use), not just a direct
    method call — a halt must mean the tool body never executes."""
    executed = []

    @tool
    def charge_card(amount: str) -> str:
        """Charges a card."""
        executed.append(amount)
        return f"charged {amount}"

    cap = Capture(Directive(DirectiveKind.ABORT, steering_text="budget exhausted"),
                  trip_on=1)
    h = FuseCallbackHandler(GOAL, monitor=cap)
    with pytest.raises(BreakerInterrupt):
        charge_card.invoke({"amount": "100"}, config={"callbacks": [h]})
    assert executed == [], "the tool body ran despite an ABORT directive"


def test_an_inject_directive_still_only_delivers_a_message():
    """The soft path must be unaffected: INJECT does not raise, and the
    correction reaches supervisor_node exactly as before."""
    cap = Capture(Directive(DirectiveKind.INJECT, steering_text="Stop calling that tool."),
                  trip_on=1)
    h = FuseCallbackHandler(GOAL, monitor=cap)
    h.on_tool_start({"name": "search_files"}, "{}", run_id="a")  # must not raise
    out = h.supervisor_node({"messages": []})
    assert "CIRCUIT BREAKER STEERING" in out["messages"][-1]["content"]
    assert h.halted is None, "an INJECT must not be treated as a halt"
