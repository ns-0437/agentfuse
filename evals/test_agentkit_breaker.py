"""AgentKitBreaker — the adapter labeled "first-class" and exported at the top
level of agentfuse.adapters, yet never exercised by any test before this file.

Found while auditing adapter discoverability: `agentfuse/adapters/__init__.py`
re-exports `AgentKitBreaker` in `__all__` and its own module docstring calls
it first-class, while `FuseRunHooks` (agentkit_hooks.py) -- the genuine,
tested `agents.RunHooks` subclass README.md actually recommends -- is
deliberately NOT re-exported (it hard-imports the optional openai-agents
package). Section 8.3/4.10 found 4 real bugs in openai_sdk.py/langgraph.py
within twenty minutes of writing their first tests; this class had never had
that exercise at all -- `test_public_api.py` only checks `hasattr(module,
"AgentKitBreaker")`, never calls a single one of its methods.

    pytest evals/test_agentkit_breaker.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import DirectiveKind  # noqa: E402
from agentfuse.adapters.agentkit import AgentKitBreaker  # noqa: E402

GOAL = "Rotate the production database credential."


def _make(**kwargs):
    return AgentKitBreaker(original_goal=GOAL, echo=False, **kwargs)


# ------------------------------------------------------------ step counting
def test_tool_call_and_its_result_share_one_step():
    b = _make()
    b.on_tool_call("search_files", {"pattern": "*.conn"})
    call_step = b._step
    b.on_tool_result("search_files", "0 files matched")
    assert b._step == call_step, (
        "a call and its paired result must be the same step, or the loop "
        "detector cannot pair them to compute a signature")


def test_successive_tool_calls_advance_the_step_counter():
    b = _make()
    b.on_tool_call("a", {})
    first = b._step
    b.on_tool_result("a", "ok")
    b.on_tool_call("b", {})
    second = b._step
    assert second == first + 1


def test_messages_advance_the_step_counter_too():
    b = _make()
    b.on_message("thinking...")
    first = b._step
    b.on_message("still thinking...")
    assert b._step == first + 1


# --------------------------------------------------------- directive return
def test_every_hook_returns_the_directive_the_monitor_produced():
    """Callers act on the return value directly (e.g. checking .kind for
    PAUSE/ABORT), not only through take_steering() -- confirm the plumbing
    from monitor.observe() to the public hook methods is not lossy.

    loop_threshold=2 trips on the SECOND identical (tool, args, result) --
    verified directly before writing this assertion, since guessing wrong
    here would pin the wrong contract rather than the real one.
    """
    b = _make(loop_threshold=2)
    d = b.on_tool_call("t", {"x": 1})
    assert d.kind is DirectiveKind.CONTINUE
    d = b.on_tool_result("t", "0 files matched")
    assert d.kind is DirectiveKind.CONTINUE
    d = b.on_tool_call("t", {"x": 1})
    assert d.kind is DirectiveKind.CONTINUE
    d = b.on_tool_result("t", "0 files matched")
    assert d.kind is DirectiveKind.INJECT, (
        f"expected the second identical repeat to trip the loop, got {d.kind}")


# --------------------------------------------------------- steering stash
def test_an_injected_steer_is_available_via_take_steering():
    b = _make(loop_threshold=2)
    for _ in range(2):
        b.on_tool_call("t", {"x": 1})
        b.on_tool_result("t", "0 files matched")
    steer = b.take_steering()
    assert steer is not None and len(steer) > 0


def test_take_steering_consumes_it_once():
    b = _make(loop_threshold=2)
    for _ in range(2):
        b.on_tool_call("t", {"x": 1})
        b.on_tool_result("t", "0 files matched")
    assert b.take_steering() is not None
    assert b.take_steering() is None, "a steer must not be handed out twice"


def test_no_steering_is_stashed_on_a_healthy_run():
    b = _make()
    for i in range(5):
        b.on_tool_call("t", {"i": i})
        b.on_tool_result("t", "ok", state={"advanced_at": i})
    assert b.take_steering() is None


# ------------------------------------------------------------------ finish
def test_finish_returns_a_summary_dict():
    b = _make()
    b.on_message("hello")
    summary = b.finish("complete")
    assert summary["status"] == "complete"


# --------------------------------------------------------------- as_hooks
def test_as_hooks_returns_bound_methods_that_still_touch_this_instance():
    """A dict of unbound functions would silently lose all state -- confirm
    the returned callables are bound to the SAME breaker, not copies."""
    b = _make()
    hooks = b.as_hooks()
    hooks["on_message"]("hi")
    assert b._step == 1, "the hook dict's callables must be bound to self"
