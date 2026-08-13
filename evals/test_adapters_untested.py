"""The two adapters the README advertised and nothing ever executed.

`README.md` says "One engine. Three runtimes." Only `agentkit_hooks` had tests.
`openai_sdk` and `langgraph` had never been run by anything — not once — which
made them the largest gap between what this project claims and what it shows.

Writing these found a real production bug in **both**, within minutes:

    LoopDetector cleared its counters on the mere PRESENCE of `event.state`.
    Both adapters attach `state` to EVERY tool result, because they cannot know
    whether a call achieved anything. So the detector reset after every single
    tool result, `_on_result` never ran, no pair was ever formed, and the loop
    detector was INERT in production — measured at 11 identical calls returning
    identical results with `_pairs` still 0.

The progress detector caught those runs as a backstop and named the wrong cause,
so the steering said "you are busy but the task is not moving" instead of "stop
calling `search_files`" — losing exactly the tool-naming that justifies keeping
the loop detector at all (measured elsewhere: attribution 84.1% vs 56.2%).

The 936-scenario benchmark could not see this. Its replay harness only emits
`state` on steps it has labelled as genuine progress, so it never produced the
one event shape that breaks the contract. **A benchmark only tests the event
orders it knows how to generate** — the same reason it missed the interleaving
bugs in `test_concurrency.py`.

    pytest evals/test_adapters_untested.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

import pytest  # noqa: E402

from agentfuse import (  # noqa: E402
    AgentEvent, CircuitBreakerMonitor, DirectiveKind, EventType, MonitorConfig, Tracer,
)
from agentfuse.adapters.openai_sdk import guarded_tool_loop  # noqa: E402
from agentfuse.adapters.langgraph import FuseCallbackHandler  # noqa: E402

GOAL = "Find the production database connection file."


# ------------------------------------------------- fake OpenAI SDK surface
class _Fn:
    def __init__(self, name, args):
        self.name, self.arguments = name, json.dumps(args)


class _ToolCall:
    def __init__(self, name, args, idx):
        self.id, self.function = f"call_{idx}", _Fn(name, args)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _Resp:
    def __init__(self, msg, tin=100, tout=20):
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"prompt_tokens": tin, "completion_tokens": tout})()


class FakeOpenAI:
    """Replays a scripted sequence of assistant turns."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []
        outer = self

        class _Completions:
            def create(self, model, messages, tools=None, **kw):
                outer.seen.append(list(messages))
                return outer.turns[min(len(outer.seen) - 1, len(outer.turns) - 1)]

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _loop_turns(n=10, tool="search_files"):
    return [_Resp(_Msg("looking", [_ToolCall(tool, {"pattern": "*.conn"}, i)]))
            for i in range(n)]


# ------------------------------------------------------------ openai_sdk
def test_openai_sdk_detects_a_loop_and_names_the_tool():
    """The regression this file exists for.

    Before the `state_hash` fix this trip came from `progress` and named no
    tool, because the loop detector could never form a single pair.
    """
    seen = []

    class Spy(Tracer):
        def trip(self, event, trip):
            seen.append((trip.detector, (trip.evidence or {}).get("tool")))
            super().trip(event, trip)

    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      drift_threshold=0.0, max_recoveries=99),
        tracer=Spy(None, False))
    guarded_tool_loop(FakeOpenAI(_loop_turns()), model="gpt-4.1", system_prompt=GOAL,
                      user_input="find it", tools=[],
                      tool_router=lambda n, a: "0 files matched",
                      max_turns=10, monitor=mon)

    assert seen, "an infinite identical tool loop produced no trip at all"
    detectors = {d for d, _ in seen}
    assert "loop" in detectors, (
        f"the loop detector never fired through this adapter; got {detectors}. "
        f"It is inert if it resets on the mere presence of event.state.")
    assert any(tool == "search_files" for d, tool in seen if d == "loop"), (
        "the loop trip did not name the offending tool, which is the one thing "
        "the loop detector contributes over the progress backstop")


def test_openai_sdk_injects_steering_into_the_conversation():
    """A directive nobody applies is not a recovery."""
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      drift_threshold=0.0, max_recoveries=99),
        tracer=Tracer(None, False))
    client = FakeOpenAI(_loop_turns())
    guarded_tool_loop(client, model="gpt-4.1", system_prompt=GOAL, user_input="go",
                      tools=[], tool_router=lambda n, a: "0 files matched",
                      max_turns=10, monitor=mon)
    steered = [m for turn in client.seen for m in turn
               if m.get("role") == "system" and "CIRCUIT BREAKER STEERING" in str(m.get("content"))]
    assert steered, "steering was produced but never reached the agent's messages"


def test_openai_sdk_reports_token_usage_to_the_monitor():
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    guarded_tool_loop(FakeOpenAI([_Resp(_Msg("done", None), tin=500, tout=75)]),
                      model="gpt-4.1", system_prompt=GOAL, user_input="go", tools=[],
                      tool_router=lambda n, a: "ok", max_turns=3, monitor=mon)
    assert mon.total_tokens == 575, "usage did not survive the adapter round trip"


def test_openai_sdk_completes_cleanly_when_the_agent_stops_calling_tools():
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    out = guarded_tool_loop(FakeOpenAI([_Resp(_Msg("all done", None))]),
                            model="gpt-4.1", system_prompt=GOAL, user_input="go",
                            tools=[], tool_router=lambda n, a: "ok", max_turns=3,
                            monitor=mon)
    assert out["status"] == "complete"


def test_openai_sdk_runs_the_tool_router_with_parsed_arguments():
    calls = []
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    turns = [_Resp(_Msg("go", [_ToolCall("search_files", {"pattern": "*.conn"}, 0)])),
             _Resp(_Msg("done", None))]
    guarded_tool_loop(FakeOpenAI(turns), model="gpt-4.1", system_prompt=GOAL,
                      user_input="go", tools=[],
                      tool_router=lambda n, a: calls.append((n, a)) or "ok",
                      max_turns=4, monitor=mon)
    assert calls == [("search_files", {"pattern": "*.conn"})], (
        "tool arguments did not survive JSON round-tripping through the adapter")


# ------------------------------------------------------------- langgraph
class _Gen:
    def __init__(self, text):
        self.text = text


class _LLMResult:
    def __init__(self, text, tin=80, tout=15):
        self.generations = [[_Gen(text)]]
        self.llm_output = {"token_usage": {"prompt_tokens": tin,
                                           "completion_tokens": tout}}


def test_langgraph_detects_a_loop_and_names_the_tool():
    seen = []

    class Spy(Tracer):
        def trip(self, event, trip):
            seen.append((trip.detector, (trip.evidence or {}).get("tool")))
            super().trip(event, trip)

    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      drift_threshold=0.0, max_recoveries=99),
        tracer=Spy(None, False))
    h = FuseCallbackHandler(original_goal=GOAL, monitor=mon)
    for _ in range(8):
        h.on_tool_start({"name": "search_files"}, '{"pattern": "*.conn"}')
        h.on_tool_end("0 files matched")

    assert seen, "an identical repeated tool call produced no trip"
    assert "loop" in {d for d, _ in seen}, (
        f"loop detector inert through the langgraph adapter; got {[d for d, _ in seen]}")


def test_langgraph_reports_token_usage():
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    h = FuseCallbackHandler(original_goal=GOAL, monitor=mon)
    h.on_llm_end(_LLMResult("thinking about the objective", tin=200, tout=40))
    assert mon.total_tokens == 240


def test_langgraph_supervisor_node_returns_a_partial_update():
    """LangGraph nodes return partial updates that a reducer merges.

    My first version of this test asserted the node echoed the whole state back
    and called the adapter buggy when it didn't. The adapter was right and the
    test was wrong: `{}` means "no change" in LangGraph, and returning the full
    state would be the unusual thing. Kept as a note because correcting working
    code to satisfy a mistaken test is the easiest way to break something.
    """
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    h = FuseCallbackHandler(original_goal=GOAL, monitor=mon)

    assert h.supervisor_node({"messages": ["a"], "counter": 3}) == {}, \
        "no pending steering should mean no state update at all"

    h.pending_steering = "Stop calling search_files."
    out = h.supervisor_node({"messages": ["a"], "counter": 3})
    assert list(out) == ["messages"], "only the touched key should be returned"
    assert len(out["messages"]) == 2 and "CIRCUIT BREAKER STEERING" in out["messages"][-1]["content"]
    assert h.pending_steering is None, "steering must be consumed exactly once"


def test_langgraph_finish_returns_a_summary():
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    h = FuseCallbackHandler(original_goal=GOAL, monitor=mon)
    h.on_tool_start({"name": "t"}, "{}")
    h.on_tool_end("ok")
    out = h.finish()
    assert "status" in out and "total_tokens" in out


# --------------------------------------------------- the contract, directly
def test_presence_of_state_is_not_progress():
    """The root cause, stated as a contract test rather than a scenario.

    An adapter that cannot tell whether a call achieved anything will attach
    `state` to every result. A detector that treats that as progress is inert.
    Identical state means nothing changed.
    """
    from agentfuse.detectors.loop import LoopDetector

    det = LoopDetector(threshold=3)
    trip = None
    for i in range(1, 8):
        det.inspect(AgentEvent(type=EventType.TOOL_CALL, step=i, tool_name="t",
                               tool_args={"q": 1}), [])
        trip = trip or det.inspect(AgentEvent(
            type=EventType.TOOL_RESULT, step=i, tool_name="t", text="same",
            state={"last_tool": "t", "result": "same"}), [])
    assert trip is not None, (
        "an unchanging state payload was treated as progress, which makes this "
        "detector inert behind every adapter that emits state unconditionally")


def test_a_genuine_state_change_still_clears_the_counters():
    """The other half of the contract: real progress must still reset."""
    from agentfuse.detectors.loop import LoopDetector

    det = LoopDetector(threshold=3)
    for i in range(1, 8):
        det.inspect(AgentEvent(type=EventType.TOOL_CALL, step=i, tool_name="t",
                               tool_args={"q": 1}), [])
        assert det.inspect(AgentEvent(
            type=EventType.TOOL_RESULT, step=i, tool_name="t", text="same",
            state={"advanced_at": i}), []) is None, (
            "a changing state hash is genuine progress and must not trip")
