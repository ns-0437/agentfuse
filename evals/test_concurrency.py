"""Concurrency safety — the supervisor must not corrupt or crash the run.

This project is pitched at long-horizon, multi-agent, parallel-tool-call work,
and until 2026-08-13 the core had no locks anywhere while ``observe()`` mutated
shared state. That was not a theoretical gap. Driving one monitor from two
threads, with the GIL switch interval lowered to expose the window, produced:

  * **mis-attributed trips** — a trip naming a tool the agent never called,
    which then becomes a steering instruction telling it to stop using that
    tool; and
  * ``RuntimeError: deque mutated during iteration`` raised out of a detector.

The second is the serious one. AgentFuse's adapters raise into the agent's call
stack, so an exception from a detector takes down the run the supervisor exists
to protect — the exact failure ``test_recovery_engine_survives_provider_failure``
was written to prevent, arriving through a different door.

The first test below needs no threads at all: the call/result pairing bug is
fully deterministic once an agent issues **parallel tool calls**, which the
OpenAI Agents SDK does by default. Threads only widened the window; they were
never the cause.

    pytest evals/test_concurrency.py -v
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse import (  # noqa: E402
    AgentEvent, CircuitBreakerMonitor, DirectiveKind, EventType, MonitorConfig,
    Tracer,
)
from agentfuse.detectors.loop import LoopDetector, _lane  # noqa: E402
from agentfuse.memory import JSONMemory, RecoveryRecord  # noqa: E402

GOAL = "Reconcile the outstanding invoices against the ledger."


# ------------------------------------------------- deterministic: parallel calls
def test_parallel_tool_calls_are_paired_with_their_own_results():
    """call(a), call(b), result(a), result(b) — no threads, still interleaved.

    A single `_pending` slot assumes calls and results strictly alternate. They
    do not: an agent that issues two tool calls in one turn produces exactly
    this order, and the old code paired A's OUTCOME with B's SIGNATURE.
    """
    det = LoopDetector(threshold=3)
    trips = []
    for _ in range(6):
        det.inspect(AgentEvent(type=EventType.TOOL_CALL, step=1, node="agent",
                               tool_name="search_files", tool_args={"d": "./x"}), [])
        det.inspect(AgentEvent(type=EventType.TOOL_CALL, step=1, node="agent",
                               tool_name="fetch_invoice", tool_args={"id": 7}), [])
        for tool, text in (("search_files", "0 files matched"),
                           ("fetch_invoice", "invoice 7 unchanged")):
            t = det.inspect(AgentEvent(type=EventType.TOOL_RESULT, step=1,
                                       node="agent", tool_name=tool, text=text), [])
            if t is not None:
                trips.append((tool, t.evidence.get("tool")))

    assert trips, "interleaved parallel calls should still detect the loop"
    for actual_tool, named_tool in trips:
        assert named_tool == actual_tool, (
            f"trip for {actual_tool!r} named {named_tool!r} — a steer built from "
            f"this tells the agent to stop calling a tool it never called")


def test_lane_prefers_an_explicit_call_id():
    """When the runtime gives us an id, ambiguity disappears entirely."""
    a = AgentEvent(type=EventType.TOOL_CALL, step=1, node="n", tool_name="t",
                   meta={"call_id": "call_abc"})
    b = AgentEvent(type=EventType.TOOL_RESULT, step=1, node="n", tool_name="t",
                   meta={"call_id": "call_xyz"})
    assert _lane(a) != _lane(b)
    assert _lane(a) == "call_abc"


def test_two_concurrent_calls_to_the_same_tool_do_not_crash():
    """The one genuinely ambiguous case must degrade, not explode.

    Without a call id from the runtime, two in-flight calls to the same tool on
    the same node cannot be told apart. The later call wins. That is a limit of
    the event stream; what matters is that it stays quiet instead of raising.
    """
    det = LoopDetector(threshold=3)
    for _ in range(4):
        for _ in range(2):
            det.inspect(AgentEvent(type=EventType.TOOL_CALL, step=1, node="a",
                                   tool_name="probe", tool_args={"q": 1}), [])
        for _ in range(2):
            det.inspect(AgentEvent(type=EventType.TOOL_RESULT, step=1, node="a",
                                   tool_name="probe", text="same"), [])


# --------------------------------------------------------------- under threads
def _hammer(mon, tid, events, errors):
    try:
        for i in range(events):
            mon.observe(AgentEvent(type=EventType.LLM_CALL, step=i, node=f"a{tid}",
                                   text="working", tokens_in=10, tokens_out=5))
            mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=i, node=f"a{tid}",
                                   tool_name=f"tool{tid}", tool_args={"q": tid}))
            mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=i, node=f"a{tid}",
                                   tool_name=f"tool{tid}", text="no change"))
    except Exception as e:                       # noqa: BLE001 - that is the point
        errors.append(e)


def test_concurrent_observe_never_raises_into_the_agent():
    """A supervisor that crashes the run it supervises is worse than none."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                  # expose the window on purpose
    try:
        mon = CircuitBreakerMonitor(
            MonitorConfig(original_goal=GOAL, echo=False, max_recoveries=99),
            tracer=Tracer(None, False))
        errors: list[Exception] = []
        threads = [threading.Thread(target=_hammer, args=(mon, t, 120, errors))
                   for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"observe() raised under concurrency: {errors[:3]}"
        assert len(mon.history) == 6 * 120 * 3, "events were lost"
    finally:
        sys.setswitchinterval(old)


def test_token_accounting_survives_concurrency():
    """`total += x` is three bytecodes; under contention updates get lost."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        mon = CircuitBreakerMonitor(
            MonitorConfig(original_goal=GOAL, echo=False, max_recoveries=99),
            tracer=Tracer(None, False))
        errors: list[Exception] = []
        threads = [threading.Thread(target=_hammer, args=(mon, t, 100, errors))
                   for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 15 tokens per llm_call, one per iteration, per thread.
        assert mon.total_tokens == 6 * 100 * 15, (
            f"lost token updates: {mon.total_tokens} != {6 * 100 * 15}")
    finally:
        sys.setswitchinterval(old)


def test_shared_memory_does_not_lose_or_corrupt_records():
    """JSONMemory is meant to be shared across runs, so it needs its own lock.

    `_flush()` rewrites the whole file, so two concurrent writers can truncate
    each other; every read iterates `_records` while another thread appends.
    """
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        mem = JSONMemory()
        errors: list[Exception] = []

        def writer(tid):
            try:
                for i in range(150):
                    rid = mem.remember(RecoveryRecord(
                        signature=f"sig{tid}", detector="loop", goal=GOAL,
                        strategy=f"s{i}", instruction="x"))
                    mem.mark_outcome(rid, worked=False)
                    mem.failed_strategies(f"sig{tid}")
            except Exception as e:               # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"memory raised under concurrency: {errors[:3]}"
        assert len(mem) == 5 * 150, f"lost records: {len(mem)}"
    finally:
        sys.setswitchinterval(old)


# ------------------------------------------------------------------- asyncio
# Every test above drives the monitor from THREADS. The AgentKit adapter's hooks
# are `async def` and call the synchronous `observe()` directly, so the path that
# ships is the one that was never exercised here. These cover it.

def test_concurrent_async_agents_keep_their_verdicts_separate():
    """One monitor per agent run is the supported model — this proves it holds
    when the runs are coroutines interleaving on a single event loop.

    A shared event loop is the case where cross-contamination would appear as a
    detector on agent A tripping because of agent B's calls, and it would look
    like a false positive rather than a concurrency bug.
    """
    import asyncio

    async def drive(tool: str, repeats: int) -> list:
        mon = CircuitBreakerMonitor(
            MonitorConfig(original_goal=f"use {tool}", echo=False,
                          loop_threshold=3, max_recoveries=0),
            tracer=Tracer(None, False))
        trips = []
        for step in range(repeats):
            # The production event shape: llm_call -> tool_call -> tool_result.
            events = [
                AgentEvent(type=EventType.LLM_CALL, step=step + 1,
                           text=f"calling {tool}"),
                AgentEvent(type=EventType.TOOL_CALL, step=step + 1,
                           tool_name=tool, tool_args={"q": "same"},
                           meta={"call_id": f"{tool}-{step}"}),
                AgentEvent(type=EventType.TOOL_RESULT, step=step + 1,
                           tool_name=tool, text="same result",
                           meta={"call_id": f"{tool}-{step}"}),
            ]
            for ev in events:
                # observe() returns a directive on EVERY event, and CONTINUE
                # means "nothing wrong". Treating any non-None return as a trip
                # counts healthy events as failures -- which is exactly what the
                # first version of this test did, and it reported a leak that
                # was not there. A control run of the healthy agent ALONE showed
                # the same three "trips", which is what gave it away.
                directive = mon.observe(ev)
                if directive is not None and directive.kind is not DirectiveKind.CONTINUE:
                    trips.append(tool)
                await asyncio.sleep(0)      # yield: let the other agent interleave
        return trips

    async def main():
        return await asyncio.gather(
            drive("search_files", 6),        # loops -> should trip
            drive("healthy_tool", 1),        # one call -> must stay silent
        )

    looping, healthy = asyncio.run(main())
    assert looping, "the looping agent did not trip while interleaved"
    assert not healthy, (
        "a healthy agent tripped because another coroutine was looping — "
        "verdicts are leaking across concurrent runs")


def test_observe_from_an_event_loop_never_raises_into_the_agent():
    """`observe()` is called from `async def` hooks with no try/except around it
    in the adapter, so anything it raises propagates into the agent's own
    lifecycle and kills the run the supervisor exists to protect."""
    import asyncio

    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal="g", echo=False, max_recoveries=0),
        tracer=Tracer(None, False))

    async def hostile(i: int):
        # Deliberately malformed-ish events: missing names, empty args, huge text.
        events = [
            AgentEvent(type=EventType.LLM_CALL, step=i, text="x" * 5000),
            AgentEvent(type=EventType.TOOL_CALL, step=i, tool_name=None,
                       tool_args=None),
            AgentEvent(type=EventType.TOOL_RESULT, step=i, tool_name=None,
                       text=None),
        ]
        for ev in events:
            mon.observe(ev)
            await asyncio.sleep(0)

    async def main():
        await asyncio.gather(*(hostile(i) for i in range(8)))

    asyncio.run(main())          # must simply not raise


# ------------------------------------------------- sharing one monitor (misuse)
def _agent_events(step, agent, tool, text):
    """Events tagged with an explicit agent identity."""
    meta = {"call_id": f"{agent}-{tool}-{step}", "agent_id": agent}
    return [
        AgentEvent(type=EventType.LLM_CALL, step=step, node=agent, text=text,
                   meta=meta),
        AgentEvent(type=EventType.TOOL_CALL, step=step, node=agent,
                   tool_name=tool, tool_args={"n": step}, meta=meta),
    ]


def test_sharing_one_monitor_across_agents_halts_healthy_runs():
    """Pins the measured cost of the unsupported configuration.

    REPORT once said locks "made it safe but not meaningful". That understates
    it: measured, two agents on different goals driving one monitor produced
    multiple spurious PAUSE directives while both were healthy on their own
    terms. `original_goal` is singular, so every drift probe scores one agent's
    reasoning against the other's objective.

    Pinned so the claim stays evidence-backed, and so anyone who later tries to
    support sharing has a test that tells them what they must fix.
    """
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal="Rotate the production database credential.",
                      echo=False, loop_threshold=3, max_recoveries=0),
        tracer=Tracer(None, False))

    pauses = 0
    for step in range(1, 5):
        for ev in (_agent_events(step, "agent_a", "read_secret",
                                 "reading the production credential")
                   + _agent_events(step, "agent_b", "send_email",
                                   "drafting the marketing newsletter")):
            d = mon.observe(ev)
            if d is not None and d.kind is not DirectiveKind.CONTINUE:
                pauses += 1
    assert pauses > 0, (
        "sharing a monitor no longer halts healthy agents — if that is now "
        "genuinely supported, update REPORT section 8 and delete this test")


def test_a_shared_monitor_warns_once_when_it_can_tell():
    """Silent misbehaviour in a configuration people will try is the bug class
    this project keeps cataloguing. It cannot be prevented, so it is announced."""
    import warnings as _w

    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal="g", echo=False, max_recoveries=0),
        tracer=Tracer(None, False))

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        for ev in _agent_events(1, "agent_a", "t", "a"):
            mon.observe(ev)
        for ev in _agent_events(1, "agent_b", "t", "b"):
            mon.observe(ev)
        for ev in _agent_events(2, "agent_b", "t", "b"):
            mon.observe(ev)

    shared = [w for w in caught if "more than" in str(w.message)]
    assert len(shared) == 1, f"expected exactly one warning, got {len(shared)}"


def test_a_single_agent_never_triggers_the_shared_warning():
    """LangGraph runs walk many nodes. Warning on that would be noise on every
    ordinary multi-node run, which is why node is not the discriminator."""
    import warnings as _w

    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal="g", echo=False, max_recoveries=0),
        tracer=Tracer(None, False))

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        for step, node in enumerate(("plan", "search", "write", "verify"), start=1):
            mon.observe(AgentEvent(type=EventType.LLM_CALL, step=step, node=node,
                                   text="working", meta={"agent_id": "one-agent"}))

    assert not [w for w in caught if "more than" in str(w.message)]
