"""Tests for verified, memoried recovery (Phase 2).

Phase 1 measured recovery and found it single-shot: the breaker produced an
instruction, injected it, and never checked whether it helped. When the agent
stayed stuck the same trip fired again, the same reasoning ran against the same
snapshot, and substantially the same advice came back — three attempts, one idea.

Phase 2 adds the two pieces that fix it: a *verification* step that decides
whether a steer actually worked, and a *memory* so the next attempt can climb to
a genuinely different intervention instead of rephrasing the last one.

    pytest evals/test_recovery_memory.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse import (  # noqa: E402
    AgentEvent, EventType, CircuitBreakerMonitor, MonitorConfig, Tracer,
)
from agentfuse.events import ExecutionSnapshot  # noqa: E402
from agentfuse.memory import JSONMemory, RecoveryRecord, failure_signature  # noqa: E402
from agentfuse.recovery import RecoveryEngine  # noqa: E402
from agentfuse.strategies import LADDER, ESCALATE, next_strategy  # noqa: E402

GOAL = "Rotate the production database credential."


def _snapshot(tool: str = "search_files", detector: str = "loop", **evidence):
    return ExecutionSnapshot(
        step=3, original_goal=GOAL, current_goal=None, total_tokens=10,
        total_cost_usd=0.0, route_history=["agent"], recent_events=[],
        trip_reason="looping", trip_detector=detector,
        trip_evidence={"tool": tool, **evidence})


# ------------------------------------------------------------------ memory
def test_signature_ignores_incidental_detector_metadata():
    """One logical failure must not split across buckets.

    The loop detector reports `signal` as either 'call+result' or 'call-only
    fallback' for the same stuck tool. Keying on it made the ladder restart from
    the bottom whenever the detector fired down its other path.
    """
    a = failure_signature("loop", "search_files", {"signal": "call+result"})
    b = failure_signature("loop", "search_files", {"signal": "call-only fallback"})
    assert a == b


def test_signature_separates_different_tools():
    assert failure_signature("loop", "search_files", {}) != \
           failure_signature("loop", "fetch_invoice", {})


def test_failed_strategies_scan_all_history_not_a_window():
    """A recency window let failed rungs age out and be retried. Real bug."""
    mem = JSONMemory()
    rid = mem.remember(RecoveryRecord(signature="sig", detector="loop", goal=GOAL,
                                      strategy="re-anchor", instruction="x"))
    mem.mark_outcome(rid, worked=False)
    for i in range(50):  # bury it well past any plausible window
        mem.remember(RecoveryRecord(signature="sig", detector="loop", goal=GOAL,
                                    strategy=f"filler-{i}", instruction="y"))
    assert "re-anchor" in mem.failed_strategies("sig")


def test_memory_persists_across_instances(tmp_path):
    path = tmp_path / "memory.jsonl"
    mem = JSONMemory(path=str(path))
    rid = mem.remember(RecoveryRecord(signature="sig", detector="loop", goal=GOAL,
                                      strategy="re-anchor", instruction="x"))
    mem.mark_outcome(rid, worked=False)

    reopened = JSONMemory(path=str(path))
    assert "re-anchor" in reopened.failed_strategies("sig")


# ------------------------------------------------------------------ ladder
def test_ladder_is_ordered_from_lightest_to_heaviest():
    assert LADDER[0] == "re-anchor"
    assert LADDER[-1] == ESCALATE


def test_next_strategy_skips_only_what_failed():
    assert next_strategy(set()) == "re-anchor"
    assert next_strategy({"re-anchor"}) == "alternate-action"
    assert next_strategy(set(LADDER[:-1])) == ESCALATE


def test_critical_severity_skips_straight_to_escalation():
    """A budget ceiling is not a reasoning problem; steering cannot fix it."""
    assert next_strategy(set(), severity="critical") == ESCALATE


# -------------------------------------------------------- engine + memory
def test_engine_climbs_the_ladder_when_steers_fail():
    engine = RecoveryEngine(backend="mock")
    seen = []
    for _ in range(len(LADDER)):
        path = engine.recover(_snapshot())
        seen.append(path.strategy)
        engine.verify(path, worked=False)

    assert seen[:4] == LADDER[:4], f"ladder did not climb in order: {seen}"
    assert seen[-1] == ESCALATE
    assert len(set(seen)) == len(seen), f"a rung was retried after failing: {seen}"


def test_alternate_action_names_the_repeated_arguments_not_just_the_tool():
    """The tool must not be banned wholesale for a call it never made.

    LoopDetector's primary signal is (tool, args, result) repeating -- the
    same CALL, not any use of the tool. REPORT.md section 3.25: without this,
    an agent that genuinely needs the same tool again with a different
    target was told not to use it at all. Reach alternate-action by failing
    re-anchor first, matching how the engine actually gets there.
    """
    engine = RecoveryEngine(backend="mock")
    snap = _snapshot(args={"dir": "./config", "pattern": "*.json"})
    engine.verify(engine.recover(snap), worked=False)   # re-anchor fails
    path = engine.recover(snap)                          # -> alternate-action

    assert path.strategy == "alternate-action"
    assert "./config" in path.instruction, (
        "the specific looping arguments must appear in the instruction")
    assert "materially different arguments" in path.instruction, (
        "the tool must remain usable with a genuinely different call")


def test_alternate_action_degrades_gracefully_with_no_known_args():
    """Not every trip carries args (progress/drift/rate trips may not)."""
    engine = RecoveryEngine(backend="mock")
    snap = _snapshot()  # no args= passed
    engine.verify(engine.recover(snap), worked=False)
    path = engine.recover(snap)
    assert path.strategy == "alternate-action"
    assert "Do NOT repeat `search_files`" in path.instruction


def test_engine_does_not_climb_when_the_steer_worked():
    """A rung that worked stays available; only failure rules one out."""
    engine = RecoveryEngine(backend="mock")
    first = engine.recover(_snapshot())
    engine.verify(first, worked=True)
    second = engine.recover(_snapshot())
    assert second.strategy == first.strategy


def test_escalations_are_not_recorded():
    """'We gave up' is not a correction that can succeed or fail."""
    engine = RecoveryEngine(backend="mock")
    for _ in range(len(LADDER)):
        engine.verify(engine.recover(_snapshot()), worked=False)
    strategies = [r.strategy for r in engine.memory._records]
    assert ESCALATE not in strategies


def test_failed_instructions_are_surfaced_for_the_next_attempt():
    engine = RecoveryEngine(backend="mock")
    first = engine.recover(_snapshot())
    engine.verify(first, worked=False)
    sig = failure_signature("loop", "search_files", {})
    assert first.instruction in engine.memory.failed_instructions(sig)


def test_memory_fault_never_breaks_the_run():
    """The supervisor must not take down the run it is supervising."""
    class Broken:
        def failed_strategies(self, sig): raise RuntimeError("memory down")
        def failed_instructions(self, sig): raise RuntimeError("memory down")
        def remember(self, rec): raise RuntimeError("memory down")
        def mark_outcome(self, rid, worked): raise RuntimeError("memory down")

    engine = RecoveryEngine(backend="mock", memory=Broken())
    path = engine.recover(_snapshot())
    assert path.instruction
    engine.verify(path, worked=True)  # must not raise


# ------------------------------------------------------------ verification
def test_monitor_marks_a_steer_failed_when_the_agent_stays_stuck():
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      max_recoveries=8),
        tracer=Tracer(None, False))
    for i in range(1, 30):
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=i, tool_name="search_files",
                               tool_args={"d": "./config"}, tokens_in=100, tokens_out=20))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=i,
                               tool_name="search_files", text="0 files matched"))

    assert mon.steers_that_failed >= 3, "repeated trips should mark steers as failed"
    tried = [r.strategy for r in mon.recovery.memory._records]
    assert len(tried) == len(set(tried)), f"a rung was retried: {tried}"


def test_monitor_marks_a_steer_worked_when_progress_follows():
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3),
        tracer=Tracer(None, False))
    for i in range(1, 5):  # trip it
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=i, tool_name="search_files",
                               tool_args={"d": "./config"}))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=i,
                               tool_name="search_files", text="0 files matched"))
    # the agent takes the advice and genuinely advances
    mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=6, tool_name="secret_get",
                           text="rotated", state={"progress": True}))

    assert mon.steers_that_worked >= 1
    assert any(r.worked is True for r in mon.recovery.memory._records)


def test_ignoring_the_steer_must_not_count_as_working():
    """THE bug. A steer the agent ignores must not register as successful.

    The production adapter (adapters/openai_sdk.py) sets `state=` on EVERY
    tool result, unconditionally -- it is not gated on genuine progress. The
    old check was `event.state is not None`, which that adapter satisfies on
    every single call, so an agent that repeats the exact call it was steered
    away from -- byte-for-byte, same tool, same args, same result -- still
    counted as a *worked* steer, because SOME state payload existed. It
    always does. Reproduced directly against the real monitor before this
    fix landed: `steers_that_worked` went from 0 to 1 on the repeat.

    The fix reuses SeenStateTracker (the same bounded-window novelty check
    NoProgressDetector and LoopDetector already use) so "state present" is
    replaced with "state genuinely new," which a byte-for-byte repeat is not.
    """
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3),
        tracer=Tracer(None, False))
    result_state = {"last_tool": "list_secrets",
                     "result": "prod/db/primary, prod/db/replica"}
    for i in range(1, 5):  # trip it
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=i,
                               tool_name="list_secrets", tool_args={}))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=i,
                               tool_name="list_secrets", text="prod/db/primary, prod/db/replica",
                               state=dict(result_state)))
    assert mon.recovery_count >= 1, "setup failed to trip the breaker"

    # The agent IGNORES the steer: same tool, same args, same result -- a
    # state payload exists (the adapter always sets one), but nothing is new.
    mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=5,
                           tool_name="list_secrets", tool_args={}))
    mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=5,
                           tool_name="list_secrets", text="prod/db/primary, prod/db/replica",
                           state=dict(result_state)))

    assert mon.steers_that_worked == 0, (
        "a byte-for-byte repeat of the pre-steer call must not verify as worked")


def test_max_recoveries_can_reach_every_steerable_rung():
    """A cap below the ladder depth makes the upper rungs dead code.

    This was shipped: max_recoveries defaulted to 3 against a 4-rung ladder, so
    an agent that would have responded to `decompose` never got the chance.
    Measured cost was 20 points of recovery rate.
    """
    from agentfuse.monitor import MonitorConfig
    from agentfuse.strategies import STEERABLE
    assert MonitorConfig(original_goal="x").max_recoveries >= len(STEERABLE)
