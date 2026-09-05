"""Durable run state — does a restart still enforce what it was enforcing?

The supervisor was in-memory only, which is a safety hole rather than an
inconvenience. An agent under a 500,000-token ceiling that dies at 480,000 and is
restarted comes back with its budget at zero: the restart hands a runaway run a
brand new allowance, and the guard whose whole job is bounding unattended spend
has been rearmed instead of enforced.

So these tests are written against *behaviour after a restart*, not against the
bytes in the file. A serialiser can round-trip perfectly and still lose the thing
that makes the breaker fire; the only question worth asking is whether a resumed
supervisor still trips when it should.

    pytest evals/test_checkpoint.py -v
"""

from __future__ import annotations

import time
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
from agentfuse.checkpoint import (  # noqa: E402
    SQLiteCheckpointStore, decode, encode, load_state_dict, state_dict,
)
from agentfuse.detectors import (  # noqa: E402
    LoopDetector, NoProgressDetector, RateOfProgressDetector, SpendDetector,
)

GOAL = "Rotate the production database credential."


def _mon(path, run_id="r1", **kw):
    return CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, checkpoint_path=str(path),
                      run_id=run_id, checkpoint_every=1, **kw),
        tracer=Tracer(None, False))


def _tool_cycle(mon, step, tool="search_files", result="0 files matched", tokens=0):
    mon.observe(AgentEvent(type=EventType.LLM_CALL, step=step, node="agent",
                           text="thinking", tokens_in=tokens, tokens_out=0))
    mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, node="agent",
                           tool_name=tool, tool_args={"d": "./config"}))
    return mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, node="agent",
                                  tool_name=tool, text=result))


# ------------------------------------------------------- the safety argument
def test_a_restart_does_not_hand_a_runaway_run_a_fresh_budget(tmp_path):
    """The reason this module exists.

    Burn most of the ceiling, lose the process, come back. The resumed
    supervisor must still know the budget is nearly gone.
    """
    db = tmp_path / "runs.db"
    mon = _mon(db, max_tokens=10_000)
    for step in range(1, 10):
        mon.observe(AgentEvent(type=EventType.LLM_CALL, step=step, node="agent",
                               text="work", tokens_in=1000, tokens_out=0))
    assert mon.total_tokens == 9000

    resumed = _mon(db)                       # a brand-new process
    assert resumed.total_tokens == 0, "sanity: a fresh monitor starts at zero"
    assert resumed.restore() is True
    assert resumed.total_tokens == 9000, (
        "a restart reset the spend counter — the ceiling has been rearmed, "
        "not enforced")


def test_the_spend_ceiling_still_fires_after_a_restart(tmp_path):
    """Not just the number: the breaker must actually trip on it."""
    db = tmp_path / "runs.db"
    mon = _mon(db, max_tokens=10_000)
    for step in range(1, 10):
        mon.observe(AgentEvent(type=EventType.LLM_CALL, step=step, node="agent",
                               text="work", tokens_in=1000, tokens_out=0))

    resumed = _mon(db, max_tokens=10_000)
    resumed.restore()
    d = resumed.observe(AgentEvent(type=EventType.LLM_CALL, step=10, node="agent",
                                   text="work", tokens_in=2000, tokens_out=0))
    assert d.kind is not DirectiveKind.CONTINUE, (
        "the resumed run blew through its ceiling without tripping")


def test_a_loop_spanning_a_restart_is_still_caught(tmp_path):
    """Two repeats before the crash, the third after. It is one loop.

    `drift_threshold=0.0` disables the drift detector for this test rather than
    isolating the loop detector by hand. Without it the placeholder reasoning
    text reads as off-goal, drift trips first, and `_handle_trip` resets EVERY
    detector — so the loop counter was being cleared by an unrelated detector
    and the test measured detector interaction instead of persistence.
    """
    db = tmp_path / "runs.db"
    mon = _mon(db, loop_threshold=3, drift_threshold=0.0)
    for step in (1, 2):
        assert _tool_cycle(mon, step).kind is DirectiveKind.CONTINUE

    resumed = _mon(db, loop_threshold=3, drift_threshold=0.0)
    resumed.restore()
    d = _tool_cycle(resumed, 3)
    assert d.kind is not DirectiveKind.CONTINUE, (
        "the loop counter reset across the restart, so the agent got to start "
        "its loop over")


def test_the_learned_calibration_baseline_survives(tmp_path):
    """Losing it drops the run back into the blind pre-baseline window."""
    db = tmp_path / "runs.db"
    mon = _mon(db)
    for step in range(1, 6):
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, node="agent",
                               tool_name="t", tool_args={"i": step}))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, node="agent",
                               tool_name="t", text="ok", state={"at": step}))
    assert mon.calibrator.baseline.samples >= 1

    resumed = _mon(db)
    resumed.restore()
    assert resumed.calibrator.baseline.samples == mon.calibrator.baseline.samples
    assert resumed.calibrator.baseline.ready is mon.calibrator.baseline.ready


# --------------------------------------------------- recovery ladder memory
def _trip_loop(mon, step_start):
    """Repeat one identical (tool, args, result) triple enough times to trip
    LoopDetector's default threshold. Returns the step counter after."""
    step = step_start
    for _ in range(4):
        step += 1
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, node="agent",
                               tool_name="list_secrets", tool_args={}))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, node="agent",
                               tool_name="list_secrets", text="a,b",
                               state={"last_tool": "list_secrets", "result": "a,b"}))
    return step


def test_a_restart_does_not_reset_the_ladder_to_rung_one(tmp_path):
    """The same failure shape checkpointing was built to fix (this module's own
    docstring: a restart rearming a guard instead of enforcing it), found in
    the recovery ladder's own memory instead of the spend ceiling.

    A default-constructed RecoveryEngine's memory is in-process only, so a
    monitor with checkpoint_path set durably persisted tokens/detectors/
    calibration but silently forgot every already-tried steering rung on
    restart -- an agent that ignores corrections across a crash-restart cycle
    would get the SAME already-failed rung again, not the next one.
    """
    db = tmp_path / "runs.db"
    mon = _mon(db, loop_threshold=3)
    step = _trip_loop(mon, 0)                          # trips, steers: re-anchor
    mon.recovery.verify(mon.recovery.memory._records[-1], worked=False)
    step = _trip_loop(mon, step)                       # climbs to: alternate-action
    assert [r.strategy for r in mon.recovery.memory._records] == [
        "re-anchor", "alternate-action"]
    mon.checkpoint()

    # A monitor pointed at a checkpoint_path that has never been used before
    # starts with genuinely empty recovery memory -- confirms the persistence
    # is scoped to the path, not always-on regardless of history.
    fresh = _mon(tmp_path / "unrelated.db", loop_threshold=3)
    assert fresh.recovery.memory._records == []

    resumed = _mon(db, loop_threshold=3)                # a brand-new process
    # JSONMemory loads from its file at construction time, independent of
    # Monitor.restore() -- it is a plain file, not part of the SQLite
    # checkpoint blob, so the ladder history is already there.
    assert [r.strategy for r in resumed.recovery.memory._records] == [
        "re-anchor", "alternate-action"], (
        "the ladder's climb history did not survive the restart -- a rung "
        "already demonstrated to fail would be offered again")
    assert resumed.restore() is True


def test_the_ladder_keeps_climbing_across_a_restart_instead_of_restarting_at_rung_one(tmp_path):
    """Not just the record: the NEXT trip after a restart must pick the next
    rung, not repeat one already marked failed before the crash."""
    db = tmp_path / "runs.db"
    mon = _mon(db, loop_threshold=3)
    step = _trip_loop(mon, 0)
    mon.recovery.verify(mon.recovery.memory._records[-1], worked=False)  # re-anchor failed
    mon.checkpoint()

    resumed = _mon(db, loop_threshold=3)
    resumed.restore()
    _trip_loop(resumed, step)  # same failure re-trips post-restart

    strategies = [r.strategy for r in resumed.recovery.memory._records]
    assert strategies == ["re-anchor", "alternate-action"], (
        f"expected the ladder to climb past the pre-restart failure, got {strategies}")


def test_recovery_memory_persistence_is_opt_in_via_checkpoint_path(tmp_path):
    """No checkpoint_path, no durability requested -- must not change the
    zero-config default (in-process memory, no file written)."""
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3),
        tracer=Tracer(None, False))
    assert mon.recovery.memory.path is None


# ------------------------------------------------ pending steer verification
# The narrower gap the recovery-memory fix above left open on purpose (see its
# own commit message): a steer awaiting a verdict at the exact moment of a
# crash. Losing `_pending_steer` on restart does not just drop a tuple -- it
# means `_verify_pending` silently no-ops on every event afterwards, so the
# steer's outcome is NEVER recorded. The memory record sits at worked=None
# forever, and a rung already offered could be offered again believing it
# untried, which is the specific failure the recovery-memory fix exists to
# prevent -- reopened one step earlier in the same lifecycle.
def test_a_pending_steer_survives_a_restart_and_gets_verified(tmp_path):
    """Behavioural, not structural: the steer's OUTCOME must still get
    recorded after the restart, not merely the tuple that was saved."""
    db = tmp_path / "runs.db"
    mon = _mon(db, loop_threshold=3)
    step = _trip_loop(mon, 0)
    record_id = mon._pending_steer[0].record_id
    mon.checkpoint()

    resumed = _mon(db, loop_threshold=3)
    assert resumed.restore() is True
    assert resumed._pending_steer is not None, (
        "the steer awaiting a verdict was lost across the restart")
    assert resumed._pending_steer[0].record_id == record_id

    # A genuine advance after the restart -- this must verify the PRE-restart
    # steer, not silently no-op because _pending_steer came back empty.
    step += 1
    resumed.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, node="agent",
                               tool_name="read_secret", tool_args={"name": "prod/db/primary"}))
    resumed.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, node="agent",
                               tool_name="read_secret", text="prod/db/primary -> ok",
                               state={"last_tool": "read_secret", "result": "ok"}))

    rec = next((r for r in resumed.recovery.memory._records if r.record_id == record_id), None)
    assert rec is not None
    assert rec.worked is True, (
        f"expected worked=True, got {rec.worked!r} -- the steer's outcome was "
        f"never recorded, so this rung would look untried on the next trip")
    assert resumed._pending_steer is None


def test_a_pending_steer_that_times_out_after_restart_is_marked_failed(tmp_path):
    """The other branch of _verify_pending: no advance within verify_window
    must still resolve to worked=False post-restart, not hang forever."""
    db = tmp_path / "runs.db"
    mon = _mon(db, loop_threshold=3, verify_window=2)
    step = _trip_loop(mon, 0)
    record_id = mon._pending_steer[0].record_id
    mon.checkpoint()

    resumed = _mon(db, loop_threshold=3, verify_window=2)
    resumed.restore()

    # No progress, and enough steps pass to exceed verify_window. Varying the
    # args each step is deliberate: identical (tool, args, result) 3x would
    # trip LoopDetector again and manufacture a SECOND pending steer, which
    # would mask whether the original one actually timed out.
    for i in range(3):
        step += 1
        resumed.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, node="agent",
                                   tool_name="idle", tool_args={"i": i}))
        resumed.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, node="agent",
                                   tool_name="idle", text=f"nothing happened ({i})"))

    rec = next((r for r in resumed.recovery.memory._records if r.record_id == record_id), None)
    assert rec is not None
    assert rec.worked is False, (
        f"expected worked=False after the verify window elapsed, got {rec.worked!r}")
    assert resumed._pending_steer is None


def test_a_restart_with_nothing_pending_still_restores_cleanly(tmp_path):
    """The common case: no trip happened before the checkpoint, so there is
    nothing to persist. Must not crash and must not fabricate a pending steer."""
    db = tmp_path / "runs.db"
    mon = _mon(db)
    mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=1, node="agent",
                           tool_name="list_secrets", tool_args={}))
    mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=1, node="agent",
                           tool_name="list_secrets", text="a,b"))
    assert mon._pending_steer is None
    mon.checkpoint()

    resumed = _mon(db)
    assert resumed.restore() is True
    assert resumed._pending_steer is None


def test_an_older_checkpoint_with_no_pending_steer_key_still_restores(tmp_path):
    """A checkpoint written before this fix has no `pending_steer` key at all
    -- `.get()` must default it to None rather than KeyError, the same
    forward-compatibility promise `load_state_dict` already makes elsewhere."""
    db = tmp_path / "runs.db"
    mon = _mon(db)
    mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=1, node="agent",
                           tool_name="t", tool_args={}))
    mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=1, node="agent",
                           tool_name="t", text="ok"))
    mon.checkpoint()

    saved = mon._store.load("r1")
    del saved["totals"]["pending_steer"]           # simulate a pre-fix checkpoint
    mon._store.save("r1", saved, step=saved["totals"]["last_step"])

    resumed = _mon(db)
    assert resumed.restore() is True
    assert resumed._pending_steer is None


# ---------------------------------------------------- escalation, behaviourally
def test_a_failed_escalation_notice_survives_a_restart(tmp_path):
    """`escalation_delivered=False` means "a human was needed and nobody was
    told" -- the module's own comment calls this distinct from None ("never
    needed") on purpose, and README.md documents the same distinction as
    load-bearing. A restart must not collapse a real notification failure
    back into "nothing has happened yet".
    """
    db = tmp_path / "runs.db"
    mon = _mon(db, loop_threshold=3, max_recoveries=1)  # exhausts the ladder fast
    for _ in range(12):
        _tool_cycle(mon, mon.history[-1].step + 1 if mon.history else 1,
                    tool="list_secrets", result="a,b")
    assert mon.escalation_delivered is False, "sanity: this run must have escalated"
    assert mon.escalations >= 1
    mon.checkpoint()

    resumed = _mon(db, loop_threshold=3, max_recoveries=1)
    assert resumed.escalation_delivered is None, (
        "sanity: a fresh monitor has not escalated yet")
    assert resumed.restore() is True
    assert resumed.escalation_delivered is False, (
        "a restart turned a real notification failure back into "
        "'never needed' -- exactly the confusion escalation_delivered's "
        "None/False distinction exists to prevent")
    assert resumed.escalations == mon.escalations


def test_escalation_count_and_delivered_default_correctly_on_an_older_checkpoint(tmp_path):
    """A checkpoint written before this fix has no escalation keys at all.
    Loading it must default to "never needed" (None), not crash."""
    db = tmp_path / "runs.db"
    mon = _mon(db)
    mon.observe(AgentEvent(type=EventType.LLM_CALL, step=1, node="agent", text="x"))
    mon.checkpoint()

    resumed = _mon(db)
    assert resumed.restore() is True
    assert resumed.escalation_delivered is None
    assert resumed.escalations == 0


# ------------------------------------------------------ trace continuity
def test_a_checkpointed_restart_appends_to_the_trace_instead_of_truncating_it(tmp_path):
    """Tracer.__init__ opens its file in "w" mode by default -- correct for a
    fresh run, wrong for a restart: the monitor's own state (tokens, ladder
    memory, escalation status) all now correctly survive a checkpoint restart,
    but the one human-readable record of how it got there would be silently
    erased the instant a resumed process constructs its own default Tracer,
    before Monitor.restore() even runs. Deliberately does NOT use the `_mon`
    helper, which always passes an explicit tracer= override and so never
    exercises the monitor's own default Tracer construction this fix lives in.
    """
    db = tmp_path / "runs.db"
    trace = tmp_path / "run.jsonl"
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, checkpoint_path=str(db),
                      jsonl_path=str(trace), checkpoint_every=1))
    mon.observe(AgentEvent(type=EventType.LLM_CALL, step=1, node="agent", text="a"))
    mon.observe(AgentEvent(type=EventType.LLM_CALL, step=2, node="agent", text="b"))
    mon.checkpoint()
    lines_before = trace.read_text(encoding="utf-8").splitlines()
    assert len(lines_before) >= 3, "sanity: meta + 2 events were written"

    resumed = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, checkpoint_path=str(db),
                      jsonl_path=str(trace), checkpoint_every=1))
    lines_after_construction = trace.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_construction) >= len(lines_before), (
        "constructing a resumed monitor truncated the trace file -- the audit "
        "trail was erased even though the monitor's own state survives")
    resumed.restore()
    resumed.observe(AgentEvent(type=EventType.LLM_CALL, step=3, node="agent", text="c"))
    final = trace.read_text(encoding="utf-8").splitlines()
    assert any('"text": "a"' in l for l in final), "pre-restart event a is gone"
    assert any('"text": "c"' in l for l in final), "post-restart event c is missing"


def test_a_fresh_run_with_no_checkpoint_path_still_truncates_as_before(tmp_path):
    """The zero-config default must not change: a demo re-run against the
    same jsonl_path with no checkpoint_path is expected to start clean, not
    accumulate every past invocation forever."""
    trace = tmp_path / "run.jsonl"
    CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False,
                                        jsonl_path=str(trace))).observe(
        AgentEvent(type=EventType.LLM_CALL, step=1, node="agent", text="x"))
    first_len = len(trace.read_text(encoding="utf-8").splitlines())
    CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False,
                                        jsonl_path=str(trace)))
    second_len = len(trace.read_text(encoding="utf-8").splitlines())
    assert second_len < first_len, "a fresh (non-checkpointed) construction should truncate"


# ------------------------------------------- detector state, behaviourally
@pytest.mark.parametrize("make,drive", [
    (lambda: LoopDetector(threshold=3),
     lambda d, i: [d.inspect(AgentEvent(type=EventType.TOOL_CALL, step=i, node="a",
                                        tool_name="t", tool_args={"q": 1}), []),
                   d.inspect(AgentEvent(type=EventType.TOOL_RESULT, step=i, node="a",
                                        tool_name="t", text="same"), [])][-1]),
    (lambda: NoProgressDetector(patience=6),
     lambda d, i: [d.inspect(AgentEvent(type=EventType.TOOL_CALL, step=i,
                                        tool_name="t", tool_args={"q": 1}), []),
                   d.inspect(AgentEvent(type=EventType.TOOL_RESULT, step=i,
                                        tool_name="t", text="nothing"), [])][-1]),
    (lambda: RateOfProgressDetector(patience=8),
     lambda d, i: d.inspect(AgentEvent(type=EventType.TOOL_RESULT, step=i,
                                       tool_name="t",
                                       text=f"processed 1 of many (offset {i})",
                                       state={"at": i}), [])),
    (lambda: SpendDetector(max_tokens=5000),
     lambda d, i: d.inspect(AgentEvent(type=EventType.LLM_CALL, step=i,
                                       tokens_in=400, tokens_out=0), [])),
])
def test_detector_behaves_identically_after_a_round_trip(make, drive):
    """The guarantee that makes reflection-based serialisation safe.

    `state_dict` is written generically, so it could silently miss an attribute.
    Comparing dictionaries would not catch that — a restored detector can look
    right and still fire at the wrong time. So: drive one detector, clone it
    through a checkpoint, then drive BOTH forward and require the same verdict
    at every step, including which step first trips.
    """
    original = make()
    for i in range(1, 5):
        drive(original, i)

    clone = make()
    load_state_dict(clone, decode(encode(state_dict(original))))

    for i in range(5, 20):
        a, b = drive(original, i), drive(clone, i)
        assert (a is None) == (b is None), (
            f"restored {clone.name} diverged at step {i}: "
            f"original={'trip' if a else 'none'}, restored={'trip' if b else 'none'}")
        if a is not None:
            assert a.evidence == b.evidence, f"{clone.name} evidence diverged at {i}"


# ------------------------------------------------------------------- store
def test_encode_round_trips_the_containers_we_actually_use():
    from collections import deque
    for value in (deque([1, 2, 3], maxlen=5), (1.0, 2.0), {"a"}, {"k": [1, (2, 3)]}):
        assert decode(encode(value)) == value
    assert decode(encode(deque([1], maxlen=7))).maxlen == 7


def test_callables_and_transients_are_never_persisted():
    class Thing:
        def __init__(self):
            self.keep = 5
            self.calibrator = object()
            self._embedder = lambda t: [0.0]
            self._cache = {"big": [0.0] * 1000}

    s = state_dict(Thing())
    assert s == {"keep": 5}, f"unexpected keys persisted: {sorted(s)}"


def test_unknown_keys_do_not_break_an_older_checkpoint():
    """A missing field costs a counter; refusing to load costs the whole run."""
    d = NoProgressDetector(patience=6)
    load_state_dict(d, {"patience": 9, "a_field_from_the_future": 1})
    assert d.patience == 9


def test_store_lists_and_deletes_runs(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "runs.db"))
    store.save("run-a", {"totals": {"total_tokens": 1}}, step=3)
    store.save("run-b", {"totals": {"total_tokens": 2}}, step=9)
    assert {r["run_id"] for r in store.runs()} == {"run-a", "run-b"}
    assert store.load("run-a")["totals"]["total_tokens"] == 1
    store.delete("run-a")
    assert store.load("run-a") is None
    store.close()


def test_restore_reports_when_there_is_nothing_to_restore(tmp_path):
    mon = _mon(tmp_path / "runs.db", run_id="never-saved")
    assert mon.restore() is False


def test_checkpointing_is_off_by_default_and_costs_nothing():
    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    assert mon._store is None
    mon.checkpoint()          # must be a silent no-op, not an error
    assert mon.restore() is False


# ------------------------------------------------------------------ retention
def test_prune_keeps_the_most_recent_runs(tmp_path):
    """A service running agents continuously accumulated one row per run
    forever. Not visible in testing, where run counts are small; visible in
    month three of production."""
    store = SQLiteCheckpointStore(str(tmp_path / "runs.db"))
    for i in range(10):
        store.save(f"run-{i}", {"i": i}, step=i)
        time.sleep(0.002)          # distinct updated_at ordering

    removed = store.prune(keep_last=3)
    assert removed == 7
    survivors = {r["run_id"] for r in store.runs()}
    assert survivors == {"run-7", "run-8", "run-9"}
    store.close()


def test_prune_drops_only_what_is_older_than_the_window(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "runs.db"))
    store.save("fresh", {"a": 1})
    store.save("stale", {"a": 2})
    # Age one row by rewriting its timestamp directly.
    store._conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = 'stale'",
                        (time.time() - 40 * 86400,))
    store._conn.commit()

    assert store.prune(older_than_days=30) == 1
    assert {r["run_id"] for r in store.runs()} == {"fresh"}
    store.close()


def test_pruned_runs_are_gone_but_survivors_still_restore(tmp_path):
    """Retention must not corrupt the runs it keeps — the whole point of the
    store is that a survivor can still resume after a crash."""
    store = SQLiteCheckpointStore(str(tmp_path / "runs.db"))
    store.save("old", {"tokens": 1})
    time.sleep(0.002)
    store.save("live", {"tokens": 99})

    store.prune(keep_last=1)
    assert store.load("old") is None
    assert store.load("live") == {"tokens": 99}
    store.close()


def test_prune_with_no_arguments_deletes_nothing(tmp_path):
    """Deleting run state is irreversible, so the default must be inert."""
    store = SQLiteCheckpointStore(str(tmp_path / "runs.db"))
    store.save("a", {"x": 1})
    assert store.prune() == 0
    assert store.load("a") == {"x": 1}
    store.close()
