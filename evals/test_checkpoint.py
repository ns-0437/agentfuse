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
