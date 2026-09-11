"""The JSONL trace's own contract — can an incident review answer "did it climb?"

Found while preparing a live escalation-ladder capture: ``Tracer.recovery()``
wrote ``action``/``instruction``/``rationale``/``confidence``/``backend`` to
the trace but never ``path.strategy`` -- the rung of the ladder (re-anchor,
alternate-action, challenge-assumption, decompose) that produced the steer.
``evals/runner.py`` and ``evals/real_model.py`` read ``path.strategy`` directly
off the live Python object because synthetic/real-model runs happen in one
process, but nothing reading a persisted ``runs/*.jsonl`` file afterward --
the dashboard, a real capture script, an incident review -- could ever recover
which rung fired. A trace that cannot answer "did the ladder actually escalate,
or did it just repeat rung 1?" fails at the one thing a trace is for.

    pytest evals/test_tracer.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import AgentEvent, CircuitBreakerMonitor, EventType, MonitorConfig, Tracer  # noqa: E402

GOAL = "Rotate the production database credential."


def _trip_loop(mon: CircuitBreakerMonitor, steps: range) -> None:
    for step in steps:
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, node="agent",
                               tool_name="search_files", tool_args={"d": "./config"}))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, node="agent",
                               tool_name="search_files", text="0 files matched"))


def test_a_persisted_trace_records_which_rung_fired(tmp_path):
    trace = tmp_path / "run.jsonl"
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False))
    _trip_loop(mon, range(1, 5))

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    recoveries = [r for r in records if r.get("kind") == "recovery"]
    assert recoveries, "the loop never tripped a recovery -- nothing to check"
    assert recoveries[0].get("strategy"), (
        "a persisted recovery record has no `strategy` field -- an incident "
        "review or a real-model capture reading this file cannot tell which "
        "rung of the ladder fired, or whether it ever climbed past rung 1")
    assert recoveries[0]["strategy"] == "re-anchor", (
        "the first trip's rung should be the ladder's first rung")


def test_a_second_trip_after_a_failed_steer_climbs_the_ladder(tmp_path):
    """The escalation ladder's whole point: a failed rung must not repeat itself."""
    trace = tmp_path / "run.jsonl"
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      max_recoveries=6, jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False))
    # First trip -> re-anchor. Ignore the steer entirely (repeat the identical
    # call) so the verify-progress window closes with nothing advanced.
    _trip_loop(mon, range(1, 5))
    _trip_loop(mon, range(5, 9))

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    strategies = [r["strategy"] for r in records if r.get("kind") == "recovery"]
    assert len(strategies) >= 2, "expected at least two recoveries from two trips"
    assert strategies[0] == "re-anchor"
    assert strategies[1] != "re-anchor", (
        "a rung whose steer was ignored must be ruled out on the next trip, "
        "not repeated -- this is the section 3.22 bug (a rung only counted as "
        "failed if genuine progress was verified) showing up again if it does")
