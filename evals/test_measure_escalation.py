"""analyse()'s climbed_past_rung_1 must be a real bool, not a truthy list.

Caught while writing up the first live escalation-ladder capture: the
original expression was `len(set(strategies)) > 1 or (strategies and
strategies[0] != STEERABLE[0])` -- when no trip occurred at all, `strategies`
is `[]`, and `False or []` evaluates to `[]`, not `False`. Harmless in a
truthiness check, but wrong in the results.json artifact this script writes
(`"climbed_past_rung_1": []` instead of `false`), which is exactly the kind
of small inconsistency that erodes trust in a report's own numbers.

    pytest evals/test_measure_escalation.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import AgentEvent, CircuitBreakerMonitor, EventType, MonitorConfig, Tracer  # noqa: E402
from evals.measure_escalation import analyse  # noqa: E402

GOAL = "Rotate the production database credential."


def test_a_run_with_no_trips_reports_climbed_as_false_not_an_empty_list(tmp_path):
    trace = tmp_path / "run.jsonl"
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False))
    mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=1, tool_name="t", tool_args={}))
    mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=1, tool_name="t", text="ok"))

    result = analyse(trace)
    assert result["climbed_past_rung_1"] is False, (
        f"expected the real bool False, got {result['climbed_past_rung_1']!r} "
        f"({type(result['climbed_past_rung_1']).__name__})")
    # json round-trip must also read back as `false`, not `[]`
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["climbed_past_rung_1"] is False


def test_a_single_trip_never_counts_as_climbed(tmp_path):
    trace = tmp_path / "run.jsonl"
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, loop_threshold=3,
                      jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False))
    for step in range(1, 5):
        mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=step, tool_name="t",
                               tool_args={"d": "x"}))
        mon.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step, tool_name="t",
                               text="0 files matched"))

    result = analyse(trace)
    assert result["rungs_fired"] == ["re-anchor"]
    assert result["climbed_past_rung_1"] is False
