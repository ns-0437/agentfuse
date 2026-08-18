"""Does the benchmark's event stream match what production actually emits?

This is a validity check on the benchmark itself, not on the detectors. Every
number in the eval is produced by replaying synthetic events through the real
monitor. If those events differ in shape from what the adapters emit in
production, the whole suite measures a fiction — thresholds tuned against it
would not transfer, and no amount of scenario count would reveal the problem.

It had in fact diverged: the eval emitted `tool_call -> tool_result` while a
real `openai-agents` run emits `llm_call -> tool_call -> tool_result`, with the
token usage on the model turn rather than the tool call. Correcting it moved
recall by 4.5 points, which is the size of the error that was being made.

These tests compare against a genuine captured trace so it cannot drift again.

    pytest evals/test_fidelity.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.events import EventType  # noqa: E402
from evals.runner import _events_for_step  # noqa: E402
from evals.schema import tool, think  # noqa: E402

TRACE = ROOT / "runs" / "real_agentkit.jsonl"


def _real_event_types() -> list[str]:
    if not TRACE.exists():
        pytest.skip("run examples/real_agentkit_run.py to capture a reference trace")
    out = []
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "event":
            out.append(rec["type"])
    return out


def test_tool_step_is_preceded_by_a_model_turn():
    """In production the model turn is what produces the tool call."""
    types = [e.type for e in _events_for_step(tool("t", {"a": 1}, result="r"), 1)]
    assert types[0] is EventType.LLM_CALL, (
        "a tool call with no preceding model turn does not occur in production")
    assert types[1] is EventType.TOOL_CALL
    assert types[2] is EventType.TOOL_RESULT


def test_tokens_are_attributed_to_the_model_turn():
    """Real adapters read usage off the LLM response, never off a tool call."""
    events = _events_for_step(tool("t", {"a": 1}, result="r",
                                   tokens_in=700, tokens_out=160), 1)
    llm = next(e for e in events if e.type is EventType.LLM_CALL)
    call = next(e for e in events if e.type is EventType.TOOL_CALL)
    assert llm.tokens_in == 700 and llm.tokens_out == 160
    assert call.tokens_in == 0 and call.tokens_out == 0, (
        "tool calls must not carry token usage — the model turn does")


def test_synthetic_ordering_matches_the_real_adapter():
    """The llm->tool->result triple must appear in the real trace too."""
    real = _real_event_types()
    triple = ["llm_call", "tool_call", "tool_result"]
    found = any(real[i:i + 3] == triple for i in range(len(real) - 2))
    assert found, f"real trace never shows {triple}; sequence was {real[:12]}"


def test_no_event_type_is_invented_by_the_benchmark():
    """Every synthetic event type must be one the adapters actually produce."""
    real = set(_real_event_types())
    synthetic = set()
    for step in (tool("t", {"a": 1}, result="r", progress=True), think("x", progress=True)):
        synthetic |= {e.type.value for e in _events_for_step(step, 1)}
    invented = synthetic - real - {"route"}  # route is emitted on handoffs only
    assert not invented, f"benchmark emits event types production never does: {invented}"


def test_think_step_is_a_bare_model_turn():
    types = [e.type for e in _events_for_step(think("reasoning"), 1)]
    assert types == [EventType.LLM_CALL]


# --------------------------------------------------- captured real traces
def test_captured_traces_are_scored():
    """Real captured runs, replayed through the same scoring path as synthetic ones.

    Honest scope: this validates the event SHAPE against production, not the
    failure distribution. The scenario is still one that was designed, so it does
    not cure authoring bias — traces from agents solving tasks nobody chose for
    their failure characteristics are what would.
    """
    from evals.trace_import import load_captured
    from evals.runner import run_scenario

    captured = load_captured()
    if not captured:
        pytest.skip("no captured traces committed")

    for scenario in captured:
        assert scenario.steps, f"{scenario.id} imported with no steps"
        assert scenario.goal, f"{scenario.id} lost its objective on import"
        result = run_scenario(scenario)
        expected = "TP" if scenario.label.should_trip else "TN"
        if scenario.label.known_gap and result.outcome != expected:
            # A disagreement we have written down and explained. Keeping it in
            # the corpus with its real label is the point: deleting the trace or
            # relabelling it to match the code would erase the only evidence of
            # the weakness. See the note on the label itself.
            continue
        assert result.outcome == expected, (
            f"{scenario.id}: expected {expected}, got {result.outcome} "
            f"(detector={result.trip_detector})")


def test_documented_gaps_are_still_gaps():
    """A known gap that quietly starts passing is stale documentation.

    `known_gap` suppresses the outcome assertion, so once a gap closes nothing
    reports it — the corpus keeps warning about a weakness that no longer exists,
    and REPORT.md keeps telling readers the detector cannot do something it can.
    Since a closed gap is good news, this fails LOUDLY with instructions rather
    than silently tolerating either state.
    """
    from evals.trace_import import load_captured
    from evals.runner import run_scenario

    captured = [s for s in load_captured() if s.label.known_gap]
    if not captured:
        pytest.skip("no documented gaps in the captured corpus")

    closed = []
    for scenario in captured:
        expected = "TP" if scenario.label.should_trip else "TN"
        if run_scenario(scenario).outcome == expected:
            closed.append(scenario.id)

    assert not closed, (
        f"documented gap(s) now behave correctly: {closed}. This is good news — "
        f"remove `known_gap` from the label, update the note with what fixed it, "
        f"and correct the matching section of REPORT.md so the report stops "
        f"claiming a weakness that no longer exists.")


def test_captured_trace_uses_the_production_progress_convention():
    """The adapter attaches progress to TOOL_RESULT; no STATE_UPDATE events exist.

    If a captured trace ever contains standalone STATE_UPDATE events, either an
    adapter changed or the trace was synthesised — and the benchmark's fidelity
    claim would need re-checking.
    """
    import json
    from pathlib import Path as _P

    traces = list((_P(__file__).resolve().parent / "captured").glob("*.jsonl"))
    if not traces:
        pytest.skip("no captured traces committed")

    for path in traces:
        types = {json.loads(line)["type"]
                 for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and json.loads(line).get("kind") == "event"}
        assert "state_update" not in types, (
            f"{path.name} contains standalone STATE_UPDATE events, which the "
            f"AgentKit adapter does not emit")
