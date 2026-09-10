"""AdaptiveCalibrator's health-evidence gate — presence is not progress.

``observe()`` is only supposed to record a baseline sample when the run has
just PROVEN itself healthy: a genuine advance to a state it had not already
visited. Until fixed (REPORT.md 3.22's bug, recurring here), the gate was
``event.state is not None``, which the real adapter
(``agentfuse/adapters/openai_sdk.py``) satisfies on every single
``TOOL_RESULT`` unconditionally — so a stuck loop that never advances at all
still got scored as ten separate proofs of health, one per repeated step.

Reproduced directly before the fix: ten identical (tool, args, result)
triples recorded ``baseline.samples == 10``. The numeric consequence on
``loop_threshold``/``stall_patience`` happened to cancel out for the common
single-call-per-turn cadence (every gap computes to 1, below the configured
floor) — so the measured practical effect was "the mechanism goes inert," not
"the breaker gets less safe." Still the wrong invariant to calibrate a
supervisor on, and this test pins the fix: reuse ``SeenStateTracker``'s
bounded-window novelty check, the same class ``NoProgressDetector``,
``LoopDetector``, and ``Monitor._verify_seen`` already use to answer this
exact question.

    pytest evals/test_calibration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.calibration import AdaptiveCalibrator  # noqa: E402
from agentfuse.events import AgentEvent, EventType  # noqa: E402


def _stuck_cycle(cal: AdaptiveCalibrator, step: int) -> None:
    """One (tool, args, result) triple, always the SAME state -- a stuck loop."""
    cal.observe(AgentEvent(type=EventType.TOOL_CALL, step=step,
                           tool_name="search_files", tool_args={"pattern": "*.conn"}))
    cal.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step,
                           tool_name="search_files", text="0 files matched",
                           state={"last_tool": "search_files", "result": "0 files matched"}))


def _advancing_cycle(cal: AdaptiveCalibrator, step: int) -> None:
    """One (tool, args, result) triple with state that genuinely differs each step."""
    cal.observe(AgentEvent(type=EventType.TOOL_CALL, step=step,
                           tool_name="t", tool_args={"i": step}))
    cal.observe(AgentEvent(type=EventType.TOOL_RESULT, step=step,
                           tool_name="t", text="ok", state={"batch": step}))


def test_a_stuck_loop_does_not_fabricate_baseline_samples():
    """Ten repeats of one unchanging state must count as ONE health sample, not ten.

    Before the fix this asserted ``== 10`` -- every repeat of the identical
    state was wrongly scored as fresh evidence the run had just proven itself
    healthy.
    """
    cal = AdaptiveCalibrator()
    for step in range(1, 11):
        _stuck_cycle(cal, step)
    assert cal.baseline.samples == 1, (
        "a state hash the run has already visited must not count as a new "
        "health sample -- presence of `state` is not evidence of an advance")


def test_a_genuinely_advancing_run_still_calibrates():
    """The fix must not throw out real evidence along with the fake kind."""
    cal = AdaptiveCalibrator()
    for step in range(1, 6):
        _advancing_cycle(cal, step)
    assert cal.baseline.samples == 5, (
        "a run that genuinely advances every step must still record every "
        "sample -- the fix narrows the gate, it must not starve it")
    assert cal.baseline.ready


def test_a_state_revisited_after_genuine_progress_does_not_recount():
    """Returning to an earlier state is not a NEW advance the second time."""
    cal = AdaptiveCalibrator()
    _advancing_cycle(cal, 1)   # state={"batch": 1} -- new, counts
    _advancing_cycle(cal, 2)   # state={"batch": 2} -- new, counts
    cal.observe(AgentEvent(type=EventType.TOOL_CALL, step=3,
                           tool_name="t", tool_args={"i": 1}))
    cal.observe(AgentEvent(type=EventType.TOOL_RESULT, step=3,
                           tool_name="t", text="ok", state={"batch": 1}))  # revisit
    assert cal.baseline.samples == 2, "revisiting an already-seen state is not a fresh advance"
