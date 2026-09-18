"""Engine-level tests for ``CircuitBreakerMonitor`` itself, not any one detector.

    pytest evals/test_monitor.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse import AgentEvent, EventType, CircuitBreakerMonitor, MonitorConfig  # noqa: E402
from agentfuse.monitor import DirectiveKind  # noqa: E402
from agentfuse.detectors import DriftDetector, SpendDetector  # noqa: E402


def test_a_soft_trip_does_not_suppress_a_later_detectors_accounting():
    """A detector positioned after another in the list must still see the event.

    Independent review (2026-09-17) found this live: with drift tripping on the
    second model event and a 150-token ceiling, the monitor's own running total
    reached 200 tokens while SpendDetector — positioned after Drift — still read
    100, because the old ``_observe_locked`` returned on the FIRST trip found and
    never called ``inspect()`` on any detector after it for that event. The
    budget-enforcing detector was silently missing tokens a real supervisor is
    supposed to bound. See REPORT.md for the full writeup.
    """
    drift = DriftDetector(original_goal="database", threshold=0.9)
    spend = SpendDetector(max_tokens=150)
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal="database", echo=False, adaptive=False),
        detectors=[drift, spend],
    )
    directive = None
    for i in range(1, 12):
        directive = mon.observe(AgentEvent(
            type=EventType.LLM_CALL, step=i, text="gardening roses",
            goal="gardening roses", tokens_in=100))
        if directive.kind is not DirectiveKind.CONTINUE:
            break

    assert spend.totals["tokens"] == mon.total_tokens, (
        "SpendDetector's cumulative total drifted from the monitor's own total — "
        "an earlier detector's trip suppressed spend accounting for that event")


def test_an_exhausted_hard_budget_outranks_a_same_event_soft_trip():
    """A CRITICAL (budget-exhausted) trip must win arbitration over a TRIP-severity
    steer produced by an earlier detector on the very same event."""
    drift = DriftDetector(original_goal="database", threshold=0.9)
    spend = SpendDetector(max_tokens=150)
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal="database", echo=False, adaptive=False),
        detectors=[drift, spend],
    )
    directive = None
    for i in range(1, 12):
        directive = mon.observe(AgentEvent(
            type=EventType.LLM_CALL, step=i, text="gardening roses",
            goal="gardening roses", tokens_in=100))
        if directive.kind is not DirectiveKind.CONTINUE:
            break

    assert directive is not None
    assert directive.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT), (
        f"an exhausted hard token budget must escalate/abort, got {directive.kind}")
