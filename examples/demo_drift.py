"""AgentFuse demo — goal drift, detected and re-anchored.

The hardest long-horizon failure: no crash, no loop, no error — the agent just
quietly slides off its objective, one locally-reasonable step at a time. Here a
research agent is told to summarize Q3 revenue from the finance report, but drifts
turn by turn into an unrelated competitor-marketing tangent. The breaker measures
semantic distance from the original objective and re-anchors it before the whole
run is wasted.

Run:  python examples/demo_drift.py
(With OPENAI_API_KEY set, drift is measured with real embeddings + a real
reasoning model generates the steering path.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentfuse import (
    CircuitBreakerMonitor, MonitorConfig, AgentEvent, EventType, DirectiveKind,
)

GOAL = "Summarize the Q3 revenue figures from the attached finance report into three bullet points."

# The agent's reasoning text, turn by turn, drifting further each step.
DRIFTING_TURNS = [
    "Opening the finance report to read the Q3 revenue section.",
    "Q3 revenue looks tied to the new product line's marketing push.",
    "Let me research how competitors marketed similar products this year.",
    "Comparing competitor ad spend and social media campaign strategies in detail.",
    "Drafting recommendations for our own future social media marketing calendar.",
]

# After steering, the agent snaps back to the assigned task.
RECOVERED_TURNS = [
    "Re-focusing: extracting the three key Q3 revenue figures from the report.",
]


def main() -> None:
    print("\n" + "=" * 72)
    print("AgentFuse — long-range autonomy circuit breaker")
    print("Scenario: agent slowly DRIFTS from its objective (no crash, no loop)")
    print("=" * 72)
    print(f"OBJECTIVE: {GOAL}\n")

    monitor = CircuitBreakerMonitor(MonitorConfig(
        # No drift_threshold here on purpose: None lets the detector pick the
        # value for whichever mode it resolves to. This demo hard-coded 0.45,
        # which was right for the lexical fallback and IMPOSSIBLE for embeddings
        # — those similarities sit around 0.6-0.8, so 0.45 can never be crossed.
        # The demo therefore stopped tripping the moment local embeddings landed
        # and quietly became a demo of nothing. Exactly the same mistake had
        # already been found and fixed in the eval harness; nobody checked here.
        original_goal=GOAL, max_recoveries=3,
        jsonl_path="runs/drift.jsonl",
    ))

    step = 0
    recovered = False
    turns = list(DRIFTING_TURNS)
    i = 0
    while i < len(turns):
        step += 1
        text = turns[i]
        directive = monitor.observe(AgentEvent(
            type=EventType.LLM_CALL, step=step, node="analyst", text=text,
            goal=text, tokens_in=1200, tokens_out=300, cost_usd=0.006,
        ))
        if directive.kind is DirectiveKind.INJECT and not recovered:
            recovered = True
            monitor.observe(AgentEvent(
                type=EventType.RESUME, step=step, node="supervisor",
                text="re-anchored to original objective",
            ))
            turns = turns[: i + 1] + RECOVERED_TURNS  # agent corrects course
        i += 1

    monitor.observe(AgentEvent(
        type=EventType.COMPLETE, step=step + 1, node="analyst",
        text="Q3 revenue summarized in three bullets", state={"done": True},
    ))
    totals = monitor.finish("complete" if recovered else "incomplete")
    print(f"\n>> Re-anchored and completed. Trips: {totals['trips']} | "
          f"Recoveries: {totals['recoveries']}")
    print(">> Trace written to runs/drift.jsonl")


if __name__ == "__main__":
    main()
