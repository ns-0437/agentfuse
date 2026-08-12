"""AgentFuse demo — when NOT to auto-heal: escalate to a human.

A good circuit breaker knows its limits. Two cases here:

  1. Repeated recoveries that don't take: after ``max_recoveries`` steering
     attempts fail to fix the same failure, the breaker stops steering and
     escalates — no infinite "self-heal" loop.
  2. A hard budget ceiling (CRITICAL severity): the run is paused for a human
     regardless of how promising it looks.

This is the safety story: autonomy with a hard stop, not blind persistence.

Run:  python examples/demo_escalation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentfuse import (
    CircuitBreakerMonitor, MonitorConfig, AgentEvent, EventType, DirectiveKind,
)

GOAL = "Reconcile the vendor invoices against purchase orders and flag mismatches."


def main() -> None:
    print("\n" + "=" * 72)
    print("AgentFuse — long-range autonomy circuit breaker")
    print("Scenario: unrecoverable loop -> escalate to a human (hard stop)")
    print("=" * 72)
    print(f"OBJECTIVE: {GOAL}\n")

    monitor = CircuitBreakerMonitor(MonitorConfig(
        original_goal=GOAL,
        loop_threshold=3,
        max_recoveries=2,          # only two steering attempts, then escalate
        max_tokens=60_000,         # hard ceiling -> CRITICAL escalation
        jsonl_path="runs/escalation.jsonl",
    ))

    step = 0
    for _ in range(60):
        step += 1
        # A stubborn agent that ignores steering and keeps hammering the same
        # broken tool (e.g. an API that always 500s).
        directive = monitor.observe(AgentEvent(
            type=EventType.TOOL_CALL, step=step, node="reconciler",
            tool_name="fetch_invoice", tool_args={"vendor": "acme", "page": 1},
            tokens_in=1500, tokens_out=400, cost_usd=0.01,
        ))
        result_directive = monitor.observe(AgentEvent(
            type=EventType.TOOL_RESULT, step=step, node="reconciler",
            tool_name="fetch_invoice", text="HTTP 500 upstream error",
        ))
        # The loop detector waits for the result before tripping, so the
        # directive can arrive on either observation.
        if directive.kind is DirectiveKind.CONTINUE:
            directive = result_directive

        if directive.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT):
            print(f"\n>> HARD STOP at step {step}: control returned to a human "
                  f"instead of burning more budget on a failing action.")
            totals = monitor.finish("escalated")
            print(f">> Trips: {totals['trips']} | Recoveries attempted: "
                  f"{totals['recoveries']} | Spend: ${totals['total_cost_usd']}")
            print(">> Trace written to runs/escalation.jsonl")
            return

    monitor.finish("complete")


if __name__ == "__main__":
    main()
