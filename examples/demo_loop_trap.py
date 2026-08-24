"""AgentFuse demo — loop trap, detected and self-healed.

Runs a *simulated* long-horizon agent (no API key required) that falls into the
classic infinite-tool-loop failure: it keeps searching the same directory for a
config file that isn't there, reasoning consistently from a false premise. The
circuit breaker detects the loop, freezes state, climbs the deterministic
escalation ladder for a steering path, injects the correction, and the agent
recovers and finishes.

Run:
    python examples/demo_loop_trap.py

With a real OpenAI key set, the recovery step calls a real reasoning model
instead of the offline mock:
    OPENAI_API_KEY=... AGENTFUSE_RECOVERY_MODEL=o4-mini python examples/demo_loop_trap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentfuse import (
    CircuitBreakerMonitor,
    MonitorConfig,
    AgentEvent,
    EventType,
    DirectiveKind,
)

GOAL = (
    "Rotate the production database credential: locate the active connection "
    "string, generate a new secret, and update the secret store."
)


class SimulatedAgent:
    """A scripted agent that gets stuck, then recovers once steered.

    Behavior model:
      * By default it believes the connection string is a file in ./config and
        keeps calling `search_files` on that directory (the false premise).
      * When it receives a steering instruction that tells it to try a different
        path, it switches to the *correct* plan: query the secret manager API.
    """

    def __init__(self):
        self.step = 0
        self.steered = False
        self.done = False
        self.plan = "search_config_dir"

    def next_action(self) -> AgentEvent:
        self.step += 1
        if self.plan == "search_config_dir":
            # The trap: same tool, same args, forever.
            return AgentEvent(
                type=EventType.TOOL_CALL, step=self.step, node="researcher",
                tool_name="search_files",
                tool_args={"dir": "./config", "pattern": "*.conn"},
                tokens_in=800, tokens_out=200, cost_usd=0.004,
                # On-objective phrasing: the agent isn't drifting, it's looping —
                # doing the right goal via a doomed repeated action.
                goal="locate the active production database connection string to rotate the credential",
            )
        if self.plan == "query_secret_manager":
            self.done = True
            return AgentEvent(
                type=EventType.TOOL_CALL, step=self.step, node="researcher",
                tool_name="secret_manager.get",
                tool_args={"name": "prod/db/primary"},
                tokens_in=600, tokens_out=150, cost_usd=0.003,
                goal="retrieve the active DB credential from the secret manager",
                state={"found_connection_string": True},  # real progress!
            )
        raise RuntimeError("unknown plan")

    def apply_steering(self, instruction: str) -> None:
        """Receiving a real corrective nudge, the agent changes its plan."""
        self.steered = True
        # The steering told it to abandon the false premise and try another path.
        self.plan = "query_secret_manager"


def main() -> None:
    print("\n" + "=" * 72)
    print("AgentFuse — long-range autonomy circuit breaker")
    print("Scenario: agent stuck in an infinite tool loop (false premise)")
    print("=" * 72)
    print(f"OBJECTIVE: {GOAL}\n")

    monitor = CircuitBreakerMonitor(MonitorConfig(
        original_goal=GOAL,
        loop_threshold=3,
        max_recoveries=3,
        max_tokens=200_000,
        jsonl_path="runs/loop_trap.jsonl",
    ))
    agent = SimulatedAgent()

    max_steps = 25
    for _ in range(max_steps):
        if agent.done:
            break
        event = agent.next_action()
        directive = monitor.observe(event)

        # Simulate the tool returning "not found" for the loop plan, so no
        # state progress is ever recorded and the loop is real.
        if event.tool_name == "search_files":
            result_directive = monitor.observe(AgentEvent(
                type=EventType.TOOL_RESULT, step=event.step, node="researcher",
                tool_name="search_files", text="0 files matched",
            ))
            # The loop detector deliberately waits for the *result* before
            # tripping — a call that is about to succeed is not a loop — so the
            # directive can arrive on either observation. Take whichever acts.
            if directive.kind is DirectiveKind.CONTINUE:
                directive = result_directive

        if directive.kind is DirectiveKind.INJECT:
            agent.apply_steering(directive.steering_text or "")
            monitor.observe(AgentEvent(
                type=EventType.RESUME, step=event.step, node="supervisor",
                text="steering injected; agent resuming with corrected plan",
            ))
        elif directive.kind in (DirectiveKind.PAUSE, DirectiveKind.ABORT):
            print("\n>> Human escalation required — halting autonomous run.")
            monitor.finish("escalated")
            return

    if agent.done:
        monitor.observe(AgentEvent(
            type=EventType.COMPLETE, step=agent.step + 1, node="researcher",
            text="connection string retrieved and rotated",
            state={"rotated": True},
        ))
        totals = monitor.finish("complete")
        print(f"\n>> Objective achieved after self-healing. "
              f"Trips: {totals['trips']} | Recoveries: {totals['recoveries']} | "
              f"Steps: {totals['steps']}")
        print(">> Full machine-readable trace written to runs/loop_trap.jsonl")
    else:
        monitor.finish("incomplete")
        print("\n>> Agent did not complete within step budget.")


if __name__ == "__main__":
    main()
