"""The steering ladder — escalating *kinds* of correction, not rephrasings.

Phase 1 measured recovery and exposed the shape of the weakness: when a steer
failed, the next trip produced substantially the same advice, because it was the
same reasoning applied to the same snapshot. An agent that ignored "stop
repeating that tool" will ignore it again when it arrives slightly reworded.
Three attempts, one idea.

So corrections are organised as a ladder of genuinely different interventions,
from the lightest touch to giving up:

    1. re-anchor          restate the objective; the agent may simply have lost it
    2. alternate-action   forbid the failing action, demand a different one
    3. challenge-assumption  the plan is failing because a premise is false; find it
    4. decompose          the task may be too large to attack directly; break it up
    5. escalate           a human is needed

Each rung assumes the previous one was insufficient, so it intervenes harder and
constrains the agent more. Combined with :mod:`agentfuse.memory`, which records
what has already been tried against a given failure, the engine can pick the
first rung *not yet attempted* rather than looping on rung one.

The rungs are ordered by how much agency they remove. Escalation is last not
because it is worst, but because it is the only one that costs a human's
attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

RE_ANCHOR = "re-anchor"
ALTERNATE_ACTION = "alternate-action"
CHALLENGE_ASSUMPTION = "challenge-assumption"
DECOMPOSE = "decompose"
ESCALATE = "escalate"

LADDER = [RE_ANCHOR, ALTERNATE_ACTION, CHALLENGE_ASSUMPTION, DECOMPOSE, ESCALATE]


@dataclass(frozen=True)
class Strategy:
    """One rung: how to phrase it, and how to brief a reasoning model to write it."""

    name: str
    intent: str                       # one line, used in the supervisor prompt
    build: Callable[[dict], str]      # deterministic fallback instruction


def _goal(ctx: dict) -> str:
    return ctx.get("goal", "your original objective")


def _tool(ctx: dict) -> str:
    return ctx.get("tool") or "the tool you keep calling"


STRATEGIES: dict[str, Strategy] = {
    RE_ANCHOR: Strategy(
        name=RE_ANCHOR,
        intent="Restate the original objective and ask the agent to re-align to it.",
        build=lambda c: (
            f"Pause and re-read your assigned objective: \"{_goal(c)}\". "
            f"Your recent actions have stopped serving it. State, in one sentence, "
            f"what your next action is and how it advances that objective."
        ),
    ),
    ALTERNATE_ACTION: Strategy(
        name=ALTERNATE_ACTION,
        intent=("Forbid the specific failing action and require a materially "
                "different next step."),
        build=lambda c: (
            f"Do NOT call `{_tool(c)}` again — it has been tried and is not "
            f"advancing the task. You must take a materially different action: a "
            f"different tool, or a different approach to \"{_goal(c)}\". Name the "
            f"alternative you are choosing and why it should work where the last "
            f"one did not."
        ),
    ),
    CHALLENGE_ASSUMPTION: Strategy(
        name=CHALLENGE_ASSUMPTION,
        intent=("Surface the false premise the plan rests on; earlier nudges failed, "
                "so the problem is the reasoning, not the action."),
        build=lambda c: (
            f"Previous corrections have not worked, which means the problem is your "
            f"reasoning rather than your choice of action. List the assumptions your "
            f"recent steps depend on. Identify the one most likely to be FALSE. "
            f"Take a single concrete action that would verify or refute it, then "
            f"re-plan for \"{_goal(c)}\" from what you learn."
        ),
    ),
    DECOMPOSE: Strategy(
        name=DECOMPOSE,
        intent="The objective may be too large to attack directly; break it down.",
        build=lambda c: (
            f"Stop attacking \"{_goal(c)}\" as a single step — repeated attempts "
            f"have failed. Break it into the smallest sub-goals that can be "
            f"completed and verified independently. State them as a numbered list, "
            f"then attempt ONLY the first one and report the result before "
            f"continuing."
        ),
    ),
    ESCALATE: Strategy(
        name=ESCALATE,
        intent="Automated recovery has been exhausted; hand control to a human.",
        build=lambda c: (
            f"Automated recovery has been exhausted for \"{_goal(c)}\". Halting and "
            f"escalating to a human with a summary of what was attempted, what was "
            f"ruled out, and where the run stopped."
        ),
    ),
}


def next_strategy(already_tried: set[str], severity: str = "trip",
                  max_rungs: Optional[int] = None) -> str:
    """Pick the lightest intervention that has not already been tried.

    A CRITICAL trip skips the ladder: a hard budget ceiling is not a reasoning
    problem and no amount of steering makes the money reappear.
    """
    if severity == "critical":
        return ESCALATE
    rungs = LADDER[:max_rungs] if max_rungs else LADDER
    for name in rungs:
        if name not in already_tried:
            return name
    return ESCALATE


def build_instruction(strategy: str, context: dict) -> str:
    return STRATEGIES.get(strategy, STRATEGIES[RE_ANCHOR]).build(context)


def describe_for_prompt(strategy: str) -> str:
    """The line handed to a reasoning model so it writes the right *kind* of fix."""
    s = STRATEGIES.get(strategy)
    return f"{strategy}: {s.intent}" if s else strategy
