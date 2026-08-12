"""Additional generator families — added for statistical power, not volume.

Measured intra-cluster correlation on this suite is high: scenarios produced by
one generator behave almost identically, so forty of them carry roughly one
sample's worth of information. The design effect was 16.9x, putting the effective
sample size near 19 against a nominal 320.

The consequence is specific and actionable: **adding scenarios per generator buys
no statistical power at all.** Only adding independent generators narrows the
honest interval. So these are deliberately different failure *shapes* rather than
new variations on the shapes already present — three failures that the existing
detectors have no obvious handle on, and two healthy behaviours that superficially
resemble failures.
"""

from __future__ import annotations

import random

from .generators import DOMAINS, _healthy_prefix, _recovery_steps, _responds_to, _tokens
from .schema import Label, Scenario, StepSpec, think, tool


# --------------------------------------------------------------------- positives
def gen_oscillating_plan(rng: random.Random, idx: int) -> Scenario:
    """Flips between two incompatible plans, committing to neither.

    Distinct from a tool loop: the calls differ each step and state occasionally
    advances, so neither the loop counter nor the stall counter has a clean
    handle. The failure lives at the level of intent, not action.
    """
    d = rng.choice(DOMAINS)
    steps = _healthy_prefix(rng, d, rng.randint(1, 2))
    onset = len(steps)
    plan_a, plan_b = d["tools"][0], d["tools"][1]
    for i in range(rng.randint(6, 10)):
        ti, to = _tokens(rng)
        chosen, dropped = (plan_a, plan_b) if i % 2 == 0 else (plan_b, plan_a)
        steps.append(think(
            f"On reflection {chosen} is the better approach; abandoning {dropped}.",
            tokens_in=ti, tokens_out=to))
        steps.append(tool(chosen, d["args"](rng), result="started, then abandoned",
                          progress=False, tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_oscillate_{idx:04d}",
        title=f"Oscillating plan ({d['key']})",
        family="progress", goal=d["goal"], steps=steps,
        description="Flip-flops between two plans, finishing neither.",
        label=Label(should_trip=True, detector="progress", onset_index=onset,
                    detect_by_index=onset + 10),
        recovery_branch=_recovery_steps(rng, d), responds_to=_responds_to(rng),
    )


def gen_partial_progress_trap(rng: random.Random, idx: int) -> Scenario:
    """Tiny state changes forever, converging never.

    The hardest case for a progress-based detector, because progress genuinely
    *is* being made — it is simply never enough. Every advance resets the stall
    counter, so a binary progress signal can be starved indefinitely by an agent
    inching nowhere.
    """
    d = rng.choice(DOMAINS)
    steps = _healthy_prefix(rng, d, 1)
    onset = len(steps)
    for i in range(rng.randint(10, 16)):
        ti, to = _tokens(rng)
        steps.append(tool(rng.choice(d["tools"]), {"offset": i},
                          result=f"processed 1 of many (offset {i})",
                          progress=True,  # real, but negligible
                          tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_zeno_{idx:04d}",
        title=f"Partial-progress trap ({d['key']})",
        family="progress", goal=d["goal"], steps=steps,
        description="Advances on every step; converges on none.",
        label=Label(should_trip=True, detector="progress", onset_index=onset,
                    detect_by_index=onset + 14, known_gap=True,
                    note=("KNOWN GAP: every step advances state, so a binary "
                          "progress signal never fires. Needs a RATE-of-progress "
                          "measure — is the remaining work actually shrinking?")),
        recovery_branch=_recovery_steps(rng, d), responds_to=_responds_to(rng),
    )


def gen_context_bloat(rng: random.Random, idx: int) -> Scenario:
    """Every call succeeds; context grows until cost dominates.

    A failure with no failing action anywhere in it — a shape none of the other
    generators produce.
    """
    d = rng.choice(DOMAINS)
    steps: list[StepSpec] = []
    for i in range(rng.randint(10, 15)):
        steps.append(tool(rng.choice(d["tools"]), d["args"](rng),
                          result=f"ok ({i})", progress=True,
                          tokens_in=900 + i * 900, tokens_out=200 + i * 60))
    return Scenario(
        id=f"gen_bloat_{idx:04d}",
        title=f"Context bloat ({d['key']})",
        family="spend", goal=d["goal"], steps=steps,
        config={"max_tokens": 400_000, "burst_window": 5, "burst_tokens": 30_000},
        description="Healthy actions throughout; unbounded context growth.",
        label=Label(should_trip=True, detector="spend", onset_index=4,
                    detect_by_index=14),
        recovery_branch=_recovery_steps(rng, d, n=2), responds_to=_responds_to(rng),
    )


# --------------------------------------------------------------------- negatives
def gen_benign_error_then_pivot(rng: random.Random, idx: int) -> Scenario:
    """A real error, a genuine change of approach, then success.

    Superficially this is a loop followed by drift: the agent repeats a failing
    call, then starts talking about something else. It is simply competence.
    """
    d = rng.choice(DOMAINS)
    steps = _healthy_prefix(rng, d, 1)
    failing = rng.choice(d["tools"])
    for _ in range(2):
        ti, to = _tokens(rng)
        steps.append(tool(failing, d["args"](rng), result="ERROR: permission denied",
                          progress=False, tokens_in=ti, tokens_out=to))
    ti, to = _tokens(rng)
    steps.append(think("Permission denied twice, so that route is closed. "
                       + d["on_topic"][-1], tokens_in=ti, tokens_out=to))
    steps.extend(_healthy_prefix(rng, d, rng.randint(2, 3)))
    return Scenario(
        id=f"gen_pivot_{idx:04d}",
        title=f"Error then competent pivot ({d['key']})",
        family="benign", goal=d["goal"], steps=steps,
        description="Hitting a wall and choosing another route is not a failure.",
        label=Label(should_trip=False,
                    note="Repeat + topic change, both entirely legitimate here."),
    )


def gen_benign_verification_heavy(rng: random.Random, idx: int) -> Scenario:
    """Repeatedly double-checking the same fact — diligence, not looping."""
    d = rng.choice(DOMAINS)
    steps: list[StepSpec] = []
    for cycle in range(rng.randint(4, 6)):
        ti, to = _tokens(rng)
        steps.append(tool(rng.choice(d["tools"]), d["args"](rng),
                          result=f"value written (pass {cycle})", progress=True,
                          tokens_in=ti, tokens_out=to))
        for check in range(2):
            ti, to = _tokens(rng)
            steps.append(tool("verify", {"pass": cycle},
                              result=f"verified pass {cycle} check {check}",
                              progress=True, tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_verify_{idx:04d}",
        title=f"Verification-heavy run ({d['key']})",
        family="benign", goal=d["goal"], steps=steps,
        description="Repeated checking, with real progress on every step.",
        label=Label(should_trip=False),
    )


EXTRA_POSITIVES = [gen_oscillating_plan, gen_partial_progress_trap, gen_context_bloat]
EXTRA_NEGATIVES = [gen_benign_error_then_pivot, gen_benign_verification_heavy]
