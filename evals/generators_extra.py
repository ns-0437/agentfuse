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
        label=Label(should_trip=True, detector="rate", onset_index=onset,
                    detect_by_index=onset + 14,
                    note=("Was a documented KNOWN GAP: every step advances state, "
                          "so the binary progress signal is structurally unable to "
                          "fire. Closed by RateOfProgressDetector, which asks "
                          "whether the trace carries evidence that the remaining "
                          "work is shrinking. Paired with gen_benign_batch, which "
                          "is the same shape and genuinely converging.")),
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


def gen_benign_batch(rng: random.Random, idx: int) -> Scenario:
    """A long batch job: identical-shaped advances, but genuinely converging.

    The hard negative for the rate-of-progress detector, and the reason that
    detector cannot simply trip on "the same thing over and over". Structurally
    this is ``gen_partial_progress_trap``: dozens of consecutive advances, every
    result the same sentence with a different number in it. Superficially there
    is no difference at all.

    The one difference is the one that matters — this run reports what is LEFT.
    A real batch job knows its own denominator, so its trace carries a countdown
    and a total being approached, while a Zeno trap carries only a cursor
    climbing against nothing. If the detector ever trips here it is reading
    repetition as failure, which is the exact mistake that gets guardrails
    switched off.
    """
    d = rng.choice(DOMAINS)
    total = rng.choice([120, 240, 500])
    steps: list[StepSpec] = []
    n = rng.randint(10, 16)
    for i in range(n):
        ti, to = _tokens(rng)
        done = (i + 1) * (total // (n + 1))
        steps.append(tool(rng.choice(d["tools"]), {"cursor": i},
                          result=f"processed {done} of {total} records; "
                                 f"{total - done} remaining",
                          progress=True, tokens_in=ti, tokens_out=to))
    return Scenario(
        id=f"gen_batch_{idx:04d}",
        title=f"Converging batch job ({d['key']})",
        family="benign", goal=d["goal"], steps=steps,
        config={"max_tokens": 400_000},
        description="Repetitive identical-shaped progress that is genuinely finishing.",
        label=Label(should_trip=False,
                    note=("Repetition WITH a shrinking remainder is a batch job. "
                          "Hard negative for the rate-of-progress detector.")),
    )


# ----------------------------------------------- drift, with actions attached
#
# Every existing drift generator emits `think()` steps ONLY. That left the whole
# suite blind to any drift logic that reads the agent's actions: the 936-scenario
# numbers were identical before and after the action-grounding change of §3.9,
# and that identity was never evidence of anything. These two families give the
# benchmark the ability to see that code path at all.

def gen_drift_with_actions(rng: random.Random, idx: int) -> Scenario:
    """Drift in BOTH narration and behaviour — the case that must still trip.

    The agent stops talking about the goal and starts calling another domain's
    tools. Action grounding must not become a blanket amnesty, so this is the
    true-positive guard on that suppression.
    """
    d = rng.choice(DOMAINS)
    other = rng.choice([x for x in DOMAINS if x["key"] != d["key"]])
    steps = _healthy_prefix(rng, d, rng.randint(1, 2))
    onset = len(steps)

    for i in range(rng.randint(3, 6)):
        ti, to = _tokens(rng)
        steps.append(think(d["off_topic"][i % len(d["off_topic"])],
                           tokens_in=ti, tokens_out=to))
        steps.append(tool(other["tools"][i % len(other["tools"])],
                          other["args"](rng), result="ok",
                          tokens_in=ti, tokens_out=to))

    return Scenario(
        id=f"gen_drift_actions_{idx:04d}",
        title=f"Drift in narration AND actions ({d['key']} -> {other['key']})",
        family="drift", goal=d["goal"], steps=steps,
        description="Agent leaves the objective in what it says and what it does.",
        label=Label(should_trip=True, detector="drift", onset_index=onset,
                    detect_by_index=onset + 6,
                    note=("Both signals agree. Guards the action-grounding "
                          "suppression against becoming a blanket amnesty.")),
        recovery_branch=_recovery_steps(rng, d),
        responds_to=_responds_to(rng),
    )


def gen_benign_narrated_failure(rng: random.Random, idx: int) -> Scenario:
    """Prose diverges while the agent keeps working on the goal. Must NOT trip.

    Taken from a real captured run, not invented. The agent searched the
    directory its goal named, four different ways, got "0 files matched" every
    time, and correctly concluded the credential did not exist. Its narration
    drifted toward the absence -- "there are no .yml files here" -- because that
    is what there was to narrate. Drift tripped, and halting that run would have
    destroyed the one result a human actually needed.

    The distinguishing evidence is that every ACTION still named the goal's own
    target. That is reproduced literally here: each call carries the goal's
    entity in its arguments, exactly as the real trace carried "./config".

    Deliberately kept SHORT, and it ends on a real advance. A longer barren
    stretch would also be a genuine stall, and a scenario that is two failures at
    once cannot tell you which detector was wrong. One phenomenon per family.
    """
    d = rng.choice(DOMAINS)
    # The goal's most distinctive word, standing in for the "./config" that the
    # real agent kept passing to its tools. Derived from the goal TEXT, not from
    # the detector's own anchor logic, which would make this circular.
    target = max((w.strip(".,:;") for w in d["goal"].split()), key=len)
    tool_name = d["tools"][0]
    misses = [
        "Nothing matched under that target; trying another pattern.",
        "That came back empty as well; widening the search.",
        "Still no matching entries of any kind there.",
    ]

    # No healthy prefix, matching the captured run: this agent explores from the
    # very first step and never gets an advance. That is not incidental. The
    # progress detector deliberately grants a run that has NOT yet advanced extra
    # room (GRACE_MULTIPLIER) so exploration is not punished, and the real trace
    # sits inside that grace. Bolting a successful step on the front would switch
    # the grace off and make this a genuine stall as well as a drift case -- two
    # failures in one scenario, which cannot tell you which detector was wrong.
    steps: list[StepSpec] = []
    for i in range(3):
        ti, to = _tokens(rng)
        steps.append(think(misses[i], tokens_in=ti, tokens_out=to))
        # Distinct arguments each time: a genuine search, not a loop.
        steps.append(tool(tool_name, {"target": target, "pattern": f"variant-{i}"},
                          result="0 matches", tokens_in=ti, tokens_out=to))
    steps.append(think(f"Confirmed: no such entry exists under {target}. Reporting it.",
                       progress=True))

    return Scenario(
        id=f"gen_narrated_failure_{idx:04d}",
        title=f"Narrating repeated tool failures ({d['key']})",
        family="benign", goal=d["goal"], steps=steps,
        description="Prose drifts toward the absence; actions never leave the goal.",
        label=Label(should_trip=False,
                    note=("REGRESSION CASE for a real false positive: drift fired "
                          "on an agent correctly establishing that its task was "
                          "impossible. Every action still named the goal's target.")),
    )


EXTRA_POSITIVES = [gen_oscillating_plan, gen_partial_progress_trap, gen_context_bloat,
                   gen_drift_with_actions]
EXTRA_NEGATIVES = [gen_benign_error_then_pivot, gen_benign_verification_heavy,
                   gen_benign_batch, gen_benign_narrated_failure]
