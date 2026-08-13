"""Rate-of-progress detector — the Zeno trap and its hard negatives.

The binary progress test asks *did the state advance?* An agent that advances on
every step and converges on none answers yes forever, so the stall counter never
reaches 1 and no threshold saves it. That was recorded in ``baseline.json`` as a
known gap and capped the progress family at 67%.

The risk in closing it is obvious and worth stating: a detector that fires on
"the same thing repeatedly" will fire on every legitimate batch job in existence,
and a guardrail that interrupts correct work gets switched off. So most of what
is asserted below is the *abstentions* — the cases where the detector must stay
silent — because those are what decide whether this is usable.

    pytest evals/test_rate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.detectors.rate import (  # noqa: E402
    RateOfProgressDetector, _shape, _has_countdown, _has_bounded_approach,
    _has_pinned_and_climbing,
)
from agentfuse.events import AgentEvent, EventType  # noqa: E402

GOAL = "Reconcile the outstanding invoices against the ledger."


def _advance(det: RateOfProgressDetector, text: str, step: int):
    """One TOOL_RESULT carrying a genuine state advance, as adapters emit it."""
    return det.inspect(AgentEvent(
        type=EventType.TOOL_RESULT, step=step, tool_name="process",
        text=text, state={"advanced_at": step, "last": text[:80]}), [])


def _run(texts, patience: int = 8):
    det = RateOfProgressDetector(patience=patience)
    trip = None
    for i, text in enumerate(texts):
        trip = trip or _advance(det, text, i + 1)
    return trip


# ------------------------------------------------------------------ the trap
def test_zeno_trap_trips():
    """The failure this exists for: advancing every step, arriving never."""
    trip = _run([f"processed 1 of many (offset {i})" for i in range(12)])
    assert trip is not None, "unbounded cursor with no remaining-count was not caught"
    assert trip.detector == "rate"
    assert trip.evidence["identical_advances"] >= 8


def test_binary_progress_detector_cannot_see_the_same_trace():
    """Documents WHY a second detector was needed rather than asserting it.

    Every step advances, so NoProgressDetector's counter is reset every step and
    its trip condition is unreachable. No value of stall_patience changes that.
    """
    from agentfuse.detectors.progress import NoProgressDetector

    det = NoProgressDetector(patience=2)
    for i in range(20):
        text = f"processed 1 of many (offset {i})"
        assert det.inspect(AgentEvent(
            type=EventType.TOOL_RESULT, step=i + 1, tool_name="process", text=text,
            state={"advanced_at": i + 1, "last": text}), []) is None


def test_it_waits_for_patience_before_firing():
    assert _run([f"row {i} handled" for i in range(6)], patience=8) is None


# ----------------------------------------------------- convergence witnesses
def test_a_countdown_silences_it():
    """`214 remaining` -> `213 remaining` is a backlog being consumed."""
    assert _run([f"deleted a record; {300 - i} remaining" for i in range(14)]) is None


def test_a_bounded_approach_silences_it():
    """`7 of 240` has a denominator, and the numerator is climbing toward it."""
    assert _run([f"processed {i + 1} of 240 records" for i in range(14)]) is None


def test_a_ceiling_below_the_counter_is_not_a_bound():
    """The constant must actually be a ceiling.

    `processed 1 of many (offset 12)` contains a constant (1) and a rising number
    (12) — but the rising one has already passed the constant, so it is not
    approaching anything. Without this the trap reads as a converging batch.
    """
    series = [(1.0, float(i)) for i in range(12)]
    assert not _has_bounded_approach(series)
    assert not _has_countdown(series)


def test_witness_helpers_agree_on_a_real_batch():
    series = [(float(i), 240.0, float(240 - i)) for i in range(1, 12)]
    assert _has_bounded_approach(series)
    assert _has_countdown(series)


# ------------------------------------------- the identifiability boundary
# The first version of this detector fired on any unbounded rising counter. The
# eval answered with 44 false positives on healthy runs and FPR 8.9% -> 17.0%.
# These three tests are that lesson, written down.
def test_a_lone_climbing_counter_is_not_enough():
    """`batch 7 done` is the SAME evidence as `processed 1 of many (offset 7)`.

    One rising number against no ceiling is compatible with healthy batch work
    and with a Zeno trap alike. The trace cannot separate them, so the detector
    must abstain rather than guess — this is `gen_benign_expensive`, and it is
    where the first version cost 26 false positives.
    """
    assert _run([f"batch {i} done" for i in range(14)]) is None


def test_a_climbing_percentage_carries_its_own_ceiling():
    """`status: RUNNING 60%` is bounded by its unit — this is `gen_benign_polling`."""
    assert _run([f"status: RUNNING {int((i + 1) / 14 * 100)}%" for i in range(14)]) is None


def test_the_two_quantity_signature_is_what_fires():
    """Pinned accomplishment, climbing attempt count — an agent reporting its own
    lack of convergence. One column alone can never satisfy it."""
    assert _has_pinned_and_climbing([(1.0, float(i)) for i in range(10)])
    assert not _has_pinned_and_climbing([(float(i),) for i in range(10)])
    # Climbing but never past the pinned value: still approaching something.
    assert not _has_pinned_and_climbing([(500.0, float(i)) for i in range(10)])


# ------------------------------------------------------- deliberate silences
def test_terse_results_are_not_evidence_of_inching():
    """No quantity, no rate. Guessing here misreads sparse healthy workloads.

    `gen_long_sparse_benign` emits a run of identical `milestone` advances; a
    detector that read repetition alone as failure would trip on all of them.
    """
    assert _run(["milestone"] * 15) is None


def test_a_change_of_shape_resets_the_run():
    """Interleaved real work is what separates polling from doing nothing.

    Modelled on `gen_long_polling_benign`: three polls, then a genuine unit of
    work, repeatedly. The poll run never reaches patience.
    """
    texts = []
    for cycle in range(6):
        texts += [f"status: RUNNING cycle {cycle} step {p}" for p in range(3)]
        texts.append("unit done")
    assert _run(texts) is None


def test_repeated_state_is_left_to_the_stall_detector():
    """An unchanged state hash is not an advance, so this detector abstains."""
    det = RateOfProgressDetector(patience=3)
    for i in range(10):
        assert det.inspect(AgentEvent(
            type=EventType.TOOL_RESULT, step=i + 1, tool_name="process",
            text="no change", state={"frozen": True}), []) is None


def test_events_without_state_are_ignored():
    det = RateOfProgressDetector(patience=2)
    for i in range(10):
        assert det.inspect(AgentEvent(type=EventType.LLM_CALL, step=i + 1,
                                      text=f"thinking about item {i}"), []) is None


def test_reset_clears_the_run():
    det = RateOfProgressDetector(patience=4)
    for i in range(3):
        _advance(det, f"offset {i}", i + 1)
    det.reset()
    assert det._series == [] and det._shape is None


def test_shape_masks_values_but_not_structure():
    assert _shape("Processed 12 of 240") == _shape("processed 3 of 240")
    assert _shape("processed 3 of 240") != _shape("processed 3 records")


# -------------------------------------------------- wired into the monitor
def test_monitor_runs_the_rate_detector_by_default():
    from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer

    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                tracer=Tracer(None, False))
    assert any(d.name == "rate" for d in mon.detectors)


def test_a_supplied_detector_list_is_not_silently_extended():
    """Appending a sensor the caller did not ask for overrides an explicit choice."""
    from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer
    from agentfuse.detectors import LoopDetector

    mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, echo=False),
                                detectors=[LoopDetector()], tracer=Tracer(None, False))
    assert [d.name for d in mon.detectors] == ["loop"]


def test_rate_detection_can_be_switched_off():
    from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer

    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=GOAL, echo=False, rate_patience=None),
        tracer=Tracer(None, False))
    assert not any(d.name == "rate" for d in mon.detectors)


# --------------------------------------------- measured against the suite
def test_the_generated_trap_and_its_hard_negative_are_separated():
    """The end-to-end claim, on the generators themselves rather than on prose.

    These two families produce nearly the same trace: a long run of advances
    whose results differ only in their numbers. If the detector cannot tell them
    apart it is reading repetition as failure, and the hard negative is the only
    thing that would have caught that.
    """
    import random
    from evals.generators_extra import gen_partial_progress_trap, gen_benign_batch

    def replay(scenario):
        det = RateOfProgressDetector(patience=8)
        trip = None
        for i, step in enumerate(scenario.steps):
            if step.kind == "tool" and step.progress:
                trip = trip or _advance(det, str(step.result), i + 1)
        return trip

    for seed in range(8):
        rng = random.Random(seed)
        assert replay(gen_partial_progress_trap(rng, seed)) is not None, \
            f"missed the Zeno trap on seed {seed}"
        assert replay(gen_benign_batch(random.Random(seed), seed)) is None, \
            f"false positive on a converging batch job, seed {seed}"
