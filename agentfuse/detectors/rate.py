"""Rate-of-progress detector — the Zeno trap.

:class:`~agentfuse.detectors.progress.NoProgressDetector` asks a binary question:
*has the working state advanced?* That catches an agent frozen in a logic trap,
and it is structurally blind to the failure directly next to it — an agent that
advances the state on **every single step** and still never arrives.

    processed 1 of many (offset 0)
    processed 1 of many (offset 1)
    processed 1 of many (offset 2)
    ...

Every one of those is genuine progress by the binary test, so every one resets
the stall counter. The stall detector can be starved indefinitely by an agent
inching nowhere, and no amount of tuning ``stall_patience`` fixes it: the counter
never reaches 1. This was recorded as a known gap in ``baseline.json`` rather
than papered over, and it capped the progress family at 67%.

What the honest signal is
-------------------------
The tempting fix — trip when progress is "too small" — is not implementable: the
supervisor cannot see how much work remains, so it cannot measure a fraction of
it. What it *can* see is whether the run carries its own evidence of converging.

So this detector abstains by default and only fires when it has positive evidence
of non-convergence. It looks for a stretch of consecutive advances that are all
**formally identical** — same result shape once the numbers are masked out — and
that carry a quantity moving with no destination.

Two things count as evidence the work IS shrinking, and either one silences it:

  1. **A countdown.** Some number decreases across the stretch (``214 remaining``
     → ``213 remaining``). The agent is visibly consuming a backlog.
  2. **A bounded approach.** Some number is constant across the stretch and
     strictly larger than a second number climbing toward it (``processed 7 of
     240``), or the moving number is a percentage, which carries its ceiling in
     its unit. There is a denominator, and the numerator is approaching it.

Why two numbers are required, and what that concedes
----------------------------------------------------
The first version of this tripped on a single rising counter with no bound, and
the eval immediately produced the counter-example: ``gen_benign_expensive`` emits
``batch 0 done`` … ``batch 9 done``, which is a lone climbing number against no
ceiling — *character for character the same evidence* as ``processed 1 of many
(offset 9)``. It cost 44 false positives on healthy runs and dragged FPR from
8.9% to 17.0%.

That is not a tuning problem, it is an identifiability problem: **a single
climbing counter is compatible with both healthy batch work and a Zeno trap, and
the trace does not contain enough to tell them apart.** The supervisor should not
guess, so it doesn't.

What it fires on instead is the two-quantity signature: a number that is *pinned*
across the whole stretch while another climbs past it. ``processed 1 of many
(offset 12)`` says, explicitly, that the amount accomplished per step is not
growing while the attempt count is — an agent reporting its own lack of
convergence. ``batch 9 done`` says only that nine batches are done.

The concession is real and worth stating plainly: **a Zeno trap that reports
nothing but a bare cursor is not detectable here and will be missed.** That is a
narrower claim than "the gap is closed", and it is the one the evidence supports.

Other deliberate abstentions
----------------------------
**No numbers, no verdict.** A stretch reading ``milestone`` / ``milestone`` /
``milestone`` is not evidence of inching — it is a run whose results are terse. A
rate detector with no quantity to measure has nothing to say, and guessing fires
on exactly the sparse-but-healthy workloads the long-run generators represent.

**Any change of shape resets it.** An agent interleaving polling with real work
produces a varying result shape, which is what distinguishes it from one doing
the same non-thing repeatedly.

**No calibrator.** Every other detector takes a per-run baseline because it
compares against a fixed rhythm and no two workloads share one. This compares the
run against *itself* — the convergence witness is internal evidence, not a
threshold — so there is nothing for a baseline to widen.

``gen_benign_batch`` in the eval suite is the hard negative that keeps this
honest: same shape as the trap, genuinely converging, and it must never trip.
"""

from __future__ import annotations

import re
from typing import Optional

from ..events import AgentEvent
from .base import Detector, Severity, Trip

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")
#: A percentage states its own ceiling, so a rising one is a bounded approach
#: even when no literal total appears anywhere in the text.
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")


def _shape(text: str) -> str:
    """The result with every number masked — its form, independent of its values."""
    return _WHITESPACE.sub(" ", _NUMBER.sub("#", (text or "").lower())).strip()


def _numbers(text: str) -> tuple[float, ...]:
    return tuple(float(m) for m in _NUMBER.findall(text or ""))


def _has_countdown(series: list[tuple[float, ...]]) -> bool:
    """Some column strictly decreases end-to-end: a backlog being consumed."""
    for col in range(len(series[0])):
        values = [row[col] for row in series]
        if values[-1] < values[0] and all(b <= a for a, b in zip(values, values[1:])):
            return True
    return False


def _columns(series: list[tuple[float, ...]]) -> list[list[float]]:
    return [[row[col] for row in series] for col in range(len(series[0]))]


def _is_rising(values: list[float]) -> bool:
    return values[-1] > values[0] and all(a <= b for a, b in zip(values, values[1:]))


def _has_bounded_approach(series: list[tuple[float, ...]]) -> bool:
    """A constant ceiling with a climbing counter below it: ``7 of 240``."""
    columns = _columns(series)
    ceilings = [c[0] for c in columns if len(set(c)) == 1]
    if not ceilings:
        return False
    return any(_is_rising(v) and any(c > max(v) for c in ceilings) for v in columns)


def _has_pinned_and_climbing(series: list[tuple[float, ...]]) -> bool:
    """The two-quantity signature of inching.

    One value never moves across the whole stretch while another climbs past it:
    the agent is reporting, in its own output, that what it accomplishes per step
    is not growing while its attempt count is. A lone climbing counter does NOT
    qualify — see the module docstring for why that case is not identifiable.
    """
    columns = _columns(series)
    if len(columns) < 2:
        return False
    pinned = [c[0] for c in columns if len(set(c)) == 1]
    if not pinned:
        return False
    return any(_is_rising(v) and max(v) > min(pinned) for v in columns)


class RateOfProgressDetector(Detector):
    name = "rate"

    def __init__(self, patience: int = 8):
        #: Consecutive identically-shaped advances tolerated before the absence of
        #: a convergence witness is treated as evidence rather than as noise.
        self.patience = patience
        self._shape: Optional[str] = None
        self._series: list[tuple[float, ...]] = []
        self._first_step = 0
        self._last_state_hash: Optional[str] = None

    # ------------------------------------------------------------------
    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        h = event.state_hash
        if h is None or h == self._last_state_hash:
            return None  # not an advance; the stall detector owns that case
        self._last_state_hash = h

        shape = _shape(event.text)
        numbers = _numbers(event.text)

        if shape != self._shape or len(numbers) != (
                len(self._series[0]) if self._series else len(numbers)):
            self._shape = shape
            self._series = [numbers]
            self._first_step = event.step
            return None

        self._series.append(numbers)
        if len(self._series) < self.patience:
            return None

        # A rate needs a quantity. Terse results carry no evidence either way,
        # and inventing some would misread healthy sparse workloads as failures.
        moving = [col for col in range(len(numbers))
                  if len({row[col] for row in self._series}) > 1]
        if not numbers or not moving:
            return None

        # Positive evidence of inching, not merely absence of evidence of
        # progress. A lone climbing counter is not identifiable either way.
        if not _has_pinned_and_climbing(self._series):
            return None

        if (_has_countdown(self._series) or _has_bounded_approach(self._series)
                or _PERCENT.search(event.text or "")):
            return None

        return Trip(
            detector=self.name,
            severity=Severity.TRIP,
            reason=(
                f"{len(self._series)} consecutive state advances since step "
                f"{self._first_step} are formally identical (\"{shape}\"): one "
                f"reported quantity has not moved at all while another climbs past "
                f"it, and nothing is counting down or approaching a total. The "
                f"amount accomplished per step is not growing while the attempt "
                f"count is — the agent is advancing without converging."
            ),
            evidence={
                "identical_advances": len(self._series),
                "advance_shape": shape,
                "first_advance_step": self._first_step,
                "moving_values": [row[moving[0]] for row in self._series],
                "convergence_witness": None,
            },
        )

    def reset(self) -> None:
        self._shape = None
        self._series = []
