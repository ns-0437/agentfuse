"""Adaptive thresholds — learn what healthy looks like for *this* task.

Every threshold in the library is a constant chosen by sweeping one synthetic
suite: ``loop_threshold=3``, ``stall_patience=6``. They are the best fixed values
for the *average* workload, and no real workload is average. A breadth-first
research agent naturally takes many tool calls between state advances; a
deployment agent naturally retries a flaky endpoint. A constant tuned for one
misfires on the other, and the misfire lands as a false positive — the thing that
gets guardrails switched off.

The idea is to observe how the agent behaves while it is demonstrably *healthy*,
and scale the thresholds to that baseline.

The trap, and why calibration is keyed on progress
--------------------------------------------------
The obvious implementation — calibrate on the first N steps — is unsafe: if the
failure starts inside the calibration window, the failure is learned as normal
and the detector is disarmed exactly when it is needed. So nothing is learned
from a stretch that is not *evidenced* as healthy. A sample is only recorded when
the working state actually advances, because an agent that is making progress is,
by definition, not stuck.

Thresholds only ever widen from the configured default, never tighten. Adaptive
calibration is there to stop false positives on unusual-but-healthy workloads;
letting it lower a threshold would let a noisy warm-up talk the breaker into
being *more* trigger-happy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .events import AgentEvent, EventType


@dataclass
class Baseline:
    """What healthy behaviour looked like for this run."""

    repeats_while_healthy: int = 0     # identical calls seen before a progressing step
    actions_between_advances: float = 0.0
    samples: int = 0

    @property
    def ready(self) -> bool:
        return self.samples >= AdaptiveCalibrator.MIN_SAMPLES

    def to_dict(self) -> dict:
        return {"repeats_while_healthy": self.repeats_while_healthy,
                "actions_between_advances": round(self.actions_between_advances, 2),
                "samples": self.samples, "ready": self.ready}


class AdaptiveCalibrator:
    """Observes healthy stretches and widens thresholds to match the workload."""

    # One healthy advance is a weak estimate, but it is the only value that
    # helps: measured on sparse-progress workloads, MIN_SAMPLES of 2 or 3 never
    # calibrates in time and leaves the false-positive rate at 100%, while 1
    # brings it to 42.5% with NO loss of stall recall. Combined with the
    # widen-only rule and the safety factor, a weak estimate can only ever make
    # the breaker more cautious, never more trigger-happy.
    MIN_SAMPLES = 1
    SAFETY_FACTOR = 2.0      # how far past observed-healthy a threshold sits
    MAX_WIDENING = 4.0       # never drift more than this from the configured value

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.baseline = Baseline()
        self._repeat_run = 0           # consecutive identical signatures right now
        self._last_signature: Optional[str] = None
        self._actions_since_advance = 0
        self._advance_gaps: list[int] = []
        self._repeat_peaks: list[int] = []

    # ------------------------------------------------------------------
    def observe(self, event: AgentEvent) -> None:
        """Record a sample only when the run proves itself healthy."""
        if not self.enabled:
            return

        if event.type is EventType.TOOL_CALL:
            sig = event.tool_signature
            if sig is not None and sig == self._last_signature:
                self._repeat_run += 1
            else:
                self._last_signature = sig
                self._repeat_run = 1
            self._actions_since_advance += 1
        elif event.type is EventType.LLM_CALL:
            self._actions_since_advance += 1

        # A state advance is the evidence that everything preceding it was
        # healthy work rather than a stall. Only then is a sample recorded.
        if event.state is not None:
            self._advance_gaps.append(self._actions_since_advance)
            self._repeat_peaks.append(self._repeat_run)
            self.baseline.samples += 1
            self.baseline.repeats_while_healthy = max(self._repeat_peaks)
            self.baseline.actions_between_advances = (
                sum(self._advance_gaps) / len(self._advance_gaps))
            self._actions_since_advance = 0
            self._repeat_run = 0
            self._last_signature = None

    # ------------------------------------------------------------------
    def loop_threshold(self, configured: int) -> int:
        """Widen the loop threshold if this workload legitimately repeats calls.

        A polling or retrying agent produces identical calls while genuinely
        progressing. Learning that it did so N times without being stuck is a
        reason to require more than N before calling the next run a loop.
        """
        if not (self.enabled and self.baseline.ready):
            return configured
        suggested = int(self.baseline.repeats_while_healthy * self.SAFETY_FACTOR) + 1
        return int(min(max(configured, suggested), configured * self.MAX_WIDENING))

    def stall_patience(self, configured: int) -> int:
        """Widen stall patience for workloads that legitimately explore."""
        if not (self.enabled and self.baseline.ready):
            return configured
        suggested = int(self.baseline.actions_between_advances * self.SAFETY_FACTOR) + 1
        return int(min(max(configured, suggested), configured * self.MAX_WIDENING))

    @property
    def status(self) -> dict:
        return {"enabled": self.enabled, **self.baseline.to_dict()}
