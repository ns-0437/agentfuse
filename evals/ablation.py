"""Ablation study — which detectors actually carry the signal?

Methodology borrowed from AE Studio's *Endogenous Steering Resistance* work
(ae.studio/research/esr). They established causality for a set of SAE latents by
zero-ablating them and measuring the drop in self-correction (-25%), and guarded
against coincidence with a control group of *random latents matched for
activation frequency*.

The same two moves apply cleanly to an external breaker:

  1. **Leave-one-out ablation.** Disable one detector, re-run the suite, and
     report the delta in recall / F1 / net tokens. That delta is the detector's
     causal contribution — not a claim that it "helps", a measurement of by how
     much.

  2. **Rate-matched random control.** A detector that fires at exactly the same
     per-event rate as the real system, but at random. If our detectors do not
     comfortably beat this, the headline F1 is an artefact of trip frequency
     rather than genuine discrimination.

Without (2), a system that trips constantly scores well on recall and looks
impressive. The control is what makes the number honest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from agentfuse.detectors.base import Detector, Trip, Severity
from agentfuse.events import AgentEvent, EventType

from .metrics import Metrics, score
from .runner import run_suite
from .schema import CostModel, DEFAULT_COST, Scenario

DETECTOR_NAMES = ["loop", "drift", "progress", "spend"]


class RandomControlDetector(Detector):
    """Fires at a fixed per-event probability — the matched control.

    Deliberately carries no information about the run. Its only job is to answer:
    would a detector that trips *this often*, but knows nothing, score as well as
    ours? Anything our detectors gain over this is real discrimination.
    """

    name = "random_control"

    def __init__(self, rate: float, seed: int = 1337):
        self.rate = rate
        self._rng = random.Random(seed)

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        # Only consider the event types the real detectors act on, so the
        # comparison is like-for-like.
        if event.type not in (EventType.TOOL_CALL, EventType.LLM_CALL):
            return None
        if self._rng.random() < self.rate:
            return Trip(detector=self.name, severity=Severity.TRIP,
                        reason=f"random control fired (p={self.rate:.4f})",
                        evidence={"rate": self.rate})
        return None

    def reset(self) -> None:
        pass


@dataclass
class AblationRow:
    """One row of the ablation table."""

    label: str
    metrics: Metrics
    d_recall: float = 0.0
    d_precision: float = 0.0
    d_f1: float = 0.0
    d_net_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "recall": round(self.metrics.recall, 4),
            "precision": round(self.metrics.precision, 4),
            "f1": round(self.metrics.f1, 4),
            "fpr": round(self.metrics.false_positive_rate, 4),
            "net_tokens": self.metrics.net_tokens,
            "delta_recall": round(self.d_recall, 4),
            "delta_precision": round(self.d_precision, 4),
            "delta_f1": round(self.d_f1, 4),
            "delta_net_tokens": self.d_net_tokens,
        }


def observed_event_count(scenarios: list[Scenario]) -> int:
    """How many detector-visible events the suite produces (for rate matching)."""
    n = 0
    for s in scenarios:
        for step in s.steps:
            n += 1  # TOOL_CALL or LLM_CALL
    return n


def run_ablation(scenarios: list[Scenario], cost: CostModel = DEFAULT_COST,
                 seed: int = 1337) -> tuple[Metrics, list[AblationRow]]:
    """Full system, each leave-one-out variant, and the rate-matched control."""
    by_id = {s.id: s for s in scenarios}

    # --- baseline: everything enabled ---------------------------------
    full_results = run_suite(scenarios, cost=cost)
    full = score(full_results, by_id, label="full system")
    rows: list[AblationRow] = [AblationRow(label="full system", metrics=full)]

    # --- leave-one-out -------------------------------------------------
    for name in DETECTOR_NAMES:
        res = run_suite(scenarios, disabled={name}, cost=cost)
        m = score(res, by_id, label=f"-{name}")
        rows.append(AblationRow(
            label=f"ablate {name}",
            metrics=m,
            d_recall=m.recall - full.recall,
            d_precision=m.precision - full.precision,
            d_f1=m.f1 - full.f1,
            d_net_tokens=m.net_tokens - full.net_tokens,
        ))

    # --- rate-matched random control -----------------------------------
    total_trips = sum(len(r.all_trips) for r in full_results)
    events = observed_event_count(scenarios)
    rate = (total_trips / events) if events else 0.0
    control_results = run_suite(
        scenarios,
        disabled=set(DETECTOR_NAMES),                       # all real detectors off
        extra_detectors=[RandomControlDetector(rate=rate, seed=seed)],
        cost=cost,
    )
    control = score(control_results, by_id, label="random control")
    rows.append(AblationRow(
        label=f"random control (p={rate:.4f})",
        metrics=control,
        d_recall=control.recall - full.recall,
        d_precision=control.precision - full.precision,
        d_f1=control.f1 - full.f1,
        d_net_tokens=control.net_tokens - full.net_tokens,
    ))

    return full, rows
