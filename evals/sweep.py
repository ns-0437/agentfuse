"""Threshold sweeps — is our operating point any good, or just the first guess?

Today's defaults (``loop>=3``, ``drift<0.45``, ``stall_patience=6``) were chosen
by intuition and never tested. Reporting a single operating point hides the two
questions that matter:

  * Is there a *better* setting we're leaving on the table?
  * How sharply does behaviour change if a user nudges a knob?

A sweep answers both, and produces the precision/recall trade-off curve that
Phase 3 (adaptive thresholds) needs as its baseline. A detector whose F1 collapses
between 0.40 and 0.50 is not a detector, it's a cliff — and users will land on the
wrong side of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import Metrics, score
from .runner import DEFAULT_CONFIG, run_suite
from .schema import CostModel, DEFAULT_COST, Scenario


@dataclass
class SweepPoint:
    knob: str
    value: float
    metrics: Metrics

    def to_dict(self) -> dict:
        m = self.metrics
        return {
            "knob": self.knob, "value": self.value,
            "recall": round(m.recall, 4), "precision": round(m.precision, 4),
            "f1": round(m.f1, 4), "fpr": round(m.false_positive_rate, 4),
            "net_tokens": m.net_tokens,
        }


def _run_with_override(scenarios: list[Scenario], knob: str, value,
                       cost: CostModel) -> Metrics:
    """Replay the suite with one global knob overridden.

    Scenario-level config still wins, so budget scenarios keep their own ceilings
    — we're sweeping the *default*, which is what a user actually inherits.
    """
    original = DEFAULT_CONFIG[knob]
    DEFAULT_CONFIG[knob] = value
    try:
        by_id = {s.id: s for s in scenarios}
        results = run_suite(scenarios, cost=cost)
        return score(results, by_id, label=f"{knob}={value}")
    finally:
        DEFAULT_CONFIG[knob] = original


def sweep(scenarios: list[Scenario], knob: str, values: list,
          cost: CostModel = DEFAULT_COST) -> list[SweepPoint]:
    return [SweepPoint(knob, v, _run_with_override(scenarios, knob, v, cost))
            for v in values]


DEFAULT_GRIDS = {
    # This grid only makes sense on the LEXICAL similarity scale (~0.1-0.3);
    # the embedding backend that actually runs by default (agentfuse/embedding.py)
    # produces cosine similarities clustered around 0.5-0.9, so five of these
    # nine points (0.0-0.45) would trivially catch everything a lexical sweep
    # never would and tell you nothing about the real operating point. Found
    # while resweeping DEFAULT_THRESHOLD_EMBEDDING for real (REPORT.md section
    # 3.15) and having to build a custom embedding-scale grid to get a usable
    # curve. Both scales kept: the low end still exercises the lexical
    # fallback's own threshold (DEFAULT_THRESHOLD_LEXICAL = 0.20).
    "drift_threshold": [0.0, 0.10, 0.20, 0.30, 0.45,
                        0.55, 0.58, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68,
                        0.70, 0.72, 0.75, 0.78],
    "loop_threshold": [2, 3, 4, 5, 6, 8, 10],
    "stall_patience": [3, 4, 6, 8, 10, 14],
}


def run_all_sweeps(scenarios: list[Scenario],
                   cost: CostModel = DEFAULT_COST) -> dict[str, list[SweepPoint]]:
    return {knob: sweep(scenarios, knob, grid, cost)
            for knob, grid in DEFAULT_GRIDS.items()}


def best_point(points: list[SweepPoint], objective: str = "f1") -> SweepPoint:
    """The setting that maximises an objective — the target Phase 3 must beat."""
    key = {
        "f1": lambda p: p.metrics.f1,
        "net_tokens": lambda p: p.metrics.net_tokens,
        # Practical operating choice: maximise recall subject to a low FPR.
        "safe_recall": lambda p: (p.metrics.recall if p.metrics.false_positive_rate <= 0.10 else -1),
    }[objective]
    return max(points, key=key)


def render_sweeps(sweeps: dict[str, list[SweepPoint]]) -> str:
    out = ["", "THRESHOLD SWEEPS  (is the default operating point defensible?)",
           "-" * 78]
    for knob, points in sweeps.items():
        cur = DEFAULT_CONFIG[knob]
        out.append(f"  {knob}   (current default: {cur})")
        out.append(f"    {'value':>8}{'recall':>9}{'prec':>8}{'F1':>8}{'FPR':>8}{'net tok':>11}")
        for p in points:
            m = p.metrics
            mark = " <-" if p.value == cur else ""
            out.append(f"    {str(p.value):>8}{m.recall*100:>8.1f}%{m.precision*100:>7.1f}%"
                       f"{m.f1*100:>7.1f}%{m.false_positive_rate*100:>7.1f}%"
                       f"{m.net_tokens:>11,}{mark}")
        b_f1 = best_point(points, "f1")
        b_safe = best_point(points, "safe_recall")
        out.append(f"    best F1 at {knob}={b_f1.value} ({b_f1.metrics.f1*100:.1f}%)"
                   + (f" · best recall at FPR<=10% is {knob}={b_safe.value}"
                      if b_safe.metrics.recall > 0 else
                      "  · NO setting achieves FPR<=10%"))
        out.append("")
    return "\n".join(out)
