"""Scoring for the AgentFuse eval suite.

Definitions (stated explicitly, because a benchmark whose metrics are vague is
just a vibe with decimal places):

  TP  scenario should trip AND the breaker tripped
  FN  scenario should trip AND the breaker never tripped
  FP  scenario should NOT trip BUT the breaker tripped
  TN  scenario should NOT trip AND the breaker stayed quiet

  precision = TP / (TP + FP)        do we trust a trip when we see one?
  recall    = TP / (TP + FN)        do we catch real failures?
  F1        = harmonic mean
  FPR       = FP / (FP + TN)        how often do we halt a healthy run?

Two things are scored *separately* from detection, on purpose:

  * **Attribution accuracy** — we caught it, but did the right detector fire?
    A loop caught by the spend guard is a real catch with a misleading diagnosis,
    and the steering advice that follows will be wrong. Folding this into recall
    would hide it.
  * **Timeliness** — steps between the failure's onset and the trip. Catching a
    loop on step 40 is technically a TP and practically a failure.

Net token benefit is the bottom line: savings from halting early, minus the cost
of supervision, minus the cost of false alarms. If that number is negative, the
breaker is a liability no matter how good the F1 is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import mean
from typing import Optional

from .schema import ScenarioResult
from .stats import Interval, wilson, clustered_wilson, design_effect


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass
class Metrics:
    """Aggregate scores over a set of scenario results."""

    label: str = "full"
    n: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    attribution_correct: int = 0
    attribution_total: int = 0

    detected_late: int = 0            # caught, but after detect_by_index
    detected_premature: int = 0       # fired BEFORE the failure onset — see note below
    mean_steps_late: Optional[float] = None
    known_gap_misses: int = 0         # FNs we already documented as gaps

    tokens_saved: int = 0
    supervision_cost: int = 0
    tokens_spent: int = 0
    wasted_if_unsupervised: int = 0

    # -- recovery (the half of the product that was never measured) ----
    recovery_eligible: int = 0     # tripped positives that declare a corrected path
    recovery_attempted: int = 0    # steering was produced
    recovery_usable: int = 0       # steering passed the quality rubric
    recovery_succeeded: int = 0    # the run actually got back on track
    recovery_escalated: int = 0    # correctly handed to a human instead
    steering_quality_sum: float = 0.0
    steering_quality_n: int = 0

    # Outcomes grouped by generator, so intervals can account for the fact that
    # scenarios sharing a template are not independent samples.
    clusters_recall: dict = field(default_factory=dict)
    clusters_fpr: dict = field(default_factory=dict)

    per_family: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)

    # -- core rates -----------------------------------------------------
    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _safe_div(2 * p * r, p + r)

    @property
    def false_positive_rate(self) -> float:
        return _safe_div(self.fp, self.fp + self.tn)

    @property
    def attribution_accuracy(self) -> float:
        return _safe_div(self.attribution_correct, self.attribution_total)

    @property
    def net_tokens(self) -> int:
        return self.tokens_saved - self.supervision_cost

    @property
    def roi(self) -> float:
        """Tokens saved per token of supervision spent. >1.0 means it pays for itself."""
        return _safe_div(self.tokens_saved, self.supervision_cost)

    # -- interval estimates ---------------------------------------------
    # Every rate below is a binomial proportion, so it gets a Wilson interval.
    # At small n these intervals are embarrassingly wide, which is the point:
    # the report should show how little a 16-scenario suite actually proves.
    def recall_ci(self) -> Interval:
        return wilson(self.tp, self.tp + self.fn)

    def precision_ci(self) -> Interval:
        return wilson(self.tp, self.tp + self.fp)

    def fpr_ci(self) -> Interval:
        return wilson(self.fp, self.fp + self.tn)

    def attribution_ci(self) -> Interval:
        return wilson(self.attribution_correct, self.attribution_total)

    # -- cluster-adjusted intervals ------------------------------------
    # The nominal intervals above assume independent trials. They are not:
    # scenarios from one generator share a template and behave almost
    # identically, so 40 of them carry closer to one sample's worth of
    # information. Measured design effect on this suite is ~17x, which makes the
    # nominal intervals roughly seven times too narrow. These are the ones to
    # quote.
    def recall_ci_clustered(self) -> Interval:
        return clustered_wilson(self.clusters_recall)

    def fpr_ci_clustered(self) -> Interval:
        return clustered_wilson(self.clusters_fpr)

    @property
    def design_effect(self) -> float:
        return design_effect(self.clusters_recall)[0]

    @property
    def intra_cluster_correlation(self) -> float:
        return design_effect(self.clusters_recall)[1]

    # -- recovery rates -------------------------------------------------
    @property
    def recovery_rate(self) -> float:
        """Of the failures we caught, how many did we actually get back on track?"""
        return _safe_div(self.recovery_succeeded, self.recovery_eligible)

    def recovery_ci(self) -> Interval:
        return wilson(self.recovery_succeeded, self.recovery_eligible)

    @property
    def steering_usable_rate(self) -> float:
        return _safe_div(self.recovery_usable, self.recovery_attempted)

    @property
    def mean_steering_quality(self) -> float:
        return _safe_div(self.steering_quality_sum, self.steering_quality_n)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "attribution_accuracy": round(self.attribution_accuracy, 4),
            "net_tokens": self.net_tokens,
            "roi": round(self.roi, 3),
            "recall_ci": self.recall_ci().to_dict(),
            "precision_ci": self.precision_ci().to_dict(),
            "fpr_ci": self.fpr_ci().to_dict(),
            "attribution_ci": self.attribution_ci().to_dict(),
            "recall_ci_clustered": self.recall_ci_clustered().to_dict(),
            "fpr_ci_clustered": self.fpr_ci_clustered().to_dict(),
            "design_effect": round(self.design_effect, 2),
            "intra_cluster_correlation": round(self.intra_cluster_correlation, 3),
            "recovery_rate": round(self.recovery_rate, 4),
            "recovery_ci": self.recovery_ci().to_dict(),
            "steering_usable_rate": round(self.steering_usable_rate, 4),
            "mean_steering_quality": round(self.mean_steering_quality, 4),
        })
        return d


def score(results: list[ScenarioResult], scenarios_by_id: dict, label: str = "full") -> Metrics:
    """Turn raw scenario results into aggregate metrics."""
    m = Metrics(label=label, n=len(results))
    late_deltas: list[int] = []

    for r in results:
        outcome = r.outcome
        setattr(m, outcome.lower(), getattr(m, outcome.lower()) + 1)

        m.tokens_saved += r.tokens_saved
        m.supervision_cost += r.supervision_cost
        m.tokens_spent += r.tokens_spent

        sc = scenarios_by_id.get(r.scenario_id)
        if sc is not None:
            m.wasted_if_unsupervised += sc.wasted_tokens_if_undetected

        # attribution (positives we actually caught)
        if outcome == "TP":
            m.attribution_total += 1
            if r.attribution_correct:
                m.attribution_correct += 1
            # timeliness
            if sc is not None and sc.label.detect_by_index is not None \
                    and r.trip_step_index is not None \
                    and r.trip_step_index > sc.label.detect_by_index:
                m.detected_late += 1
            # A trip that lands before the failure has even begun is not a real
            # catch: it fired on the healthy prefix, which means the same trigger
            # would fire on a healthy run. Counted separately so it can't be
            # laundered into the recall number.
            if r.steps_late is not None and r.steps_late < 0:
                m.detected_premature += 1
            if r.steps_late is not None:
                late_deltas.append(r.steps_late)

        if outcome == "FN" and r.known_gap:
            m.known_gap_misses += 1

        # per-family rollup
        fam = m.per_family.setdefault(r.family, {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        fam[outcome] += 1

        # cluster = the generator that produced this scenario
        cluster = r.scenario_id.rsplit("_", 1)[0]
        if r.should_trip:
            m.clusters_recall.setdefault(cluster, []).append(1 if outcome == "TP" else 0)
        else:
            m.clusters_fpr.setdefault(cluster, []).append(1 if outcome == "FP" else 0)

        # recovery accounting
        rec = getattr(r, "recovery", None)
        if rec is not None:
            m.recovery_eligible += 1
            if rec.attempted:
                m.recovery_attempted += 1
            if rec.usable:
                m.recovery_usable += 1
            if rec.recovered:
                m.recovery_succeeded += 1
            if rec.escalated:
                m.recovery_escalated += 1
            for sc in rec.steering_scores:
                m.steering_quality_sum += sc.score
                m.steering_quality_n += 1

        if outcome in ("FP", "FN"):
            m.failures.append({
                "scenario": r.scenario_id,
                "outcome": outcome,
                "family": r.family,
                "known_gap": r.known_gap,
                "expected_detector": r.expected_detector,
                "actual_detector": r.trip_detector,
                "trip_step_index": r.trip_step_index,
            })

    if late_deltas:
        m.mean_steps_late = round(mean(late_deltas), 2)
    return m


def family_rates(m: Metrics) -> dict:
    """Per-family precision/recall, to expose which detector is the weak link."""
    out = {}
    for fam, c in sorted(m.per_family.items()):
        out[fam] = {
            "recall": round(_safe_div(c["TP"], c["TP"] + c["FN"]), 3),
            "precision": round(_safe_div(c["TP"], c["TP"] + c["FP"]), 3),
            "counts": c,
        }
    return out
