"""Validity checks on the benchmark itself — the numbers behind the numbers.

The eval reports recall, precision and F1 with confidence intervals. Those are
statements about the *benchmark*, and they are only statements about AgentFuse
to the extent the benchmark is sound. This module tests the benchmark.

Three questions, each of which can invalidate a headline result:

1. **Do the thresholds generalise, or are they fitted to this data?**
   Every constant in the library — ``drift_threshold``, ``loop_threshold``,
   ``MIN_SAMPLES``, ``max_recoveries``, the EMA weight — was chosen by sweeping
   one suite and is reported on that same suite. That is fitting and reporting on
   identical data.

2. **Does the complexity beat a trivial alternative?**
   Four detectors, a steering ladder, a memory and a calibrator have to earn
   their place against "stop after N steps". Without that comparison, a good F1
   says nothing about whether any of it was necessary.

3. **Are the samples independent?**
   Wilson intervals assume independent Bernoulli trials. The suite draws from a
   handful of generators sharing templates and vocabulary, so scenarios within a
   generator are highly correlated. If they are, the reported intervals are too
   narrow — and the correction is large.

    python evals/validity.py
"""

from __future__ import annotations

import collections
import os
import statistics
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse.detectors.base import Detector, Trip, Severity  # noqa: E402
from agentfuse.events import AgentEvent, EventType  # noqa: E402

from evals.generators import generate_suite  # noqa: E402
from evals.metrics import score  # noqa: E402
from evals.runner import run_suite  # noqa: E402
from evals.stats import wilson, Interval  # noqa: E402

BAR = "=" * 74


# --------------------------------------------------------------------------
# Trivial baselines the system must beat to justify itself
# --------------------------------------------------------------------------
class StepCapDetector(Detector):
    """Trip after N actions. The cheapest guardrail anyone would write first."""

    name = "loop"

    def __init__(self, cap: int = 12):
        self.cap = cap
        self.n = 0

    def inspect(self, event: AgentEvent, history) -> Optional[Trip]:
        if event.type in (EventType.TOOL_CALL, EventType.LLM_CALL):
            self.n += 1
        if self.n >= self.cap:
            self.n = 0
            return Trip(detector="loop", severity=Severity.TRIP,
                        reason=f"step cap {self.cap} reached", evidence={})
        return None

    def reset(self) -> None:
        self.n = 0


class NaiveRepeatDetector(Detector):
    """Count consecutive identical tool NAMES — no results, no state, no calibration."""

    name = "loop"

    def __init__(self, k: int = 3):
        self.k = k
        self.last: Optional[str] = None
        self.run = 0

    def inspect(self, event: AgentEvent, history) -> Optional[Trip]:
        if event.type is not EventType.TOOL_CALL:
            return None
        if event.tool_name == self.last:
            self.run += 1
        else:
            self.last, self.run = event.tool_name, 1
        if self.run >= self.k:
            self.run = 0
            return Trip(detector="loop", severity=Severity.TRIP,
                        reason=f"same tool {self.k}x", evidence={})
        return None

    def reset(self) -> None:
        self.run = 0


# --------------------------------------------------------------------------
def check_generalisation(seeds=(20260812, 777, 31337, 424242), n=40) -> list[dict]:
    """Re-run on unseen seeds. Note what this does and does not prove."""
    out = []
    for seed in seeds:
        sc = generate_suite(n_per_generator=n, seed=seed)
        m = score(run_suite(sc), {s.id: s for s in sc})
        out.append({"seed": seed, "recall": m.recall, "precision": m.precision,
                    "fpr": m.false_positive_rate, "f1": m.f1})
    return out


def check_baselines(n=40, seed=20260812) -> list[dict]:
    sc = generate_suite(n_per_generator=n, seed=seed)
    by_id = {s.id: s for s in sc}
    off = {"loop", "drift", "progress", "spend"}
    rows = [{"system": "AgentFuse (full)", **_m(score(run_suite(sc), by_id))}]
    for cap in (8, 12, 20):
        m = score(run_suite(sc, disabled=off, extra_detectors=[StepCapDetector(cap)]), by_id)
        rows.append({"system": f"step cap = {cap}", **_m(m)})
    for k in (3, 5):
        m = score(run_suite(sc, disabled=off, extra_detectors=[NaiveRepeatDetector(k)]), by_id)
        rows.append({"system": f"naive repeat k={k}", **_m(m)})
    return rows


def _m(m) -> dict:
    return {"recall": m.recall, "precision": m.precision,
            "fpr": m.false_positive_rate, "f1": m.f1}


def check_independence(n=40, seed=20260812) -> dict:
    """Estimate the design effect from between-generator variance.

    Scenarios inside one generator share a template, so they are not independent
    draws. The standard correction is the design effect
    ``DEFF = 1 + (m - 1) * ICC`` (Kish), where ICC is the intra-cluster
    correlation and ``m`` the cluster size. Effective sample size is ``n / DEFF``,
    and every interval should be computed on that, not on the raw count.
    """
    sc = generate_suite(n_per_generator=n, seed=seed)
    res = run_suite(sc)

    groups: dict[str, list[int]] = collections.defaultdict(list)
    for r in res:
        if r.should_trip:
            groups[r.scenario_id.rsplit("_", 1)[0]].append(1 if r.outcome == "TP" else 0)

    rates = {g: sum(v) / len(v) for g, v in groups.items()}
    total = sum(len(v) for v in groups.values())
    hits = sum(sum(v) for v in groups.values())
    p = hits / total if total else 0.0
    k = len(groups)
    m_avg = total / k if k else 1

    between = statistics.variance(list(rates.values())) if k > 1 else 0.0
    expected = p * (1 - p) / m_avg if m_avg else 0.0
    icc = max(0.0, (between - expected) / (p * (1 - p))) if 0 < p < 1 else 0.0
    deff = 1 + (m_avg - 1) * icc
    eff_n = total / deff if deff > 0 else total

    return {"per_generator": rates, "clusters": k, "n": total, "point": p,
            "between_var": between, "binomial_var": expected, "icc": icc,
            "deff": deff, "effective_n": eff_n,
            "reported_ci": wilson(hits, total),
            "adjusted_ci": wilson(int(p * eff_n), max(1, int(eff_n)))}


def main() -> int:
    print(BAR); print("1. GENERALISATION ACROSS SEEDS"); print(BAR)
    for row in check_generalisation():
        tag = "   <-- thresholds were tuned on this seed" if row["seed"] == 20260812 else ""
        print(f"  seed {row['seed']:<9} recall={row['recall']*100:5.1f}%  "
              f"prec={row['precision']*100:5.1f}%  FPR={row['fpr']*100:5.1f}%  "
              f"F1={row['f1']*100:5.1f}%{tag}")
    print("\n  CAVEAT: a different seed re-samples the SAME generators with the same")
    print("  templates. It shows the thresholds are not fitted to one random draw;")
    print("  it says nothing about generalising to real agents.\n")

    print(BAR); print("2. VERSUS TRIVIAL BASELINES"); print(BAR)
    for row in check_baselines():
        print(f"  {row['system']:<22} recall={row['recall']*100:5.1f}%  "
              f"prec={row['precision']*100:5.1f}%  FPR={row['fpr']*100:5.1f}%  "
              f"F1={row['f1']*100:5.1f}%")

    print("\n" + BAR); print("3. SAMPLE INDEPENDENCE"); print(BAR)
    ind = check_independence()
    for g, r in sorted(ind["per_generator"].items()):
        print(f"    {g:<22} recall={r*100:5.1f}%")
    print(f"\n  clusters                 : {ind['clusters']}")
    print(f"  intra-cluster correlation: {ind['icc']:.3f}")
    print(f"  design effect            : {ind['deff']:.2f}x")
    print(f"  nominal n={ind['n']} -> effective : {ind['effective_n']:.0f}")
    print(f"\n  reported recall CI       : {ind['reported_ci'].render()}")
    print(f"  cluster-adjusted CI      : {ind['adjusted_ci'].render()}")
    print("\n  Adding more scenarios per generator does NOT buy statistical power.")
    print("  More independent GENERATORS does.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
