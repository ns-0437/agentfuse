"""Statistics for the eval harness — because a rate without an interval is a rumor.

With nine positive scenarios, a recall of 77.8% carries a 95% confidence interval
of roughly [45%, 94%]. Reporting the point estimate alone would imply a precision
the sample size cannot support. Everything here exists so the report can state
how *uncertain* each number is, and so "we beat the random control" is a claim
backed by a test rather than a single lucky seed.

Pure stdlib: Wilson score intervals for proportions, bootstrap CIs for derived
quantities, and a permutation test for comparing two systems.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval and its sample size."""

    point: float
    low: float
    high: float
    n: int

    def pct(self, digits: int = 1) -> str:
        if self.n == 0:
            return "  n/a"
        return f"{self.point * 100:.{digits}f}%"

    def ci_pct(self, digits: int = 1) -> str:
        if self.n == 0:
            return "n/a"
        return f"[{self.low * 100:.{digits}f}–{self.high * 100:.{digits}f}]"

    def render(self) -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.pct()} {self.ci_pct()} n={self.n}"

    @property
    def width(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict:
        return {"point": round(self.point, 4), "ci_low": round(self.low, 4),
                "ci_high": round(self.high, 4), "n": self.n}


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sanely at small n and at rates near 0 or 1 — which is exactly the
    regime a young benchmark lives in.
    """
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), n)


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float],
                 iterations: int = 2000, seed: int = 7, alpha: float = 0.05) -> Interval:
    """Percentile bootstrap CI for an arbitrary statistic (e.g. mean tokens saved)."""
    if not values:
        return Interval(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(values)
    point = stat(values)
    draws = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(stat(sample))
    draws.sort()
    lo = draws[int((alpha / 2) * iterations)]
    hi = draws[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return Interval(point, lo, hi, n)


@dataclass
class ComparisonTest:
    """Result of comparing our system against a control."""

    observed_diff: float
    p_value: float
    control_mean: float
    control_sd: float
    n_trials: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def render(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "NOT significant"
        return (f"Δ={self.observed_diff:+.3f}  p={self.p_value:.4f}  "
                f"control={self.control_mean:.3f}±{self.control_sd:.3f} "
                f"(n={self.n_trials} seeds)  → {verdict}")

    def to_dict(self) -> dict:
        return {"observed_diff": round(self.observed_diff, 4),
                "p_value": round(self.p_value, 5),
                "control_mean": round(self.control_mean, 4),
                "control_sd": round(self.control_sd, 4),
                "n_trials": self.n_trials,
                "significant": self.significant}


def compare_to_control(system_score: float, control_scores: Sequence[float]) -> ComparisonTest:
    """Empirical test: how often does a rate-matched random control match us?

    p is the fraction of control runs scoring at least as high as the real system.
    Running the control across many seeds is what turns "we beat random" from an
    anecdote into a measurement — AE Studio's ESR work makes the same move with
    frequency-matched random latents.
    """
    if not control_scores:
        return ComparisonTest(0.0, 1.0, 0.0, 0.0, 0)
    n = len(control_scores)
    mean = sum(control_scores) / n
    var = sum((s - mean) ** 2 for s in control_scores) / n if n > 1 else 0.0
    sd = math.sqrt(var)
    at_least = sum(1 for s in control_scores if s >= system_score)
    # add-one smoothing: with k seeds you can never honestly claim p < 1/(k+1)
    p = (at_least + 1) / (n + 1)
    return ComparisonTest(system_score - mean, p, mean, sd, n)


# --------------------------------------------------------------------------
# Cluster-aware intervals
# --------------------------------------------------------------------------
def design_effect(clusters: dict[str, list[int]]) -> tuple[float, float]:
    """Return ``(deff, icc)`` for outcomes grouped by cluster.

    Wilson assumes independent trials. Benchmark scenarios generated from a
    shared template are not independent: within one generator they behave almost
    identically, so the information content of 40 scenarios is closer to that of
    one. The standard correction is Kish's design effect,
    ``DEFF = 1 + (m - 1) * ICC``, where ICC is the intra-cluster correlation and
    ``m`` the mean cluster size. Effective sample size is ``n / DEFF``.
    """
    sizes = [len(v) for v in clusters.values() if v]
    if len(sizes) < 2:
        return 1.0, 0.0
    n = sum(sizes)
    hits = sum(sum(v) for v in clusters.values())
    p = hits / n if n else 0.0
    if p <= 0.0 or p >= 1.0:
        return 1.0, 0.0

    m_avg = n / len(sizes)
    rates = [sum(v) / len(v) for v in clusters.values() if v]
    between = statistics.variance(rates)
    expected = p * (1 - p) / m_avg          # variance if draws were independent
    icc = max(0.0, (between - expected) / (p * (1 - p)))
    return 1.0 + (m_avg - 1) * icc, icc


def clustered_wilson(clusters: dict[str, list[int]], z: float = Z95) -> Interval:
    """Wilson interval computed on the EFFECTIVE sample size.

    Reporting the nominal interval over-states precision whenever samples are
    clustered — on this benchmark by roughly sevenfold.
    """
    n = sum(len(v) for v in clusters.values())
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    hits = sum(sum(v) for v in clusters.values())
    p = hits / n
    deff, _ = design_effect(clusters)
    eff_n = max(1, int(round(n / deff)))
    return wilson(int(round(p * eff_n)), eff_n, z)
