"""Score the detectors against real captured runs, negatives included.

Replays every trace in `evals/captured/suite/` through a real
`CircuitBreakerMonitor` at production defaults and compares the trip against the
behavioural label from `real_suite.classify()` — a label computed from the
agent's own actions, never from the breaker's output.

Reports precision, recall and FPR next to the synthetic suite. The comparison is
the point: if the numbers match, the synthetic suite was at least *representative*
even though it is saturated. If real precision is materially worse, the synthetic
suite has been flattering the detectors and every headline figure in the README
needs the real number beside it.

Two failure modes of this script are worth naming rather than hiding:

1. **Small n.** A dozen runs cannot resolve a 0.6% FPR. Wilson intervals are
   printed so the width is visible instead of implied, and no claim is made that
   the interval will not support.
2. **Trace-length confound.** Real runs are shorter than synthetic scenarios, and
   a shorter trace has fewer chances to trip. Median lengths for both classes are
   printed so a difference in FPR cannot be quietly attributed to detector
   quality when it is really a difference in exposure.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.runner import run_scenario  # noqa: E402
from evals.schema import Label  # noqa: E402
from evals.trace_import import scenario_from_trace  # noqa: E402

SUITE = ROOT / "evals" / "captured" / "suite"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Clamped: floating error puts the lower bound a hair below zero when k=0,
    # and an interval printed as "-0.0%" invites the reader to distrust every
    # other number on the page.
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def replay(spec: dict) -> dict:
    """Score one real trace through the SAME runner the synthetic suite uses.

    Using `run_scenario` rather than a bespoke loop is deliberate: a second
    replay path would quietly diverge from the one the headline numbers come
    from, and then the comparison below would be measuring the harness instead
    of the detectors.
    """
    scen = scenario_from_trace(
        SUITE.parent / spec["trace"],
        label=Label(**spec["label"]),
        goal=spec.get("goal"),
        scenario_id=spec["id"],
        family=spec.get("world", "captured"),
    )
    res = run_scenario(scen)
    return {"tripped": res.tripped, "detector": res.trip_detector,
            "n_events": len(scen.steps), "steps_late": res.steps_late}


def main() -> int:
    labels_path = SUITE / "labels.json"
    if not labels_path.exists():
        print("No suite captured yet. Run: python evals/real_suite.py --base-url ...")
        return 2
    specs = json.loads(labels_path.read_text(encoding="utf-8"))

    rows = []
    for spec in specs:
        out = replay(spec)
        rows.append({**spec, **out, "truth": spec["label"]["should_trip"]})

    tp = sum(1 for r in rows if r["truth"] and r["tripped"])
    fp = sum(1 for r in rows if not r["truth"] and r["tripped"])
    fn = sum(1 for r in rows if r["truth"] and not r["tripped"])
    tn = sum(1 for r in rows if not r["truth"] and not r["tripped"])

    print("=" * 78)
    print("REAL-TRACE SUITE — scored against a behavioural oracle")
    print("=" * 78)
    print(f"\n{'run':<22}{'truth':<10}{'breaker':<10}{'detector':<14}{'events':>7}")
    print("-" * 78)
    for r in rows:
        mark = "  ok" if r["truth"] == r["tripped"] else "  <-- MISS"
        print(f"{r['id']:<22}{('FAIL' if r['truth'] else 'healthy'):<10}"
              f"{('trip' if r['tripped'] else 'silent'):<10}"
              f"{str(r['detector'] or '-'):<14}{r['n_events']:>7}{mark}")

    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")

    print("\n" + "-" * 78)
    print(f"  n={n}   TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if tp + fp:
        lo, hi = wilson(tp, tp + fp)
        print(f"  precision {prec:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    if tp + fn:
        lo, hi = wilson(tp, tp + fn)
        print(f"  recall    {rec:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    if fp + tn:
        lo, hi = wilson(fp, fp + tn)
        print(f"  FPR       {fpr:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]   "
              f"(on {fp + tn} real healthy runs)")

    # Exposure confound: a shorter trace has fewer chances to trip, so a low FPR
    # may be a length artifact rather than a detector property.
    hl = [r["n_events"] for r in rows if not r["truth"]]
    fl = [r["n_events"] for r in rows if r["truth"]]
    if hl and fl:
        print(f"\n  median events: healthy {statistics.median(hl):.0f}  "
              f"vs failing {statistics.median(fl):.0f}")
        if statistics.median(hl) < statistics.median(fl) / 2:
            print("  ** healthy runs are much shorter — a low FPR here is partly an")
            print("     exposure artifact, not purely detector precision.")

    print("\n  synthetic suite for comparison: precision 99.4%, recall 97.8%, "
          "FPR 0.6%\n  (1016 scenarios, 23 families)")
    if fp + tn and fp + tn < 30:
        lo, hi = wilson(fp, fp + tn)
        print(f"\n  HONEST LIMIT: {fp + tn} healthy runs cannot resolve a 0.6% FPR — "
              f"the\n  interval [{lo:.1%}, {hi:.1%}] spans it either way. This suite "
              "detects gross\n  regressions, not small ones. More real healthy runs "
              "are the only fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
