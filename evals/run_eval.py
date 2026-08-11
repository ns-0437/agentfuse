"""CLI for the AgentFuse eval suite.

    python evals/run_eval.py                 # full suite + ablation, console report
    python evals/run_eval.py --json          # also write evals/results/
    python evals/run_eval.py --no-ablation   # detection metrics only (fast)
    python evals/run_eval.py --scenario loop_exact_repeat --verbose

Exit code is non-zero when a *regression* is found — a false positive, or a miss
that isn't a documented known gap — so this drops straight into CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.ablation import run_ablation  # noqa: E402
from evals.metrics import score  # noqa: E402
from evals.report import console_report, write_artifacts  # noqa: E402
from evals.runner import run_suite, run_scenario  # noqa: E402
from evals.scenarios import ALL_SCENARIOS, by_id  # noqa: E402
from evals.schema import CostModel  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="AgentFuse detection eval")
    p.add_argument("--scenario", help="run a single scenario by id")
    p.add_argument("--no-ablation", action="store_true", help="skip the ablation study")
    p.add_argument("--json", action="store_true", help="write results.json + REPORT.md")
    p.add_argument("--out", default="evals/results", help="artifact directory")
    p.add_argument("--recovery-cost", type=int, default=1500,
                   help="tokens charged per steering call")
    p.add_argument("--verbose", action="store_true", help="print per-trip detail")
    args = p.parse_args()

    cost = CostModel(recovery_call_tokens=args.recovery_cost,
                     false_positive_penalty=args.recovery_cost)

    # -- single scenario ------------------------------------------------
    if args.scenario:
        sc = by_id(args.scenario)
        r = run_scenario(sc, cost=cost)
        print(f"\n{sc.id} — {sc.title}")
        print(f"  family        {sc.family}")
        print(f"  expected      {'TRIP' if sc.label.should_trip else 'stay quiet'}"
              f"{' (known gap)' if sc.label.known_gap else ''}")
        print(f"  outcome       {r.outcome}")
        print(f"  detector      {r.trip_detector or '—'}")
        print(f"  trip at step  {r.trip_step_index if r.trip_step_index is not None else '—'}"
              f" of {len(sc.steps)}")
        print(f"  tokens saved  {r.tokens_saved:,}   supervision {r.supervision_cost:,}"
              f"   net {r.net_tokens:,}")
        if sc.label.note:
            print(f"  note          {sc.label.note}")
        if args.verbose:
            for t in r.all_trips:
                print(f"    · step {t['step']:>2} [{t['detector']}/{t['severity']}] {t['reason']}")
        return 0 if r.outcome in ("TP", "TN") or sc.label.known_gap else 1

    # -- full suite -----------------------------------------------------
    scenarios = ALL_SCENARIOS
    by_id_map = {s.id: s for s in scenarios}

    if args.no_ablation:
        results = run_suite(scenarios, cost=cost)
        full = score(results, by_id_map)
        rows = []
    else:
        full, rows = run_ablation(scenarios, cost=cost)
        results = run_suite(scenarios, cost=cost)

    print(console_report(full, rows, results, cost))

    if args.json:
        out_dir = Path(args.out)
        write_artifacts(full, rows, results, cost, out_dir)
        print(f"  artifacts written to {out_dir}/results.json and {out_dir}/REPORT.md\n")

    # CI gate: false positives and undocumented misses are regressions.
    regressions = [f for f in full.failures
                   if f["outcome"] == "FP" or not f["known_gap"]]
    if regressions:
        print(f"  {len(regressions)} regression(s) — see ERRORS TO FIX above.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
