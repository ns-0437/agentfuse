"""Audit every negative (should-not-trip) generator for the artifact section
4.3 found and fixed in exactly one place: `gen_long_sparse_benign` drew its
"varied work" from a pool of 4 tools and 4 argument dicts, so in 199 of 200
runs it emitted 3+ identical (tool, args, result) triples with no state
change -- a loop with a benign label. Section 7's action item ("audit the
remaining generators for the 4-tool/4-argument artifact") was never done.
This does it.

Structural, not detector-level, on purpose. The current suite already shows
0 errors, which only proves today's detector thresholds happen to tolerate
whatever each generator currently produces -- it says nothing about whether
the underlying trajectory is a realistic "healthy" shape or an accidentally-
narrow one that a threshold change could re-expose. So this measures the
generator's own action-space diversity directly: across many seeds, does a
healthy-labeled generator ever repeat the exact (tool, args, result) triple
3+ times with no progress between them -- the literal shape LoopDetector's
primary signal watches for, independent of whether the detector currently
catches it.

    python evals/audit_generator_entropy.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.generators import NEGATIVE_GENERATORS  # noqa: E402
from evals.generators_extra import EXTRA_NEGATIVES  # noqa: E402
from agentfuse.detectors.loop import looks_like_error  # noqa: E402

SEEDS_PER_GENERATOR = 200
# LoopDetector's own defaults (agentfuse/detectors/loop.py). An error-shaped
# repeat -- "HTTP 503" three times over -- is the signature of a healthy
# retry against a flaky endpoint and the detector deliberately tolerates
# retry_multiplier times the patience for it. Flagging that here would not
# be the gen_long_sparse_benign artifact; it would be a false alarm against
# the detector's own documented, measured design.
REPEAT_THRESHOLD = 3
RETRY_MULTIPLIER = 2


def _triple(step) -> tuple:
    return (step.tool_name, str(step.tool_args), step.result)


def worst_nonerror_repeat_run(scenario) -> int:
    """Longest run of an identical (tool, args, result) triple, with zero
    progress steps in between, EXCLUDING error-shaped results -- exactly
    LoopDetector's primary signal at its non-retry threshold, computed
    directly against the scenario rather than by running the detector, so a
    future threshold change cannot silently hide the finding.
    """
    best = 0
    run_len = 0
    last = None
    for step in scenario.steps:
        if step.kind != "tool":
            continue
        if step.progress or looks_like_error(step.result):
            run_len = 0
            last = None
            continue
        t = _triple(step)
        if t == last:
            run_len += 1
        else:
            run_len = 1
            last = t
        best = max(best, run_len)
    return best


def worst_error_repeat_run(scenario) -> int:
    """Same, but ONLY over error-shaped results, checked against the LOOSER
    retry-tolerant threshold -- a generator that repeats a failing call more
    than even a flaky-endpoint retry should get away with is its own finding.
    """
    best = 0
    run_len = 0
    last = None
    for step in scenario.steps:
        if step.kind != "tool":
            continue
        if step.progress or not looks_like_error(step.result):
            run_len = 0
            last = None
            continue
        t = _triple(step)
        if t == last:
            run_len += 1
        else:
            run_len = 1
            last = t
        best = max(best, run_len)
    return best


def audit_one(gen) -> dict:
    worst_seed = -1
    worst_run = 0
    flagged_seeds = 0
    pool: Counter = Counter()

    for seed in range(SEEDS_PER_GENERATOR):
        scenario = gen(random.Random(seed), seed)
        nonerror_run = worst_nonerror_repeat_run(scenario)
        error_run = worst_error_repeat_run(scenario)
        # A finding either way: a non-error run past the base threshold, or
        # an error-shaped run past even the retry-tolerant threshold.
        flagged = (nonerror_run >= REPEAT_THRESHOLD
                  or error_run >= REPEAT_THRESHOLD * RETRY_MULTIPLIER)
        run = max(nonerror_run, error_run)
        if run > worst_run:
            worst_run, worst_seed = run, seed
        if flagged:
            flagged_seeds += 1
        for step in scenario.steps:
            if step.kind == "tool":
                pool[(step.tool_name, str(step.tool_args))] += 1

    return {
        "name": gen.__name__,
        "worst_run": worst_run,
        "worst_seed": worst_seed,
        "flagged_seeds": flagged_seeds,
        "flagged_pct": 100 * flagged_seeds / SEEDS_PER_GENERATOR,
        "distinct_calls": len(pool),
        "total_calls": sum(pool.values()),
    }


def main() -> int:
    generators = list(NEGATIVE_GENERATORS) + list(EXTRA_NEGATIVES)
    print("=" * 88)
    print(f"GENERATOR ENTROPY AUDIT — {len(generators)} negative generators × "
          f"{SEEDS_PER_GENERATOR} seeds each")
    print(f"Flags a generator if ANY seed produces >= {REPEAT_THRESHOLD} identical")
    print("(tool,args,result) triples with zero progress between them (or"
          f" >= {REPEAT_THRESHOLD * RETRY_MULTIPLIER} for error-shaped results, matching")
    print("LoopDetector's own retry tolerance) — the artifact section 4.3 found in")
    print("gen_long_sparse_benign, checked here in every OTHER negative generator.")
    print("=" * 88)
    print(f"\n{'generator':<32}{'worst run':>10}{'flagged seeds':>16}{'distinct calls':>16}")
    print("-" * 88)

    flagged = []
    for gen in generators:
        r = audit_one(gen)
        mark = "  <-- FLAGGED" if r["flagged_seeds"] > 0 else ""
        print(f"{r['name']:<32}{r['worst_run']:>10}"
              f"{r['flagged_seeds']:>10}/{SEEDS_PER_GENERATOR:<5}"
              f"{r['distinct_calls']:>16}{mark}")
        if r["flagged_seeds"] > 0:
            flagged.append(r)

    print("-" * 88)
    if not flagged:
        print(f"\n=> CLEAN. None of the {len(generators)} negative generators reproduce "
              "the gen_long_sparse_benign artifact at LoopDetector's own threshold "
              f"({REPEAT_THRESHOLD}), across {SEEDS_PER_GENERATOR} seeds each.")
    else:
        print(f"\n=> {len(flagged)} generator(s) can produce a loop-shaped healthy "
              "scenario:")
        for r in flagged:
            print(f"   {r['name']}: seed {r['worst_seed']} hits a run of "
                  f"{r['worst_run']} identical triples ({r['flagged_pct']:.0f}% of "
                  "seeds flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
