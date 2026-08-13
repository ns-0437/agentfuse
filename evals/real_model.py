"""Real-model validation of the recovery engine.

Every recovery number this project has published came from the offline mock: a
deterministic template picked by rung, scored by a rubric written alongside those
templates. That measures the ladder, not the reasoning model, and the whole
premise of AgentFuse is that a *separate reasoning model* writes better
corrections than a fixed rule could. That premise had never been tested.

This runs the same trip snapshots through both backends and scores them with the
same rubric, so the comparison is like-for-like:

    python evals/real_model.py --n 6
    python evals/real_model.py --n 6 --base-url http://127.0.0.1:8080/v1

Point ``AGENTFUSE_LLM_BASE_URL`` at any OpenAI-compatible endpoint. A local
llama.cpp server needs no key and costs nothing:

    python -m llama_cpp.server --model models/<model>.gguf --n_ctx 4096 --port 8080

Why a sample rather than the full suite
---------------------------------------
A 3B model on CPU takes ~20s per steering call. The full 936-scenario suite fires
~440 trips, which is over three hours for one configuration and useless as a
feedback loop. This takes a **stratified sample across detectors and rungs**
instead, which is enough to compare two backends on the same inputs. It is a
paired comparison on identical snapshots, so it is far more sensitive than the
sample size suggests — but the sample is small, and any result should be read as
directional rather than settled.

What it deliberately does NOT measure
-------------------------------------
Whether the agent *obeys* the steer. That still comes from the scenario's
synthetic ``responds_to`` field, so the agent half of the loop remains simulated.
This closes the supervisor half only. Reporting it as "recovery works" would be
the same circularity this file exists to break.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.events import ExecutionSnapshot  # noqa: E402
from agentfuse.recovery import RecoveryEngine  # noqa: E402
from agentfuse.memory import JSONMemory  # noqa: E402
from evals.steering import score_steering  # noqa: E402

GOALS = [
    "Rotate the production database credential and update the app config.",
    "Reconcile the outstanding invoices against the general ledger.",
    "Summarize the Q3 revenue figures from the finance report.",
    "Deploy the release candidate to staging and verify the health checks.",
]

#: One snapshot per detector, each a real trip shape the eval produces.
TRIPS = [
    ("loop", "search_files", "trip",
     "Tool 'search_files' returned an identical result for identical arguments "
     "3x within the last 5 calls, with no state progress since step 1."),
    ("drift", None, "trip",
     "Interpreted goal has diverged from the original objective for 3 consecutive "
     "turns (similarity 0.41, trend falling)."),
    ("progress", "compute_totals", "trip",
     "14 actions (9 tool calls) since the last state advance at step 2 - the agent "
     "is busy but the task is not moving, which suggests a false premise."),
    ("rate", "process_batch", "trip",
     "11 consecutive state advances are formally identical: one reported quantity "
     "is pinned while another climbs past it, and nothing counts down."),
    ("spend", None, "critical",
     "Cumulative token budget exceeded: 512,000 of 500,000 tokens consumed."),
]


def _snapshot(goal: str, detector: str, tool, reason: str, severity: str, step: int):
    evidence = {"severity": severity}
    if tool:
        evidence["tool"] = tool
    return ExecutionSnapshot(
        step=step, original_goal=goal, current_goal=None, total_tokens=8200,
        total_cost_usd=0.0, route_history=["agent"], recent_events=[],
        trip_reason=reason, trip_detector=detector, trip_evidence=evidence)


def run(engine: RecoveryEngine, cases) -> dict:
    scores, latencies, actions, instructions = [], [], [], []
    for goal, (detector, tool, severity, reason), step in cases:
        snap = _snapshot(goal, detector, tool, reason, severity, step)
        t = time.time()
        path = engine.recover(snap)
        latencies.append(time.time() - t)
        engine.verify(path, worked=False)     # climb the ladder, as production does
        s = score_steering(path, original_goal=goal, trip_detector=detector,
                           trip_severity=severity, failing_tool=tool)
        scores.append(s)
        actions.append(path.action.value)
        instructions.append((detector, path.strategy, path.backend, path.instruction))
    checks = {c: sum(bool(getattr(s, c)) for s in scores) / len(scores)
              for c in type(scores[0]).CHECKS}
    return {
        "n": len(scores),
        "mean_quality": statistics.mean(s.score for s in scores),
        "usable_rate": sum(s.usable for s in scores) / len(scores),
        "checks": checks,
        "latency_mean": statistics.mean(latencies),
        "instructions": instructions,
        "malformed": getattr(engine, "malformed_responses", 0),
        "fell_back": sum(1 for _, _, b, _ in instructions if b == "mock"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4,
                    help="goals per detector (n x 5 steering calls per backend)")
    ap.add_argument("--base-url", default=os.getenv("AGENTFUSE_LLM_BASE_URL"))
    ap.add_argument("--model", default=os.getenv("AGENTFUSE_RECOVERY_MODEL"))
    ap.add_argument("--show", type=int, default=5, help="instructions to print")
    args = ap.parse_args()

    if not args.base_url:
        print("No endpoint configured. Set AGENTFUSE_LLM_BASE_URL or pass --base-url.\n"
              "A local llama.cpp server costs nothing:\n"
              "  python -m llama_cpp.server --model models/<m>.gguf --n_ctx 4096 --port 8080")
        return 2

    cases = [(GOALS[i % len(GOALS)], trip, 7 + i)
             for i in range(args.n) for trip in TRIPS]

    print("=" * 78)
    print(f"REAL-MODEL RECOVERY VALIDATION - {len(cases)} paired snapshots x 2 backends")
    print(f"endpoint: {args.base_url}   model: {args.model or '(endpoint default)'}")
    print("=" * 78)

    results = {}
    for label, kwargs in (("mock", {"backend": "mock"}),
                          ("real", {"backend": "real", "base_url": args.base_url,
                                    "model": args.model})):
        # A fresh memory per backend, or the second one inherits the first's
        # ladder position and is asked for different rungs on the same inputs.
        eng = RecoveryEngine(memory=JSONMemory(), **kwargs)
        if label == "real" and eng.backend != "real":
            print("\n!! could not reach the endpoint - is the server running?")
            return 2
        print(f"\nrunning {label} ...", flush=True)
        results[label] = run(eng, cases)

    m, r = results["mock"], results["real"]
    print(f"\n{'metric':<22}{'mock':>12}{'real':>12}{'delta':>12}")
    print("-" * 58)
    for key, fmt in (("mean_quality", "{:.1%}"), ("usable_rate", "{:.1%}"),
                     ("latency_mean", "{:.1f}s")):
        d = r[key] - m[key]
        print(f"{key:<22}{fmt.format(m[key]):>12}{fmt.format(r[key]):>12}"
              f"{('+' if d >= 0 else '') + fmt.format(d):>12}")
    print("-" * 58)
    for c in m["checks"]:
        d = r["checks"][c] - m["checks"][c]
        print(f"  {c:<20}{m['checks'][c]:>11.0%}{r['checks'][c]:>12.0%}"
              f"{('+' if d >= 0 else '') + format(d, '.0%'):>12}")
    print(f"\nreal backend: {r['fell_back']} of {r['n']} fell back to mock, "
          f"{r['malformed']} malformed responses")

    print(f"\n--- sample instructions (real) ---")
    for detector, strategy, backend, text in r["instructions"][: args.show]:
        print(f"\n[{detector} / {strategy} / {backend}]\n  {text[:300]}")

    verdict = ("the reasoning model beats the templates"
               if r["usable_rate"] > m["usable_rate"] else
               "the TEMPLATES are doing the work, not the model")
    print(f"\n=> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
