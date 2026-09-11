"""Does the escalation ladder climbing past re-anchor actually help?

REPORT.md section 3.24 found the single highest-value open question this
project has: every real trace ever captured before 2026-08-24 tested obedience
to `re-anchor`, repeated, because a bug (section 3.22) marked a rung as having
"worked" whenever ANY tool call landed afterward -- including the exact same
failing call the agent was steered away from. With that fixed, a rung is only
credited as having worked when the run genuinely advances, so a real second
trip on the same failure shape should climb to `alternate-action` instead of
re-issuing `re-anchor`. Whether that climb actually helps -- not just whether
it happens -- is untested. This script is the first attempt at measuring it.

Method
------
Run real agent tasks against a local model with the breaker ARMED
(`max_recoveries=6`, enough to reach all four steerable rungs), backend="mock"
(the deterministic ladder -- section 8.1 already settled that a real reasoning
model loses to it, so this isolates the ladder-mechanism question from that
already-answered one). Read the resulting trace's `strategy` field (fixed this
round -- see agentfuse/tracer.py) off every recovery record to see which rungs
actually fired, and whether the run's outcome differs between tasks that never
climbed and tasks that did.

Deliberately small by default (see --tasks): this machine has hard-restarted
under sustained live-capture load before. Keep batches small; --n_threads 6.

    python -m llama_cpp.server --model models/qwen2.5-3b-instruct-q4_k_m.gguf \\
        --n_ctx 8192 --port 8080 --n_gpu_layers 0 --n_threads 6
    python evals/measure_escalation.py --base-url http://127.0.0.1:8080/v1

Deliberately NOT --chat_format chatml-function-calling: that handler cannot
terminate once handed a finished answer (see CLAUDE.md). Native template +
ToolCallShim recovers the tool calls it leaves as unparsed text.

What a negative result looks like, so it cannot be explained away afterwards:
if every task's steer sequence is a single rung (never climbs because the
first steer always works, or climbs but every arm fails regardless of rung),
that is reported exactly as clearly as a positive finding. This script does
not exist to prove the ladder works.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer  # noqa: E402
from agentfuse.adapters.openai_sdk import guarded_tool_loop  # noqa: E402
from agentfuse.recovery import RecoveryEngine  # noqa: E402
from agentfuse.strategies import STEERABLE  # noqa: E402
from evals.measure_resistance import TASKS  # noqa: E402

OUT = ROOT / "evals" / "captured" / "escalation"

#: The "empty" world variants make the task's premise genuinely false (the
#: file the agent is told to find does not exist), which is what should force
#: repeated failure -- and therefore repeated trips against the SAME failure
#: shape, which is exactly the condition that should climb the ladder.
DEFAULT_TASKS = [name for name, (_, world) in TASKS.items() if world == "empty"]


def run_task(name: str, base_url: str, model: str, max_turns: int,
            intervention: str = "system") -> Path:
    prompt, world = TASKS[name]
    from evals.capture_real_runs import make_router
    router, _ = make_router(world)
    OUT.mkdir(parents=True, exist_ok=True)
    # Timestamped, not `{name}.jsonl`: an earlier version of this script
    # overwrote a capture that showed a distinct failure mode (hallucinated
    # tool invention, REPORT.md 3.33) the moment the same task name was run
    # again. Real-model sampling is not deterministic -- two runs of the same
    # task are two different data points, not one being refreshed.
    trace = OUT / f"{name}.{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    from openai import OpenAI

    from evals.toolcall_shim import ToolCallShim
    client = ToolCallShim(OpenAI(base_url=base_url, api_key="not-needed"))

    from evals.capture_real_runs import TOOL_SCHEMA
    # loop_threshold=2 / stall_patience=3 are deliberately more sensitive than
    # the library defaults (3 / 6). This script exists to observe whether the
    # ladder CLIMBS across repeated trips within a bounded turn budget, not to
    # simulate production thresholds -- a real model's sampling is not
    # deterministic (the same task tripped 0 or 1 times run to run at the
    # default thresholds), and a single trip within max_turns can never show
    # climbing at all.
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=prompt, echo=False, loop_threshold=2,
                      stall_patience=3, max_recoveries=6, jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False),
        # Deterministic ladder, not a reasoning model: section 8.1 already
        # settled that a real reasoning model loses to the templates at every
        # size tested, so this isolates the mechanism (does climbing rungs
        # help) from that already-answered question (who writes the rung).
        recovery=RecoveryEngine(backend="mock"))
    # intervention="system" (append, don't restart) rather than the production
    # default "rerun": sections 3.5-3.6 measured "rerun" completing 6 of 8 real
    # tasks precisely because discarding the failing turns gives the model a
    # clean shot at succeeding on the very next attempt -- which means it
    # rarely NEEDS a second correction, and this experiment could never
    # observe the ladder climb if it used that default. "system" was
    # measured at 0% task completion in that same study, which is exactly
    # the condition (repeated failure) this experiment needs to exercise
    # rungs 2+ at all.
    guarded_tool_loop(client, model=model, system_prompt=prompt, user_input=prompt,
                      tools=TOOL_SCHEMA, tool_router=router, max_turns=max_turns,
                      monitor=mon, intervention=intervention)
    return trace


def analyse(trace: Path) -> dict:
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    strategies = [r["strategy"] for r in records if r.get("kind") == "recovery"]
    detectors = [r["detector"] for r in records if r.get("kind") == "trip"]
    summary: dict = next((r for r in records if r.get("kind") == "summary"), {})
    climbed = bool(strategies) and (len(set(strategies)) > 1 or strategies[0] != STEERABLE[0])
    return {
        "trips": len(detectors),
        "detectors_seen": sorted(set(detectors)),
        "rungs_fired": strategies,
        "max_rung_index": (max(STEERABLE.index(s) for s in strategies)
                           if strategies else -1),
        "climbed_past_rung_1": climbed,
        "outcome": summary.get("status", "?"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.getenv("AGENTFUSE_LLM_BASE_URL"))
    ap.add_argument("--model", default=os.getenv("AGENTFUSE_MODEL", "local"))
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--tasks", default=None,
                    help="comma-separated subset (default: the 'empty'-world "
                         "tasks most likely to force repeated failure)")
    ap.add_argument("--intervention", default="system",
                    choices=["system", "user", "rerun", "drop_tool"],
                    help="delivery mechanism (default: system -- the "
                         "worst-performing, most repeat-failure-prone arm "
                         "from sections 3.5-3.6, deliberately chosen so this "
                         "experiment can observe rungs 2+ at all)")
    args = ap.parse_args()
    if not args.base_url:
        print("Set --base-url. A local llama.cpp server costs nothing.")
        return 2

    names = args.tasks.split(",") if args.tasks else DEFAULT_TASKS
    print("=" * 78)
    print("ESCALATION LADDER — does climbing past re-anchor help, with the fix active?")
    print("=" * 78)

    results = {}
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} ...", flush=True)
        try:
            trace = run_task(name, args.base_url, args.model, args.max_turns,
                            intervention=args.intervention)
        except Exception as e:                       # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {str(e)[:120]}")
            continue
        result = analyse(trace)
        results[name] = result
        print(f"    trips={result['trips']} detectors={result['detectors_seen']} "
              f"rungs={result['rungs_fired']} outcome={result['outcome']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n",
                                      encoding="utf-8")

    print("\n" + "-" * 78)
    climbed = [n for n, r in results.items() if r["climbed_past_rung_1"]]
    never_climbed = [n for n, r in results.items() if r["rungs_fired"] and not r["climbed_past_rung_1"]]
    no_trip = [n for n, r in results.items() if not r["rungs_fired"]]
    print(f"Tasks that climbed past re-anchor:    {climbed or 'NONE'}")
    print(f"Tasks that tripped but never climbed: {never_climbed or 'none'}")
    print(f"Tasks that never tripped at all:      {no_trip or 'none'}")
    all_detectors = sorted({d for r in results.values() for d in r["detectors_seen"]})
    print(f"Detector types actually exercised:    {all_detectors or 'NONE'}")
    if climbed:
        outcomes = {n: results[n]["outcome"] for n in climbed}
        print(f"Outcomes for climbed tasks:            {outcomes}")
    print("\nSee evals/captured/escalation/results.json for full detail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
