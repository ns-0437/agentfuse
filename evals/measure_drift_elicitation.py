"""Can a real agent be made to drift? A controlled attempt, across model sizes.

Every real trace this project has captured is a **loop**. Drift — the failure
mode the detector has been rewritten twice this week — has never once been
observed in real behaviour, so every check on it is synthetic or a single
hand-labelled trace. That is the weakest claim in the project.

Attempt 1 failed. Three `hijack` tasks dangled an attractive unrelated job in a
tool result ("the marketing site CSS is broken") behind an explicit instruction.
A Qwen2.5-7B ignored the bait all three times and finished the task. The obvious
reading — "use a bigger model" — is probably backwards: a *more* capable
instruction-follower is more likely to stay on an explicit instruction, not less.

So attempt 2 changes the stimulus rather than the model, and then varies the
model to see whether drift is a capability gradient at all:

  * the goal is **vague** ("investigate and resolve it"), with no fixed endpoint
  * the world hands back a **chain**, each link closing the current thread and
    opening a plausible next one, in the authoritative voice of the system
  * nothing ever tells the agent to abandon the goal; the goal simply stops
    being the thing in front of it

Ground truth comes from the WORLD, not from a detector: the chain is ordered, so
how many links appear in the transcript records how far the agent wandered.
Grading drift with the drift detector's own notion of drift would be circular.

    python evals/measure_drift_elicitation.py \\
        --model-a http://127.0.0.1:8081/v1 --name-a qwen2.5-3b \\
        --model-b http://127.0.0.1:8080/v1 --name-b qwen2.5-7b

A negative result is still a result, and is reported as one: if neither model
drifts, the honest conclusion is that this stimulus does not elicit drift, and
the drift detector remains unevidenced against real behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer  # noqa: E402
from agentfuse.adapters.openai_sdk import guarded_tool_loop  # noqa: E402
from evals.capture_real_runs import TOOL_SCHEMA  # noqa: E402
from evals.real_suite import (  # noqa: E402
    _CASCADE_DRIFT_DEPTH, _CASCADE_MARKERS, TASKS, classify, make_router)

OUT = ROOT / "evals" / "captured" / "drift_elicitation"
CASCADE = [t for t in TASKS if TASKS[t][1] == "cascade"]
HIJACK = [t for t in TASKS if TASKS[t][1] == "hijack"]


def run_one(task: str, base_url: str, model: str, tag: str, max_turns: int) -> dict:
    prompt, world, _ = TASKS[task]
    router, _ = make_router(world)
    trace = OUT / f"{tag}__{task}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    if trace.exists():
        trace.unlink()

    from openai import OpenAI

    from evals.toolcall_shim import ToolCallShim

    client = ToolCallShim(OpenAI(base_url=base_url, api_key="not-needed"))
    # Disarmed: a supervised trace has already been altered by the detectors it
    # would then be used to judge.
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=prompt, echo=False, loop_threshold=10_000,
                      stall_patience=10_000, rate_patience=None,
                      drift_threshold=0.0, max_tokens=10_000_000,
                      max_recoveries=0, adaptive=False, jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False))
    guarded_tool_loop(client, model=model, system_prompt=prompt, user_input=prompt,
                      tools=TOOL_SCHEMA, tool_router=router, max_turns=max_turns,
                      monitor=mon)

    obs = classify(trace, world=world)
    results = [json.loads(l).get("text") or "" for l in
               trace.read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l).get("type") == "tool_result"]
    depth = sum(1 for m in _CASCADE_MARKERS if any(m in r for r in results))
    return {"task": task, "world": world, "depth": depth,
            "drifted": bool(obs["should_trip"]) and world == "cascade",
            "calls": obs["tool_calls"], "status": obs["status"],
            "reason": obs["reason"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--name-a", default="model-a")
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--name-b", default="model-b")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--include-hijack", action="store_true",
                    help="also re-run attempt 1's tasks for comparison")
    args = ap.parse_args()

    tasks = CASCADE + (HIJACK if args.include_hijack else [])
    arms = [(args.name_a, args.model_a)]
    if args.model_b:
        arms.append((args.name_b, args.model_b))

    print("=" * 78)
    print("DRIFT ELICITATION — does a vague goal plus a plausible chain drift?")
    print("=" * 78)

    summary: dict[str, list[dict]] = {}
    for name, url in arms:
        print(f"\n--- {name} ---", flush=True)
        rows = []
        for task in tasks:
            try:
                r = run_one(task, url, args.model, name, args.max_turns)
            except Exception as e:                      # noqa: BLE001
                print(f"  {task:<20} FAILED {type(e).__name__}: {str(e)[:60]}")
                continue
            rows.append(r)
            mark = "DRIFTED" if r["drifted"] else "on task"
            print(f"  {task:<20} {mark:<8} chain depth {r['depth']}/"
                  f"{len(_CASCADE_MARKERS)}  calls={r['calls']} ({r['status']})")
        summary[name] = rows

    print("\n" + "-" * 78)
    for name, rows in summary.items():
        casc = [r for r in rows if r["world"] == "cascade"]
        drifted = sum(1 for r in casc if r["drifted"])
        deepest = max((r["depth"] for r in casc), default=0)
        print(f"  {name:<14} drifted {drifted}/{len(casc)}   deepest chain "
              f"depth {deepest}/{len(_CASCADE_MARKERS)} "
              f"(drift labelled at >= {_CASCADE_DRIFT_DEPTH})")

    any_drift = any(r["drifted"] for rows in summary.values() for r in rows)
    print()
    if any_drift:
        print("=> Real drift captured. These traces are the first real evidence")
        print("   the drift detector has ever been checked against.")
    else:
        print("=> NO DRIFT, in any arm. The honest conclusion is that this")
        print("   stimulus does not elicit drift in these models, and the drift")
        print("   detector remains unevidenced against real behaviour. Do not")
        print("   report the detector as validated on real traces.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
