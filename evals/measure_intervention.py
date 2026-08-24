"""Is a system-message injection simply too weak an intervention?

`evals/measure_resistance.py` measured the thing this project exists to do and
found it does not work: **41 corrections injected into a real Qwen2.5-7B, 1
obeyed, 40 ignored.** The corrections were the deterministic ladder templates —
the ones our own rubric scores at 100% "usable" — so the failure was not the
wording (REPORT.md section 3.5).

Every experiment before that varied the *text* of the correction. None varied how
it was *delivered*. And there is a loose thread already in the repository: the one
captured run that self-healed came through the AgentKit adapter, which differs
from `guarded_tool_loop` in two ways at once —

    guarded_tool_loop : steer as a SYSTEM message, conversation CONTINUES
    agentkit_hooks    : steer as a USER message, in-flight run ABORTED

— so the two are confounded in the only positive evidence we have. This separates
them, and adds a fifth option that is not a request at all.

    system      append as system, continue                (the 2.4% baseline)
    user        append as user, continue                  (isolates the ROLE)
    rerun       restart from the original prompt + steer  (isolates the MECHANISM)
    drop_tool   as user, and the failing tool is REMOVED from the schema

The hypothesis behind `rerun` is specific. A model that has just emitted the same
tool call four times is being asked to contradict several rounds of its own
visible, committed behaviour. Appending a correction argues with that history;
discarding it removes the thing being argued with.

The hypothesis behind `drop_tool` is blunter: stop asking. If the tool is not in
the schema, the loop is not available to repeat.

    python -m llama_cpp.server --model models/<m>.gguf --n_ctx 8192 --port 8080 \\
        --chat_format chatml-function-calling
    python evals/measure_intervention.py --base-url http://127.0.0.1:8080/v1

Compliance is read off BEHAVIOUR — did the next tool call differ from the one
that tripped the breaker — exactly as in measure_resistance.py, so the arms are
comparable with the existing baseline rather than scored on a new rubric.

If every arm sits near 2.4%, then steering-by-message does not work on this model
in any form, and the honest product is detection plus escalation, with the
"self-healing" claim removed from the README. That outcome is as reportable as
any other and is stated here in advance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer  # noqa: E402
from agentfuse.adapters.openai_sdk import guarded_tool_loop  # noqa: E402
from evals.capture_real_runs import TOOL_SCHEMA, make_router  # noqa: E402
from evals.measure_resistance import TASKS  # noqa: E402
from evals.steering_compliance import compliance_from_trace  # noqa: E402

OUT = ROOT / "evals" / "captured" / "intervention"
ARMS = ("system", "user", "rerun", "drop_tool")


def run_arm(arm: str, base_url: str, model: str, names: list[str],
            max_turns: int) -> dict:
    complied = total = 0
    per_task = {}
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="not-needed")

    for name in names:
        prompt, world = TASKS[name]
        router, _ = make_router(world)
        trace = OUT / arm / f"{name}.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)

        tools = list(TOOL_SCHEMA)
        if arm == "drop_tool":
            # The blunt arm: the schema itself is the constraint. Removing the
            # tool the agent is stuck on means the loop is not merely
            # discouraged, it is unavailable.
            tools = [t for t in TOOL_SCHEMA
                     if t["function"]["name"] != "search_files"]

        mon = CircuitBreakerMonitor(
            MonitorConfig(original_goal=prompt, echo=False, loop_threshold=3,
                          max_recoveries=6, jsonl_path=str(trace)),
            tracer=Tracer(jsonl_path=str(trace), echo=False))
        try:
            guarded_tool_loop(client, model=model, system_prompt=prompt,
                              user_input=prompt, tools=tools, tool_router=router,
                              max_turns=max_turns, monitor=mon,
                              intervention=arm)
        except Exception as e:                       # noqa: BLE001
            print(f"      {name}: FAILED {type(e).__name__}: {str(e)[:90]}")
            continue
        c, t = compliance_from_trace(trace)
        per_task[name] = (c, t)
        complied += c
        total += t
    return {"arm": arm, "complied": complied, "total": total, "per_task": per_task}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.getenv("AGENTFUSE_LLM_BASE_URL"))
    ap.add_argument("--model", default=os.getenv("AGENTFUSE_RECOVERY_MODEL", "local"))
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--tasks", type=int, default=8, help="tasks per arm")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    if not args.base_url:
        print("Set --base-url. A local llama.cpp server costs nothing.")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    names = list(TASKS)[: args.tasks]
    arms = [a for a in args.arms.split(",") if a in ARMS]

    print("=" * 78)
    print("INTERVENTION MECHANISM — does HOW a correction is delivered change "
          "whether it lands?")
    print(f"{len(arms)} arms x {len(names)} tasks   baseline to beat: 2.4% (1/41)")
    print("=" * 78)

    results = []
    for arm in arms:
        print(f"\n-- {arm} ...", flush=True)
        r = run_arm(arm, args.base_url, args.model, names, args.max_turns)
        results.append(r)
        rate = r["complied"] / r["total"] if r["total"] else 0.0
        lo, hi = wilson(r["complied"], r["total"])
        print(f"   complied {r['complied']}/{r['total']} = {rate:.1%}  "
              f"[95% CI {lo:.1%}-{hi:.1%}]")

    (OUT / "intervention.json").write_text(json.dumps(results, indent=2) + "\n",
                                           encoding="utf-8")

    print("\n" + "-" * 78)
    print(f"  {'arm':<12}{'complied':>12}{'rate':>10}{'95% CI':>20}")
    best = None
    for r in results:
        if not r["total"]:
            print(f"  {r['arm']:<12}{'no steers':>12}")
            continue
        rate = r["complied"] / r["total"]
        lo, hi = wilson(r["complied"], r["total"])
        print(f"  {r['arm']:<12}{r['complied']:>5}/{r['total']:<6}{rate:>9.1%}"
              f"{f'{lo:.1%}-{hi:.1%}':>20}")
        if best is None or rate > best[1]:
            best = (r["arm"], rate, lo, hi)
    print("-" * 78)

    if best is None:
        print("\n=> NO STEERS OBSERVED — nothing to compare.")
        return 0

    arm, rate, lo, hi = best
    # 0.024 is the measured baseline from measure_resistance.py (1/41). An arm
    # only counts as better if its lower bound clears it — anything else is a
    # difference this sample size cannot resolve.
    if lo > 0.024:
        print(f"\n=> DELIVERY MATTERS — '{arm}' lands {rate:.1%} of corrections "
              f"(95% CI lower bound {lo:.1%}, above the 2.4% baseline). The "
              f"failure was the MECHANISM, not the wording, and the adapter "
              f"default should change.")
    else:
        print(f"\n=> DELIVERY DOES NOT RESCUE IT — the best arm '{arm}' reaches "
              f"{rate:.1%} but its interval [{lo:.1%}-{hi:.1%}] includes the 2.4% "
              f"baseline. Steering by message does not work on this model in any "
              f"form tested. The honest product is detection plus escalation, and "
              f"the 'self-healing' claim should come out of the README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
