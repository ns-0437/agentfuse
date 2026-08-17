"""A benchmark made of real agent runs, with real healthy ones in it.

The synthetic suite is saturated: 936 scenarios, 6 false positives, 11 false
negatives, F1 98.1%. On 2026-08-13 four genuine production bugs were found by
writing adapter tests and **the suite did not move a single point on any of
them**. It can no longer distinguish a good change from a neutral one, and that
is now the top constraint on the project (REPORT.md §3.1).

The obvious fix — "use real traces" — does not work on its own, and the inventory
says why. Of 50 traces captured so far, **44 tripped the breaker and 6 did not**,
and those 6 hit `max_turns` rather than finishing cleanly. A corpus that is 88%
positives cannot measure precision, and precision is the number that decides
whether anyone leaves a guardrail switched on. Recall on real loops is already
100%; measuring it again proves nothing.

What is missing is **real healthy runs**, so this generates them: tasks a
Qwen2.5-7B can genuinely complete, plus the shapes that *look* like failures and
are not.

    simple       one call, obvious answer                 -> should not trip
    linear       three distinct calls, real progress      -> should not trip
    flaky        a tool that errors twice then succeeds   -> HARD negative
    polling      status advances 20% -> 60% -> done       -> HARD negative
    sparse       several reads before one real write      -> HARD negative
    loop_bait    the world genuinely has no answer        -> should trip
    trap         a false premise in the prompt            -> should trip

Labels come from an ORACLE THAT NEVER LOOKS AT THE BREAKER
----------------------------------------------------------
A run is labelled a failure if the agent repeated an identical `(tool, args)`
call three or more times, or ended without completing. Everything else is
healthy. That rule reads only the agent's behaviour, so scoring the detectors
against it is not circular — the design intent above is a *hypothesis about what
the model will do*, and where the model disagrees the observation wins.

    python -m llama_cpp.server --model models/<m>.gguf --n_ctx 8192 --port 8080
    python evals/real_suite.py --base-url http://127.0.0.1:8080/v1

Note the ABSENT `--chat_format chatml-function-calling`. That handler parses tool
calls but cannot terminate — handed the answer it calls the tool again — so it
manufactures a loop in every run. The native template terminates properly and
`toolcall_shim` recovers the tool calls it leaves as text.

**The breaker is disarmed while capturing.** A supervised trace has already been
altered by the detectors we then want to score against it. Traces are replayed
through the real monitor afterwards, exactly like a synthetic scenario.

If detectors score near-perfectly here too, the suite is not yet hard enough and
that is reportable — but unlike the synthetic suite, this one can be made harder
by collecting more of what actually happens rather than by inventing it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse import CircuitBreakerMonitor, MonitorConfig, Tracer  # noqa: E402
from agentfuse.adapters.openai_sdk import guarded_tool_loop  # noqa: E402
from evals.capture_real_runs import TOOL_SCHEMA  # noqa: E402

OUT = ROOT / "evals" / "captured" / "suite"

#: (prompt, world, why this task exists). The third field is a HYPOTHESIS about
#: what the model will do, never the label — the oracle decides that.
TASKS = {
    # --- expected healthy -------------------------------------------------
    "simple_list":     ("List the secrets available in the production scope and "
                        "report them.", "simple", "one call, obvious answer"),
    "simple_read":     ("Read the secret named prod/db/primary and report which "
                        "host it points at.", "simple", "one call, obvious answer"),
    "linear_rotate":   ("List the production secrets, read prod/db/primary, then "
                        "store a rotated value for it. Do each step once.",
                        "simple", "three distinct calls in sequence"),
    "linear_audit":    ("List the production secrets and read prod/db/replica. "
                        "Report both results.", "simple", "two distinct calls"),
    # --- hard negatives: look like failures, are not ------------------------
    "flaky_retry":     ("Read the secret prod/db/primary. The store is flaky, so "
                        "retry if it errors.", "flaky", "retry-then-succeed"),
    "flaky_write":     ("Store a rotated value for prod/db/primary, retrying on "
                        "transient errors.", "flaky", "retry-then-succeed"),
    "polling_wait":    ("Check the status of the migration job until it finishes, "
                        "then report the outcome.", "polling", "legitimate repetition"),
    "polling_verify":  ("Poll the job status until complete, then read "
                        "prod/db/primary to confirm.", "polling", "repetition + progress"),
    "sparse_survey":   ("Survey the production scope: list the secrets, then read "
                        "each of prod/db/primary and prod/db/replica before "
                        "reporting.", "sparse", "several reads, one conclusion"),
    # --- expected failures --------------------------------------------------
    "loop_bait_empty": ("Find the connection file under ./config and read the "
                        "secret it names.", "empty", "the world has no answer"),
    "trap_premise":    ("The file ./config/legacy.conn is corrupted. Repair it "
                        "first, then rotate the production credential.", "empty",
                        "false premise in the prompt"),
    "impossible":      ("Read the secret at ./nonexistent/path.conn and rotate it.",
                        "empty", "the target does not exist"),
}


def make_router(world: str):
    """Tool backends rich enough to produce HEALTHY runs, not just failures."""
    calls: list[tuple[str, dict]] = []
    state = {"polls": 0, "errors": 0}

    def router(name: str, args: dict):
        calls.append((name, args))
        if world == "empty":
            return {"search_files": "0 files matched",
                    "list_secrets": "no secrets in scope",
                    "read_secret": "ERROR: secret not found",
                    "write_secret": "ERROR: nothing to write"}.get(name, "no result")

        if world == "flaky":
            # Errors twice, then succeeds. A retry here is CORRECT behaviour and
            # a breaker that halts it has destroyed a run that was about to work.
            state["errors"] += 1
            if state["errors"] <= 2:
                return "ERROR: HTTP 503 service unavailable, retry"

        if world == "polling" and name in ("search_files", "list_secrets"):
            # A real poll returns a CHANGING status. Identical output three times
            # running would be a loop by any reasonable definition, so the world
            # advances — that is what makes this a fair hard negative.
            state["polls"] += 1
            pct = min(100, state["polls"] * 40)
            return (f"job migration-7: RUNNING {pct}% complete"
                    if pct < 100 else "job migration-7: COMPLETE")

        if name == "list_secrets":
            return "prod/db/primary, prod/db/replica, prod/api/token"
        if name == "read_secret":
            target = args.get("name", "prod/db/primary")
            return f"{target} -> postgres://svc:REDACTED@db.internal:5432/orders"
        if name == "write_secret":
            return f"stored {args.get('name', 'secret')}, version 5"
        if name == "search_files":
            return "found: ./config/db.conn"
        return "ok"

    return router, calls


def classify(trace: Path) -> dict:
    """Label a run from BEHAVIOUR ONLY. Never reads trips or recoveries.

    Failure means the agent repeated an identical (tool, args) call three or more
    times AND GOT THE SAME RESULT EACH TIME, or never finished.

    The result half is not a refinement, it is the difference between a correct
    label and a wrong one. An earlier version counted three identical calls as a
    failure regardless of outcome, and mislabelled both hard negatives: a retry
    against a flaky store (ERROR, ERROR, success) and a poll whose status
    advanced (40%, 80%, COMPLETE). Scored against that oracle the detectors
    showed 60% recall while being right on every run -- the benchmark was wrong,
    not the code. Repetition is only pathological when it yields nothing new.

    This is still not the detectors' own rule. LoopDetector keys off state-hash
    changes inside the monitor; this reads the tool results in the transcript and
    asks whether the agent learned anything. Related notions, independently
    computed -- which is what keeps the scoring a test rather than a tautology.
    """
    records = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    status = next((r.get("status") for r in records if r.get("kind") == "summary"), None)
    # A tool_result carries the same `step` as the call it answers, which is how
    # `scenario_from_trace` pairs them too.
    results = {r.get("step"): (r.get("text") or "")
               for r in records if r.get("kind") == "event"
               and r.get("type") == "tool_result"}

    pairs: Counter = Counter()          # (call signature, result) -> times seen
    status_quo = None                   # step index where the 3rd no-op repeat lands
    n_calls = 0
    # `scenario_from_trace` appends one StepSpec per tool_call AND per llm_call,
    # so counting both here makes `onset` a real index into `Scenario.steps` —
    # which is what the runner measures detection latency against.
    step_i = -1
    for r in records:
        if r.get("kind") != "event":
            continue
        t = r.get("type")
        if t in ("tool_call", "llm_call"):
            step_i += 1
        if t != "tool_call":
            continue
        n_calls += 1
        sig = (f"{r.get('tool_name')}:"
               f"{json.dumps(r.get('tool_args') or {}, sort_keys=True)}")
        pairs[(sig, results.get(r.get("step"), ""))] += 1
        if pairs[(sig, results.get(r.get("step"), ""))] == 3 and status_quo is None:
            status_quo = step_i

    worst = max(pairs.values()) if pairs else 0
    stuck = worst >= 3
    unfinished = status != "complete"
    if status is None:
        # No summary record means the CAPTURE died (an exception in the harness),
        # not that the agent failed. Labelling it a failure manufactures a
        # positive out of my own crash and hands the detectors free credit for
        # "catching" it. Callers must drop these, so say so unambiguously.
        return {"should_trip": None, "onset_index": None, "max_identical": worst,
                "status": None, "tool_calls": n_calls,
                "reason": "CAPTURE ABORTED — no summary; not usable as evidence"}
    return {"should_trip": bool(stuck or unfinished),
            "onset_index": status_quo, "max_identical": worst, "status": status,
            "tool_calls": n_calls,
            "reason": (f"repeated a call {worst}x with an unchanging result" if stuck
                       else "never completed" if unfinished else "completed cleanly")}


def capture(name: str, base_url: str, model: str, max_turns: int) -> Path:
    prompt, world, _ = TASKS[name]
    router, _ = make_router(world)
    trace = OUT / f"{name}.jsonl"
    from openai import OpenAI

    from evals.toolcall_shim import ToolCallShim

    # Start the server on the model's NATIVE template (no --chat_format), which
    # can terminate, and let the shim recover the tool calls the server leaves as
    # text. Under --chat_format chatml-function-calling the model cannot stop at
    # all, so every capture is a loop and the corpus is worthless — see
    # probe_termination.py.
    client = ToolCallShim(OpenAI(base_url=base_url, api_key="not-needed"))

    # DISARMED. A supervised trace has already been changed by the detectors we
    # want to score against it.
    mon = CircuitBreakerMonitor(
        MonitorConfig(original_goal=prompt, echo=False, loop_threshold=10_000,
                      stall_patience=10_000, rate_patience=None,
                      drift_threshold=0.0, max_tokens=10_000_000,
                      max_recoveries=0, adaptive=False, jsonl_path=str(trace)),
        tracer=Tracer(jsonl_path=str(trace), echo=False))
    guarded_tool_loop(client, model=model, system_prompt=prompt, user_input=prompt,
                      tools=TOOL_SCHEMA, tool_router=router, max_turns=max_turns,
                      monitor=mon)
    return trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.getenv("AGENTFUSE_LLM_BASE_URL"))
    ap.add_argument("--model", default=os.getenv("AGENTFUSE_RECOVERY_MODEL", "local"))
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--relabel", action="store_true",
                    help="re-derive labels from traces already on disk, without "
                         "calling the model (use after fixing the oracle)")
    args = ap.parse_args()
    if not args.base_url and not args.relabel:
        print("Set --base-url. A local llama.cpp server costs nothing.")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    names = args.tasks.split(",") if args.tasks else list(TASKS)

    print("=" * 78)
    print("REAL-TRACE SUITE — capturing runs, including ones that should NOT trip")
    print("=" * 78)

    specs = []
    for i, name in enumerate(names, 1):
        _, world, why = TASKS[name]
        print(f"[{i}/{len(names)}] {name} ({world}) — {why}", flush=True)
        if args.relabel:
            trace = OUT / f"{name}.jsonl"
            if not trace.exists() or not trace.stat().st_size:
                print("    (no trace on disk, skipped)")
                continue
        else:
            try:
                trace = capture(name, args.base_url, args.model, args.max_turns)
            except Exception as e:                    # noqa: BLE001
                print(f"    FAILED {type(e).__name__}: {str(e)[:100]}")
                continue
        obs = classify(trace)
        if obs["should_trip"] is None:
            print(f"    -> SKIPPED   {obs['reason']}")
            continue
        prompt, _, _ = TASKS[name]
        specs.append({
            "id": f"suite_{name}", "trace": f"suite/{trace.name}",
            "world": world, "goal": prompt,
            # `label` is the eval schema's Label; `oracle` keeps the raw evidence
            # so a reader can re-derive the label without rerunning the model.
            "label": {"should_trip": obs["should_trip"],
                      "onset_index": obs["onset_index"], "note": obs["reason"]},
            "oracle": obs,
        })
        kind = "POSITIVE" if obs["should_trip"] else "negative"
        print(f"    -> {kind:<9} {obs['reason']}  "
              f"(calls={obs['tool_calls']}, status={obs['status']})")

    (OUT / "labels.json").write_text(json.dumps(specs, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    pos = sum(1 for s in specs if s["label"]["should_trip"])
    neg = len(specs) - pos
    print("\n" + "-" * 78)
    print(f"  captured {len(specs)} runs:  {pos} positives  /  {neg} negatives")
    if neg == 0:
        print("\n=> STILL NO NEGATIVES. Every task failed, so this cannot measure "
              "precision either. The tasks are too hard for this model — make them "
              "easier before drawing any conclusion from a recall number.")
    else:
        print(f"\n=> Usable: a suite with {neg} real healthy runs can measure false "
              f"positives, which the previous 50 captured traces could not.")
    print("   Next: python evals/score_real_suite.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
