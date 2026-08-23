# AgentFuse — CLAUDE.md

A logical circuit breaker for long-running LLM agents. It watches the
telemetry every agent framework already emits (tool calls, graph routes,
state deltas, token spend), trips when a long-horizon failure mode crosses a
threshold, and steers the agent back or escalates to a human. The thing
judging the run is never the thing performing it.

Repo: https://github.com/ns-0437/agentfuse (public) · Dashboard: https://ns-0437.github.io/agentfuse/

## Current state (2026-08-23)

Synthetic suite: 1018 scenarios, **0 errors**, precision/recall/F1 all 100.0%,
FPR 0.0% — re-verified with a full ablation, 25-seed significance test, 4-seed
regeneration check, and richer trivial-baselines table. **This is not "solved"**:
every one of the 14 errors it used to carry turned out to be the benchmark's own
construction bug (REPORT.md 3.13–3.14), not a detector gap, and a 0-error score
on a suite one person wrote is evidence of internal consistency, not correctness
against the world. Real-trace suite: 23 runs (21 healthy), 100%/100%/0% FPR —
still far too small to trust on its own. See the improvement list kept in memory
(ask "what needs improving") for the prioritized set of genuinely open gaps —
the biggest is the pure-reasoning drift-grounding gap (REPORT.md 3.13), which
this session tried and rejected 4 different fixes for.

## Code structure

```
agentfuse/
  agentfuse/                  core library (stdlib-only, no dependencies)
    monitor.py                 CircuitBreakerMonitor, MonitorConfig, Directive — the engine
    events.py                  AgentEvent, EventType, SeenStateTracker (bounded-window progress check)
    detectors/
      base.py                   Detector / Trip contract
      loop.py                   repeated (tool, args, result) — the classic infinite-tool-loop
      progress.py                NoProgressDetector — busy but not advancing (logic traps)
      rate.py                    RateOfProgressDetector — the "Zeno" trap (progress that never finishes)
      spend.py                   token/cost ceiling and burst-rate breach
      drift.py                   goal drift — embedding/lexical similarity + action grounding
    confidence.py               Tier-1 signal: token logprobs vs the run's own baseline (opt-in, ships OFF)
    calibration.py              adaptive per-run threshold calibration
    recovery.py + strategies.py  the steering ladder (recovery reasoning model, separate from the agent)
    memory.py                   JSONMemory (append-only) / QdrantMemory (vector) for steering history
    checkpoint.py, notify.py, pricing.py, redact.py, sanitize.py, tracer.py
    adapters/
      openai_sdk.py              plain OpenAI SDK adapter (guarded_tool_loop) — used by all real captures
      agentkit.py, agentkit_hooks.py  real OpenAI AgentKit RunHooks integration
      langgraph.py                LangGraph adapter

  evals/                      the benchmark + real-trace pipeline
    generators.py               23-family synthetic scenario generator (1018 scenarios, seed=20260812)
    schema.py                   Scenario / StepSpec / Label — the eval data model
    runner.py                   replays a Scenario through a real CircuitBreakerMonitor
    run_eval.py                 main synthetic-suite entry point (--generated N --no-ablation / --sweep)
    trace_import.py             converts a captured real JSONL trace into a Scenario
    real_suite.py                captures real Qwen runs + a behavioural oracle (never reads the breaker)
    score_real_suite.py          scores captured/suite/ against the oracle labels
    capture_real_runs.py         earlier single-trace capture tool (superseded by real_suite.py)
    toolcall_shim.py             recovers tool calls llama.cpp's native template leaves as text
    measure_*.py                 one-off measurement scripts (drift elicitation, intervention, etc.)
    captured/                    committed real traces + hand/oracle-written labels (*.json + *.jsonl)
      suite/                      real_suite.py's own corpus + labels.json
    test_*.py                   303 tests total, `pytest evals/ -q`

  models/                     local GGUF weights (qwen2.5-3b, qwen2.5-7b) — no API key needed
  dashboard/                  static HTML dashboard, published via GitHub Pages
  REPORT.md                   the primary research log — numbered findings, every measured result, every
                               failed experiment. This is the source of truth, more than README.md.
  README.md                   headline numbers + narrative summary for a first-time reader
```

## Points to remember

1. **Be brutally honest; report negative results as results.** Never claim a
   number that wasn't measured. If a fix looks good, check it against BOTH
   corpora (synthetic + real) before believing it — this project has shipped
   the same class of bug (a detector reset that never fires) independently in
   three different detectors, each caught by a benchmark control, not by
   reasoning.
2. **Never trust a "fix" without measuring the failure mode it claims to
   close, then re-measuring after.** Every real fix this project has made
   followed the same shape: reproduce the failure directly (usually with a
   throwaway script against the detector or a live monitor, not just the eval
   harness), confirm the mechanism, try the obvious fix, and — critically —
   check it against the *other* corpus before committing. Two fixes were
   rejected specifically because they were measured to backfire on
   `gen_driftsub` before they reached a commit.
3. **Before writing "impossible" or "cannot" in a report, check the actual
   data.** A first draft of REPORT.md section 3.13 claimed a driftsub failure
   mode was "structurally impossible" — checked against instrumented data
   before it shipped, and it was wrong (some 0-tail draws did trip, by
   domain-specific luck). Corrected in place. Overclaiming certainty is a
   mistake this project treats the same as an unmeasured number.
4. **No API key, no credits: local models only** (`models/qwen2.5-3b` and
   `-7b` GGUF via `llama.cpp.server`). `llama_cpp_python` is installed
   **globally**, not in `.venv` — use the system `python`, not
   `.venv/Scripts/python.exe`, when running the server or real-trace capture
   scripts.
5. **Never start `llama.cpp.server` with `--chat_format
   chatml-function-calling`.** That handler cannot terminate — handed the
   finished answer, it calls the tool again — which silently corrupted the
   entire first real-trace corpus (all 50 traces) before this was found.
   Run the model's native template and let `evals/toolcall_shim.py` recover
   tool calls from the text it leaves unparsed.
6. **Laptop overheats and hard-restarts under sustained CPU load.** Run
   `llama.cpp.server` with `--n_threads 6`. `evals/real_suite.py` resumes
   from traces already on disk by default (`--force` to recapture), so a
   shutdown mid-capture costs at most one run, not the whole suite.
7. **Commit after every completed unit of work, with descriptive messages
   explaining WHY, not just what.** When asked for many commits, split along
   real work boundaries (one fix = one commit; label/data updates separate
   from code; docs separate from code) rather than padding — see recent git
   log for the pattern. Always re-run the relevant test suite (and ideally
   `pytest evals/ -q`, ~5 min) before pushing.
8. **Keep README.md and REPORT.md current with every result** — including
   failed experiments and stale numbers found in passing. A hardcoded
   "97.8% recall" string was found stale in `score_real_suite.py` itself
   during an unrelated fix and corrected on the spot rather than left.
9. **`known_gap` in a captured trace's label must be revisited whenever the
   detectors change.** `test_documented_gaps_are_still_gaps` fails loudly if
   a gap silently closes — that's a signal to update the label and note, not
   to treat as a broken test.
10. **NEVER use the `§` symbol** — write "section 3.10". Applies everywhere,
    including commit messages and code comments.
11. **Verify a claim about existing code before citing it as an answer**,
    even in a report. A REPORT.md draft once named `confidence.py` as an
    untried "LLM judge" candidate for a gap — rereading its own docstring
    before shipping showed it reads token logprobs, not semantic content, so
    it wasn't a candidate at all. Caught and corrected before the commit,
    which is the point: check the file, don't reason from the name.
12. **To hit a specific commit-count request without padding**, use
    `git add -p <file>` (pipe `y`/`n`/`s` answers) to stage real paragraph/
    table boundaries as separate commits, even from one Edit pass. When git's
    hunk-splitter can't separate genuinely distinct concerns bundled in one
    contiguous diff, reconstruct manually: save final content to scratch,
    `git show <base>:<file> > <file>` to revert the working tree, then
    re-apply in stages with Edit + commit between each, diffing against
    scratch at the end. Only use `git reset --soft` to redo commit
    granularity when `git log origin/main..HEAD` confirms nothing is pushed
    yet — and watch for a stale staged index leaking into an unrelated
    commit after a reset (happened once; caught via `git show --stat HEAD`
    immediately after committing, fixed with another soft reset).

## Useful commands

```bash
python evals/run_eval.py --generated 40 --no-ablation     # full synthetic suite (1018 scenarios, ~1 min)
python evals/run_eval.py --generated 40 --json --significance 25  # + ablation + 25-seed significance (~6 min)
python evals/validity.py                                  # seed-generalisation + trivial baselines + ICC (~3 min)
python evals/score_real_suite.py                          # real-trace scoring
python evals/real_suite.py --relabel                      # re-derive labels only, no model calls
pytest evals/ -q                                           # 303-test gate (~5-6 min)

python -m llama_cpp.server --model models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --n_ctx 8192 --port 8080 --n_gpu_layers 0 --n_threads 6
  # do NOT add --chat_format chatml-function-calling; it cannot terminate.
```
