# ⚡ AgentFuse — a Logical Circuit Breaker for Long-Range Agents

> Autonomy that knows when it's going wrong — and steers itself back.

![AgentFuse breaking a live agent loop and self-healing](assets/demo.gif)

**▶ Live observability dashboard:** https://ns-0437.github.io/agentfuse/ — explore every supervised run (timeline, trips, steering recoveries, token spend) right in the browser.

Long-running agents (hours → days, hundreds of steps) don't usually fail with a
crash. They fail *quietly*: an infinite tool loop, a slow drift from the original
objective, a logical trap where the model reasons flawlessly from a false premise,
or a budget silently burned to zero. The agent that drifted is the **worst possible
judge** of whether it drifted.

**AgentFuse is a supervisor that sits *above* the agent's execution graph.** It
watches the telemetry every framework already emits — tool calls, graph routes,
state changes, token spend — and **trips a circuit breaker** the moment a
long-horizon failure mode crosses a threshold. On a trip it freezes state and asks
a **separate reasoning model** for a *steering recovery path*, injects that
correction, and resumes. When recovery isn't safe, it **escalates to a human**
instead of blindly retrying.

One engine. Three runtimes: **OpenAI AgentKit** (first-class), plain **OpenAI SDK**,
and **LangGraph**.

---

## Why this wins where it's judged

| Theme | How AgentFuse fits |
|---|---|
| **Observability** *(primary)* | Live trace of every graph route, state delta, tool signature, and token/$ spend, streamed to console **and** JSONL for any backend. This is agent observability — but *active*, not a passive dashboard. |
| **Security / Safety** *(secondary)* | A hard stop for runaway autonomy: budget ceilings, loop guards, and human escalation prevent an unattended agent from spending or acting without bound. |

The differentiator: most observability tools *watch*. AgentFuse **watches and
intervenes** — a closed-loop, self-healing safety layer.

---

## 60-second demo (no API key, nothing to install)

The core is **stdlib-only**. Clone and run:

```bash
python examples/demo_loop_trap.py     # infinite tool loop -> detected -> self-healed
python examples/demo_drift.py         # goal drift -> re-anchored to objective
python examples/demo_escalation.py    # unrecoverable failure -> hard stop / human escalation
```

Each prints a live trace and writes a machine-readable `runs/*.jsonl`.
`pip install rich` for colored panels; set `OPENAI_API_KEY` to swap the offline
mock for a real reasoning model + real embedding-based drift detection.

### Real AgentKit run (not simulated)

```bash
pip install openai-agents
python examples/real_agentkit_run.py
```

This drives the **genuine `openai-agents` SDK** — a real `Agent`, real
`@function_tool`s, the real `Runner`, and AgentFuse's real `FuseRunHooks`
observing the live lifecycle. The agent falls into a real infinite tool loop;
the breaker — watching the actual SDK hooks — **trips, aborts the runaway run,
injects a steering instruction into the conversation, and re-runs**, after which
the agent completes. Only the model's token generation is stubbed (a
`ScriptedModel`) so the run is free; pointing it at a real model is a one-line
change (drop the `RunConfig` override, set `OPENAI_API_KEY`) — the hooks and
breaker code are byte-for-byte identical.

### What you see

```
🔧 step 1  tool_call   search_files({"dir":"./config","pattern":"*.conn"})
🔧 step 2  tool_call   search_files({"dir":"./config","pattern":"*.conn"})
🔧 step 3  tool_call   search_files({"dir":"./config","pattern":"*.conn"})
┌── ⚡ CIRCUIT BREAKER TRIPPED - LOOP (trip) ──────────────────────────────┐
│ Tool 'search_files' called with identical arguments 3x, no state progress │
└───────────────────────────────────────────────────────────────────────────┘
┌── 🧭 STEERING RECOVERY - action=inject (via o4-mini) ─────────────────────┐
│ STOP repeating `search_files`… re-read your objective… try another path.  │
└───────────────────────────────────────────────────────────────────────────┘
▶️  step 3  resume      steering injected; agent resuming with corrected plan
🔧 step 4  tool_call   secret_manager.get({"name":"prod/db/primary"})   ← recovered
✅ step 5  complete     objective achieved after self-healing
```

---

## Architecture

```
        ┌──────────────── AGENT EXECUTION GRAPH ────────────────┐
        │   parallel nodes · tool calls · handoffs · state Δ    │
        └────────────────────────┬──────────────────────────────┘
              emits AgentEvents   │  (tool / route / state / spend)
                                  ▼
        ┌──────────── CircuitBreakerMonitor (supervisor) ───────┐
        │  Detectors (independent sensors):                     │
        │    • LoopDetector        repetitive tool signatures   │
        │    • DriftDetector       goal vs. system-prompt dist. │
        │    • NoProgressDetector  activity w/ no state change  │
        │    • SpendDetector       token/$ ceiling + burn rate  │
        └────────────────────────┬──────────────────────────────┘
                     trip!        │  freeze ExecutionSnapshot
                                  ▼
        ┌──────────── RecoveryEngine (SEPARATE model) ──────────┐
        │  reasoning model → SteeringPath                       │
        │    { inject correction · escalate to human · abort }  │
        └────────────────────────┬──────────────────────────────┘
                                  ▼
                 Directive → adapter injects steering & resumes
```

Design principle: **the thing judging the run is never the thing performing it.**
Detectors are independent and composable; adding a new failure-mode sensor is one
class implementing `inspect(event, history) -> Trip | None`.

---

## Use it with your framework

### OpenAI AgentKit (first-class, real `RunHooks`)

`FuseRunHooks` is a genuine `agents.RunHooks` subclass — pass it straight to
`Runner.run`. It observes the live lifecycle and raises `BreakerInterrupt` to
abort a runaway run so you can inject steering and re-run. See a complete,
runnable end-to-end integration in
[`examples/real_agentkit_run.py`](examples/real_agentkit_run.py).

```python
from agents import Agent, Runner
from agentfuse.adapters.agentkit_hooks import FuseRunHooks, BreakerInterrupt
from agentfuse import DirectiveKind

agent = Agent(name="rotator", instructions=GOAL, tools=[...])
fuse = FuseRunHooks(original_goal=GOAL, loop_threshold=3, max_tokens=500_000)

input_items = [{"role": "user", "content": TASK}]
while True:
    try:
        result = await Runner.run(agent, input_items, hooks=fuse, max_turns=12)
        fuse.finish(); break
    except BreakerInterrupt as bi:
        if bi.directive.kind is DirectiveKind.INJECT:
            input_items.append({"role": "user",
                                "content": f"[CIRCUIT BREAKER STEERING] {fuse.take_steering()}"})
            continue          # re-run with the corrective nudge
        fuse.finish("escalated"); break   # PAUSE / ABORT -> hand to a human
```

### Plain OpenAI SDK (framework-free)

```python
from openai import OpenAI
from agentfuse.adapters.openai_sdk import guarded_tool_loop

guarded_tool_loop(OpenAI(), model="gpt-4.1", system_prompt=GOAL,
                  user_input=TASK, tools=TOOLS, tool_router=run_tool,
                  max_tokens=200_000)   # breaker steers the loop automatically
```

### LangGraph

```python
from agentfuse.adapters.langgraph import FuseCallbackHandler
handler = FuseCallbackHandler(original_goal=GOAL)
graph.invoke(state, config={"callbacks": [handler]})
# add handler.supervisor_node as a node to inject steering between agent turns
```

### Low-level (any runtime)

```python
from agentfuse import CircuitBreakerMonitor, MonitorConfig, AgentEvent, EventType, DirectiveKind

mon = CircuitBreakerMonitor(MonitorConfig(original_goal=GOAL, max_tokens=200_000))
d = mon.observe(AgentEvent(type=EventType.TOOL_CALL, step=n,
                           tool_name="search", tool_args={"q": q}))
if d.kind is DirectiveKind.INJECT:
    agent.add_system_message(d.steering_text)
```

---

## What each detector catches

| Detector | Failure mode | Trip condition |
|---|---|---|
| `LoopDetector` | Infinite / repetitive tool loop | Same `(tool, args)` signature N× in a window with no state progress |
| `DriftDetector` | Goal drift | Semantic similarity to the original objective drops below threshold for K turns (real embeddings, or offline lexical fallback) |
| `NoProgressDetector` | Logical trap / stall | Many actions, zero change to working-state hash |
| `SpendDetector` | Runaway cost | Cumulative token/$ ceiling (→ escalate) or burn-rate spike (→ steer) |

---

## Does it actually work? (measured, not claimed)

Most guardrail projects assert they work. This one is scored against a benchmark
with ground truth, confidence intervals, and a significance test — and the
numbers are published, including the unflattering ones.

```bash
python evals/run_eval.py --generated 40 --json    # 880 scenarios + ablation
python evals/run_eval.py --generated 40 --sweep   # threshold sweeps
python evals/validity.py                          # checks on the benchmark itself
pytest evals/ -q                                  # 94-test CI gate
```

**880 scenarios** from **20 parameterised generator families** across 6 domains,
with ground truth true *by construction*. 440 are genuine failures; **440 are
hard negatives** — healthy runs that look like failures: a legitimate retry,
polling that really is progressing, a sub-goal that reads as drift, a
**paraphrased objective**, an error followed by a competent pivot. Hard negatives
are what make the false-positive rate measurable, and that rate decides whether
anyone leaves a guardrail switched on. Everything replays deterministically in
~30s — no API key, no cost.

### Current baseline (2026-08-12, replay mode, local embeddings)

| Metric | Value | Read as |
|---|---:|---|
| Recall | **88.4%** | real failures caught |
| Precision | **90.9%** | can you trust a trip |
| F1 | **89.6%** | |
| False-positive rate | **8.9%** | healthy runs halted |
| Recovery rate | **71.7%** | caught failures put back on track |
| **Recall, cluster-adjusted** | **84.6% [57.8–95.7]** | ← **the honest interval** |

**Read the last row, not the first.** Scenarios generated from one template
behave near-identically, so they are not independent samples. The measured design
effect is ~17–33×, putting the **effective sample size near 13** against a
nominal 880. Nominal Wilson intervals are roughly seven times too narrow. A
consequence worth knowing: *adding scenarios per generator buys no statistical
power at all* — sweeping 40/20/10/5 per family (total 880→110) moved effective n
only 13→14. Only more independent **families** narrow the interval.

### Against trivial baselines

Four detectors, a steering ladder, a memory and a calibrator have to beat "stop
after N steps" or the complexity is unjustified:

| System | Recall | Precision | F1 |
|---|---:|---:|---:|
| **AgentFuse** | 88.4% | **90.9%** | **89.6%** |
| step cap = 12 | **96.2%** | 54.0% | 69.2% |
| naive repeat counter | 49.1% | 66.2% | 56.4% |

Note what this actually says: **a dumb step cap gets 96% recall.** What AgentFuse
buys is *precision* — not noticing failures, but not halting healthy runs.

**What's still broken, stated plainly:**

- **Partial-progress traps are missed** (`progress` family 67%). An agent that
  advances state on *every* step while converging on none starves a binary
  progress signal. Needs a rate-of-progress measure.
- **Sparse-progress workloads still false-positive at ~42%** — down from 100%
  with calibration off, but the first cycle trips before any baseline exists.
- **`steering_usable = 100%` is circular** — that rubric scores instructions
  built from templates written alongside it. It is not evidence and is flagged as
  such in `baseline.json`.
- **Everything is synthetic** except one captured trace, which validates event
  *shape* against production, not the failure *distribution*.

### What the benchmark already changed

The threshold sweep found the shipped `drift_threshold=0.45` was badly wrong:

| drift_threshold | Recall | Precision | F1 | FPR |
|---:|---:|---:|---:|---:|
| 0.20 (**new default**) | 65.1% | 83.9% | **73.3%** | **10.8%** |
| 0.45 (old default) | 71.5% | 61.4% | 66.0% | 39.0% |

Six points of recall bought **28 points of false-positive rate**, took attribution
from 87% to 99.4%, and eliminated all 54 premature trips. That one-line change was
found by measurement, not intuition — which is the entire argument for building
this before building anything else.

### Drift needs a semantic signal — and the model size has a floor

Lexical similarity cannot separate the case that matters. Neither can a *small*
embedding model, which is the surprising part:

| Signal | on-task | paraphrase | **gradual drift** | Separable? |
|---|---:|---:|---:|---|
| lexical (difflib + Jaccard) | 0.323 | 0.332 | 0.276 | ~0.05 overlapping window |
| **bge-small-en-v1.5 (33M)** | 0.712 | 0.764 | **0.756** | ❌ **inverted** |
| **bge-base-en-v1.5 (110M)** | 0.708 | 0.769 | **0.665** | ✅ gap +0.043 |

The 33M model is not merely weaker — it scores gradual drift as *more* similar to
the objective than genuinely on-task text, so any threshold built on it fires
backwards. **110M is the floor; a billion parameters buys nothing here.**

This runs **locally and free**: `pip install agentfuse[embeddings]` pulls a
~120MB ONNX model that needs no API key, touches no network, and costs ~4ms per
sentence on CPU — faster than a hosted round trip, and the agent's reasoning
never leaves the machine. Drift-family recall went **76.2% → 90.0%**.

`AGENTFUSE_OFFLINE` disables only the *hosted* backend; a local model spends
nothing, so treating it as "offline" would force the weakest signal for no gain.

### Ablation — which detectors carry the signal

Methodology adapted from AE Studio's [ESR research](https://ae.studio/research/esr):
they established causality for a set of SAE latents by zero-ablating them and
measuring the drop, controlled against *random latents matched for activation
frequency*. Both moves apply here — leave-one-out per detector, plus a random
detector rate-matched to our own trip frequency and run across 25 seeds.

Without the control, a system that simply trips often would post a respectable
F1. It is the control that makes the headline number mean anything.

### Honest limitations of the benchmark itself

- **Synthetic.** The generators encode *my* model of agent failure, so they fix
  sampling error, not authoring bias. `evals/trace_import.py` converts real
  captured runs into labelled cases; that is the only real cure.
- **Detection only.** We score whether a failure is *caught*, never whether the
  steering that follows actually fixes it. That needs live models (Phase 2).
- **Notional token savings.** We assume halting saves everything downstream, and
  charge a flat 1,500 tokens per steering call.

For scale context: AE Studio's ESR baseline ran **7,892 trials**. This suite runs
880 — but its *effective* sample size is around 13, because the scenarios are
clustered by generator. That is the number to reason about, and it is not enough
to call anything settled.

### Prior work

The ablation design is adapted from AE Studio's research on **Endogenous Steering
Resistance** ([paper](https://arxiv.org/abs/2602.06941) ·
[code](https://github.com/agencyenterprise/endogenous-steering-resistance)) —
methodology only; no code is used or derived. Full attribution, and a note on how
intrinsic (ESR) and extrinsic (AgentFuse) approaches complement each other, is in
[CITATION.md](CITATION.md).

---

## Project layout

```
agentfuse/
  events.py            normalized AgentEvent + ExecutionSnapshot
  monitor.py           CircuitBreakerMonitor — the engine
  recovery.py          RecoveryEngine — separate reasoning-model steering (real + mock)
  tracer.py            live console trace + JSONL observability
  detectors/           loop · drift · progress · spend
  adapters/            agentkit · agentkit_hooks (real RunHooks) · openai_sdk · langgraph
  embedding.py         local ONNX first, hosted second, lexical last
  memory.py            what was steered, and whether it worked
  strategies.py        the escalating ladder of interventions
  calibration.py       per-run thresholds learned from healthy stretches
  sanitize.py          agent/tool output is untrusted input
examples/              demo_loop_trap · demo_drift · demo_escalation · real_agentkit_run
evals/                 the benchmark — ground-truth scenarios, metrics, ablation
  schema.py            Scenario / Label / CostModel
  scenarios/           positives (real failures) · negatives (healthy lookalikes)
  runner.py            deterministic replay through the real monitor
  metrics.py           precision · recall · FPR · attribution · net tokens
  ablation.py          leave-one-out + rate-matched random control
  validity.py          checks on the BENCHMARK: generalisation, baselines, clustering
  captured/            a real openai-agents trace, scored like any other case
  results/             REPORT.md + results.json (regression baseline)
```

## Design choices that matter to reviewers

- **Zero required dependencies** in the core — runs anywhere, demos never break.
- **Graceful degradation** everywhere: no OpenAI key → mock recovery + lexical
  drift; no `rich` → plain text; unknown terminal encoding → ASCII markers.
- **Recovery never crashes the run** — a failure in the supervisor falls back to
  a deterministic steer.
- **Framework-agnostic core** proven by three adapters over one engine.
