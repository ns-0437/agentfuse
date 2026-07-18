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

## Project layout

```
agentfuse/
  events.py            normalized AgentEvent + ExecutionSnapshot
  monitor.py           CircuitBreakerMonitor — the engine
  recovery.py          RecoveryEngine — separate reasoning-model steering (real + mock)
  tracer.py            live console trace + JSONL observability
  detectors/           loop · drift · progress · spend
  adapters/            agentkit · agentkit_hooks (real RunHooks) · openai_sdk · langgraph
examples/              demo_loop_trap · demo_drift · demo_escalation · real_agentkit_run
```

## Design choices that matter to reviewers

- **Zero required dependencies** in the core — runs anywhere, demos never break.
- **Graceful degradation** everywhere: no OpenAI key → mock recovery + lexical
  drift; no `rich` → plain text; unknown terminal encoding → ASCII markers.
- **Recovery never crashes the run** — a failure in the supervisor falls back to
  a deterministic steer.
- **Framework-agnostic core** proven by three adapters over one engine.
