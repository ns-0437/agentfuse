# AgentFuse — Project Report

**As of 2026-08-13** · 69 commits · 164 tests green · 936 benchmark scenarios
Repo: <https://github.com/ns-0437/agentfuse> · Dashboard: <https://ns-0437.github.io/agentfuse/>

This report is written to be useful to someone deciding whether to rely on the
system, which means the unflattering results are in it and near the top. Where a
number is weak, circular, or measured against something I wrote myself, it says
so.

---

## 1. What the system is

A supervisor that sits *above* an agent's execution graph. It consumes the
telemetry every framework already emits — tool calls, graph routes, state deltas,
token spend — and trips a circuit breaker when a long-horizon failure mode
crosses a threshold. On a trip it freezes state, asks a *separate* model for a
steering correction, injects it, and resumes; or escalates to a human.

The design principle: **the thing judging the run is never the thing performing
it.** An agent in a logical trap is the worst available judge of its own trap.

Five detectors, one engine, three runtime adapters (OpenAI AgentKit via real
`RunHooks`, plain OpenAI SDK, LangGraph).

---

## 2. Headline numbers

Measured on 936 generated scenarios across 21 families and 6 domains, replayed
deterministically through the production monitor.

| Metric | Value | Read as |
|---|---:|---|
| Recall | **97.6%** | real failures caught |
| Precision | **98.6%** | can a trip be trusted |
| F1 | **98.1%** | |
| False-positive rate | **1.2%** | healthy runs wrongly halted |
| Attribution | **83.8%** | correct detector named |
| Recovery rate | **67.6%** | caught failures put back on track |
| Confusion | TP 438 · FP 6 · FN 11 · TN 481 | |

Against trivial baselines — the complexity has to earn itself:

| System | Recall | Precision | F1 |
|---|---:|---:|---:|
| **AgentFuse** | 97.6% | **98.6%** | **98.1%** |
| step cap = 12 | 96.2% | 54.0% | 69.2% |
| naive repeat counter | 49.1% | 66.2% | 56.4% |

**A dumb step cap gets 96% recall.** What this system buys is *precision* — not
noticing failures, but not halting healthy runs.

### Ablation (leave-one-out + rate-matched random control)

| Variant | Recall | Precision | F1 | ΔF1 |
|---|---:|---:|---:|---:|
| full system | 97.6% | 98.6% | 98.1% | |
| ablate `progress` | 78.8% | 98.3% | 87.5% | **−10.6** |
| ablate `spend` | 80.2% | 98.4% | 88.3% | −9.8 |
| ablate `drift` | 81.1% | 100.0% | 89.5% | −8.6 |
| ablate `rate` | 88.6% | 98.5% | 93.3% | −4.8 |
| ablate `loop` | 97.6% | 98.6% | 98.1% | +0.0 |
| random control | 82.9% | 50.0% | 62.4% | −35.7 |

The random control is what makes the rest mean anything: a detector that simply
trips at our frequency reaches F1 62.4%.

---

## 3. Three results that should temper the headline

### 3.1 The benchmark is saturated

Six false positives and eleven false negatives out of 936. **A suite you score
98% on has stopped being a measuring instrument** — it can no longer separate a
good change from a neutral one, because the entire remaining signal is 17
scenarios wide. This is now the top constraint on the project.

Part of the most recent gain came from **correcting a generator that was wrong**
(§4.3). That was legitimate and evidenced, but making the test easier is exactly
how benchmarks quietly stop meaning anything, so it is recorded in place rather
than absorbed into the number.

### 3.2 The statistics improved for the wrong reason

Cluster-adjusted recall moved from 84.6% [57.8–95.7] to 97.7% [94.8–99.0], and
effective *n* from 13 to 222 (design effect 16.9× → 2.0×, ICC 0.407 → 0.048).

**That is a ceiling artifact.** 18 of 20 recall clusters are now all-successes,
so between-cluster variance has nowhere to live and the design effect collapses
toward 1 by construction. The suite gained **one** generator family, 20 → 21.

Quote the **family count**, never the effective *n*. A separate measured result:
sweeping scenarios-per-generator across 40/20/10/5 moved effective *n* only 13→14
— **only more independent families buy statistical power.**

### 3.3 The central premise is unproven

The project's claim is that a *separate reasoning model* writes better
corrections than a fixed rule. Tested for the first time on 2026-08-13, paired on
identical trip snapshots, against a local Qwen2.5-3B-Instruct Q4:

| | mock (templates) | real (Qwen 3B) |
|---|---:|---:|
| **usable rate** | **100%** | **25%** |
| mean quality | 89.2% | 74.2% |
| actionable | 100% | 60% |
| goal-anchored | 100% | 60% |
| latency | ~0 ms | 19.3 s |

A bias was found on *each* side and fixed before publishing this:

- **Our prompt was wrong.** The model wrote *about* the intervention — `inject a
  new task to check if…` — addressed to the supervisor rather than the agent.
  That text is incoherent by the time the agent reads it.
- **Our rubric was biased.** `_ACTIONABLE` matched `realign` but not `re-align`,
  and scored `do not repeat any previous actions involving searching files` — an
  explicit prohibition — as prescribing no action. It recognised its own authors'
  vocabulary.

Both fixes helped (15% → 20% → 25%). **The direction never changed.**

One failure is worse than unhelpfulness. Told to forbid the failing action, the
model forbade the *objective*: *"Do not repeat any steps involving credential
rotation or updating app config until you receive further instructions from a
human."* A weak supervisor can instruct the agent to abandon its task.

**This does not prove reasoning-model steering is a bad idea** — a 3B Q4 model is
the floor of what could work. It proves the **ladder templates have been carrying
the recovery numbers all along**, and that the premise is untested rather than
supported.

---

## 4. Findings worth keeping

### 4.1 Never judge an action before its outcome arrives

A tool step emits `llm_call → tool_call → tool_result`. A detector that tests its
threshold on the *call* can halt a run **one event before the result that would
have cleared it**.

This shipped in **two detectors independently**; a test written afterwards
immediately found a **third** instance. Every time, the runs it killed were ones
**about to succeed** — an agent retrying a flaky endpoint, halted on the final
successful attempt. That is the worst failure mode a guardrail has: not missing a
problem, but destroying work that was fine. Fixing one instance moved FPR 7.4% →
4.1%. Now asserted for every stateful detector.

### 4.2 Detector defects only a benchmark finds

| Defect | Symptom | Effect of fix |
|---|---|---|
| `NoProgressDetector` structurally inert | needed a repeated state hash to detect *absent* state updates | recall 64.3% → 89.2% |
| `LoopDetector` never reset in production | matched `STATE_UPDATE`, but adapters attach state to `TOOL_RESULT` | real production bug the benchmark could not see |
| Benchmark emitted an event stream production never produces | no `llm_call` before `tool_call` | recall 87.9% → 92.4% |
| `max_recoveries=3` against a 4-rung ladder | upper rungs unreachable dead code | recovery 55.4% → 75.2% |
| Eval hard-coded the *lexical* drift threshold | silently zeroed drift recall under embeddings | looked like model failure, was config |
| Identical error treated as a loop | benign retries halted | FP 22 → 6 |

### 4.3 Sometimes the benchmark is what's wrong

`gen_long_sparse_benign` drew its "varied work" from a pool of **4 tools and 4
argument dicts**, so in **199 of 200 runs** it emitted 3+ identical
`(tool, args, result)` triples — worst case **11 identical calls** with no state
change, labelled healthy. That is a loop with a benign label, and it accounted
for 14 false positives no legitimate detector change could remove.

A detector fix was tried first; after correcting the generator it measured
**exactly zero effect**, and was deleted rather than shipped.

**The 4-tool/4-argument entropy is a broader realism weakness. The other
generators have not been audited for it.**

### 4.4 Drift needs a semantic signal, and model size has a floor

| Signal | on-task | paraphrase | **gradual drift** | Separable? |
|---|---:|---:|---:|---|
| lexical | 0.323 | 0.332 | 0.276 | ~0.05 overlapping |
| bge-small (33M) | 0.712 | 0.764 | **0.756** | ❌ **inverted** |
| bge-base (110M) | 0.708 | 0.769 | **0.665** | ✅ gap +0.043 |

The 33M model ranks gradual drift as *more* similar to the goal than on-task
text — any threshold built on it fires backwards. **110M is the floor; a billion
parameters buys nothing here.** Runs locally, free, ~4 ms/sentence.

### 4.5 A concurrency bug that needed no concurrency

`LoopDetector` held **one** `_pending` slot for the call awaiting its result.
That assumes calls and results strictly alternate — false the moment an agent
issues parallel tool calls, which the OpenAI Agents SDK does by default. The
sequence `call(a) → call(b) → result(a)` paired A's *outcome* with B's
*signature*.

Measured by restoring the old logic and driving that exact sequence — **no
threads, no timing:**

| | trips detected | mis-attributed |
|---|---:|---:|
| old (single slot) | 4 | **4 of 4 (100%)** |
| fixed (per-lane) | **8** | 0 |

The harm is not a bad label. Trip evidence feeds the steering instruction, so a
loop on `search_files` produced a correction telling the agent to **stop calling
`fetch_invoice`** — a tool it had used correctly. And the old code found only
*half* the loops, because clobbered pairings never reached the threshold.

Under threads it was worse: `RuntimeError: deque mutated during iteration` raised
straight out of the detector. These adapters raise into the agent's call stack,
so a detector exception **takes down the run the supervisor exists to protect.**

`observe()`, `finish()` and `JSONMemory` are now serialised; `_pending` is keyed
by call id or `(node, tool)`.

**Why a suite scoring 98.1% F1 never saw any of this:** the replay harness emits
strictly sequential events, so it *cannot* produce interleaving. The same
structural blind spot hid the `LoopDetector` reset bug (§4.2). A benchmark only
tests the event orders it knows how to generate.

### 4.6 A restart used to disarm the breaker

The supervisor was in-memory only, which reads like a missing convenience and is
actually a safety hole. An agent under a 500,000-token ceiling that dies at
480,000 and restarts came back with its budget at **zero** — the restart *rearmed*
the guard whose whole purpose is bounding unattended spend. Loop counters, the
stall counter and the learned calibration baseline reset identically, so the
agent resumed against a supervisor that had forgotten why it was worried.

Now persisted to stdlib SQLite (WAL, so a hard kill leaves a recoverable file),
keyed by `run_id`, off by default, and always written on a trip regardless of
interval — that is the state a crash most often follows and the most expensive to
lose, since it carries the recovery ladder's position.

`state_dict` is generated by reflection rather than five hand-written serialisers
that drift from the classes they mirror. That is only safe because the guarantee
is *behavioural*: clone a detector through a checkpoint, drive both forward, and
require the same verdict and evidence at every step.

It earned that on the first run. `_goal_vec` is a list of `numpy.float32` — it
passes an "is it a list?" check but is not JSON, and a permissive `default=str`
round-tripped it into a list of **strings**. The restored detector then died
inside a cosine similarity, several layers from the cause. Every attribute must
now prove itself `json.dumps`-able individually, and `default=` coercion is gone:
dropping a value costs a counter that restarts, writing a corrupted one costs a
supervisor that appears to restore and then breaks the run.

### 4.7 The dollar ceiling was decorative

`SpendDetector` has accepted `max_cost_usd` since the first commit. **Nothing ever
populated `AgentEvent.cost_usd`**, so `_total_cost` stayed at 0.0 for the life of
every run and the ceiling could not fire.

Measured before the fix: a monitor with `max_cost_usd=1.0` burned **11,940,000
tokens** without tripping once, reporting `$0.00` throughout. An operator who set
a dollar budget got no protection at all, and nothing said so.

The design decision that matters is what an *unknown* model costs. Returning
`0.0` is convenient and rebuilds the bug exactly — spend accumulating at zero
under a ceiling that never fires. So `estimate_cost` returns `None`, unpriced
tokens are counted separately, `cost_is_complete` goes false, and an
unenforceable ceiling **warns at construction**, while it can still be fixed.

| Same workload, `max_cost_usd=1.0` | Result |
|---|---|
| no model (old behaviour) | no trip, `$0.00`, silent |
| no model (now) | no trip, but **23,940,000 unpriced tokens**, `cost_is_complete=False` |
| `model="gpt-4.1"` | **trips at step 6, $1.08**, 0 unpriced |

The bundled table is a convenience default, not a source of truth: it carries an
as-of date, `AGENTFUSE_PRICING_FILE` overrides it with no code change, and the
figures are guardrail estimates that will not reconcile with an invoice.

### 4.8 Escalation nobody received

The breaker escalates when steering is exhausted or a hard ceiling is hit. That
meant returning a `PAUSE` directive and **printing to the console** — for a
supervisor whose premise is *unattended* runs of hours to days, a notification
sent to a process nobody is watching, which is the exact situation it exists for.

Now delivered over a webhook (stdlib `urllib`, no new dependency; works with
Slack, Discord, PagerDuty or an internal endpoint). Three decisions carry the
weight:

- **Delivery is verified, not assumed.** `send()` returns whether it worked.
  `finish()` reports `escalation_delivered`, where `None` means *never needed*
  and `False` means *needed, and nobody was told* — those must not look alike.
  One failure is never erased by a later success.
- **Egress is treated as egress.** The payload naturally contains the agent's
  reasoning and tool output. Free text is sanitised — trip evidence is
  agent-produced, so it is untrusted on the way *out* too — truncated, and
  `escalation_include_agent_text=False` drops it entirely.
- **A webhook outage never propagates.** Bounded timeout, two retries, returns
  `False` rather than raising.

### 4.9 The pattern behind four of these bugs

Four defects in this project share one shape: **a guard that looks armed and is
not.**

| Guard | What it did when it should have fired |
|---|---|
| `NoProgressDetector` | nothing — the trip condition was unreachable |
| `max_cost_usd` | nothing — 11.94M tokens under a $1 ceiling |
| spend counter across a restart | reset to zero, granting a fresh budget |
| "escalate to a human" | printed to a console nobody was reading |

All four passed their tests. All four read correctly in review. None was found by
inspection — each surfaced only from asking *what does this guard actually do at
the moment it is supposed to work?*

That question is now the project's standing check on anything described as a
safety mechanism, and it is why each fix above ships with a test that drives the
guard to its firing point rather than asserting its configuration.

### 4.10 A backspace that disabled a security defence

A shell heredoc wrote a literal `\x08` where `\b` was intended in an
injection-detection regex — invisible to grep, editors, and file reads, and it
silently disabled the defence. Found only by printing `repr()` of the compiled
pattern. **Rule adopted: never edit regexes or code through shell heredocs.**
All sources are now scanned for control characters by a test.

---

## 5. Phase status

| Phase | Scope | Status |
|---|---|---|
| **1 — Eval harness** | ground-truth scenarios, hard negatives, Wilson + clustered CIs, ablation, random control | ✅ Done |
| **2 — Verified + memoried recovery** | steering ladder, failure→steer→outcome memory, closed verification loop | ✅ Done *(premise unproven — §3.3)* |
| **3 — Adaptive thresholds** | per-run baselines from evidenced-healthy stretches, widen-only | ✅ Done |
| **4 — Signal ladder** | logprobs, self-probe, activation probes | ❌ Not started |
| **5 — Productionisation** | injection hardening ✅ · thread-safety ✅ · SQLite checkpoints ✅ · real cost table ✅ · webhook escalation ✅ · PyPI ❌ | 🟡 5 of 6 |

**3 of 5 complete.** Phase 5 is at 5 of 6 — only PyPI packaging remains.

---

## 6. Honest readiness assessment

Verified against the code, not asserted:

| Requirement | State |
|---|---|
| Detection quality | **Strong** — but on a saturated, self-authored suite |
| Steering quality | **Unproven** — templates beat the only real model tested |
| Persistence / checkpoints | **Fixed 2026-08-13** (§4.7). SQLite checkpoints; a resumed run keeps its spend ceiling, loop counters and calibration baseline |
| Thread / async safety | **Fixed 2026-08-13** (§4.5). `observe()` and the recovery memory are serialised; parallel tool calls pair correctly. One monitor per agent run remains the supported model |
| Packaging | Not on PyPI |
| Real-model validation | Supervisor half only, with a 3B model; agent obedience still synthetic |
| Prompt-injection hardening | Done — sanitise, fence, trust-boundary clause, 15 tests |
| Cost safety | Done — `AGENTFUSE_OFFLINE`, $0 ever spent on this project |

**Verdict: a strong research instrument, not a deployable product.** The
methodology — hard negatives, leave-one-out ablation, a rate-matched random
control, cluster-adjusted intervals, benchmark-validity checks — is more rigorous
than most guardrail projects publish. The engineering underneath it is not yet
production-grade, and the headline claim is still unvalidated.

---

## 7. What would move it forward, in order

1. **Settle §3.3 with a larger model.** If a 7B closes the gap it is a model-size
   problem; if it does not, the honest product is the *deterministic ladder plus
   detectors* — simpler, and still valuable. Everything downstream depends on
   which. `evals/real_model.py --base-url …` already runs this.
2. **De-circularise the rubric.** The mock's 100% is rigged by construction. An
   independent judge or a human spot-check would make the comparison real.
3. **Import captured real traces.** The suite is saturated; more synthetic
   families will not help much. `evals/trace_import.py` exists for this.
4. **Audit the remaining generators** for the 4-tool/4-argument artifact (§4.3).
5. **Subtle drift** — 8 of 11 remaining false negatives plus 6 false positives,
   all riding the thin ±0.043 embedding separation.
6. **Phase 5**: persistence, thread-safety, packaging.

---

## 8. Reproducing everything here

```bash
python evals/run_eval.py --generated 40 --json    # full suite + ablation
python evals/run_eval.py --generated 40 --sweep   # threshold sweeps
python evals/validity.py                          # checks on the benchmark itself
python evals/real_model.py --base-url …           # templates vs a real model
pytest evals/ -q                                  # 164-test CI gate
```

No API key required, nothing billed. `evals/baseline.json` records every floor,
every known weakness, and the reasoning behind each threshold.

---

*Methodology for the ablation design is adapted from AE Studio's research on
Endogenous Steering Resistance — methodology only, no code used or derived. Full
attribution in [CITATION.md](CITATION.md).*
