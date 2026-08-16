# AgentFuse — Project Report

**As of 2026-08-16** · 96 commits · 212 tests green · 936 synthetic scenarios + 17 captured real traces
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
`RunHooks`, plain OpenAI SDK, LangGraph), all three tested — adding those tests
found four bugs, §4.10.

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
| Recovery rate | **67.6%** ⚠ | **Do not read this as "put back on track".** It is scored against synthetic ground truth. Measured against a real agent, corrections were obeyed **2.4%** of the time — §3.5 |
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

## 3. Five results that should temper the headline

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

### 3.3 The central premise is disproven at every size we can test

The project's claim is that a *separate reasoning model* writes better
corrections than a fixed rule. **It does not** — not at 3B, and not at 7B.
First tested 2026-08-13, paired on identical trip snapshots, against a local
Qwen2.5-3B-Instruct Q4:

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

**Settled 2026-08-14 by scaling the model.** The obvious defence of the 3B result
was that 3B is simply too small. It is not the explanation:

| | usable | quality | actionable |
|---|---:|---:|---:|
| templates (mock) | **100%** | 89.2% | 100% |
| Qwen2.5-3B-Q4 | 25% | 74.2% | 60% |
| Qwen2.5-7B-Q4 *(native format)* | 30% | 75.0% | **35%** |
| Qwen2.5-7B-Q4 *(fn-calling format)* | 40% | 78.3% | 50% |

Both 7B chat formats were run on purpose: the first showed bare goal
restatements, which could have been an artifact of running under a tool-calling
handler with a JSON grammar. They agree.

**The driver is `actionable`.** Asked for a concrete corrective instruction, the
7B answers with a restatement of the objective — *"Rotate the production database
credential and update the app config."* — which prescribes nothing. That is not a
rubric artifact; `_ACTIONABLE` was already widened once to remove template bias.

**So the ladder templates have been carrying the recovery numbers all along, and
scaling does not change that.** The honest product is the deterministic ladder
plus the detectors. Frontier reasoning models remain untested for want of
credits.

**Corrected 2026-08-16 — read §3.5 before trusting this table.** Every figure
above is a *rubric score*: it measures the text of a correction, not its effect.
When compliance was finally measured, the winning arm — the templates, at 100%
"usable" — was obeyed **2.4% of the time**. So this section compares two texts,
not two effects, and "model-written steering is worse" is a claim about wording
whose downstream consequence was never established.

### 3.4 A measured signal that must not ship

Phase 4 Tier 1 reads the model's own token logprobs. Measured against
Qwen2.5-3B-Instruct-Q4, n=14 per condition:

| condition | mean logprob | gap vs healthy | Cohen d | |
|---|---:|---:|---:|---|
| on_task | −0.686 (sd 0.068) | — | — | baseline |
| stuck_loop | −0.775 | +0.089 | 0.87 | separates |
| offtopic | −0.830 | +0.144 | 1.44 | separates |
| trap | −0.842 | +0.156 | 1.68 | separates |
| **ambiguous** *(control)* | −0.707 | +0.022 | 0.28 | **control holds** |

The signal is real and it tracks *failure*, not *difficulty* — the healthy-but-hard
control did not drop. **And the detector is still harmful**: ablation puts it at
**ΔF1 +10.8 for removal**, with identical recall and 118 extra false positives. It
ships off by default. See §4.12 for why, because the reason generalises.

### 3.5 The corrections are ignored — 40 times out of 41

Measured 2026-08-16 against a real Qwen2.5-7B across 12 supervised tasks, with
compliance read off **behaviour** (did the next tool call differ from the one
that tripped the breaker):

| | |
|---|---:|
| steers injected | **41** |
| complied | **1** |
| resisted | **40** |
| **compliance rate** | **2.4%** |

**The steers were the deterministic templates.** All 43 recovery records carry
`backend=mock` — the ladder instructions that score **100% "usable"** on
`evals/steering.py`. Instructions our own rubric rates as perfect were obeyed
once in forty-one attempts.

`steering_usable = 100%` was already flagged as circular, because the rubric was
written alongside the templates it grades. It is now worse than circular: it is
**disconnected from outcomes.** The rubric measures whether an instruction reads
well. A real agent ignores 97.6% of the instructions it approves.

This is the single most consequential measurement in the project, and it lands
against the half of the system that had never been tested end to end.

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

### 4.10 Four bugs in twenty minutes, from testing what was never tested

`openai_sdk` and `langgraph` were advertised in the README and had never been
executed by a test. Writing that coverage found four defects almost immediately,
three of them in shipped code:

1. **`LoopDetector` was inert behind both adapters.** It cleared its counters on
   the mere *presence* of `event.state`. Both adapters attach `state` to **every**
   tool result, because they cannot know whether a call achieved anything — so
   the detector reset after every result, never formed a pair, and never fired.
   Measured: 11 identical calls returning identical results, `_pairs` still 0.
   The progress detector caught those runs as a backstop and named the wrong
   cause, so the steer said *"you are busy but the task is not moving"* instead
   of *"stop calling `search_files`"*. **Presence is not progress.**
2. **`openai_sdk` discarded the directive from `TOOL_RESULT`** — which is exactly
   where the loop detector now fires, since it waits for the outcome. Steering
   was generated, written to the trace, and never applied. The observability
   showed a recovery the agent never received.
3. **`langgraph`'s `on_tool_end` emitted no tool name**, so a result could not be
   matched to its call and no trip could name the offending tool.
4. One of mine, introduced while fixing #2: I added the SDK `call_id` to the
   result event but not the call, breaking lane matching.

**Why the 936-scenario benchmark missed all of it:** the replay harness only
emits `state` on steps it has labelled as genuine progress, so it never produced
the event shape that breaks the contract. Eval numbers were **unchanged** by the
fix — which is the point. A benchmark only tests the event orders it knows how to
generate.

One of my tests was wrong and the adapter was right: `supervisor_node` returning
a partial dict is correct LangGraph convention. I corrected the test, not the
code.

### 4.11 Tier 2 — a real signal, beaten by three lines of string comparison

Activation probes are ESR's actual method, and this was recorded as blocked for
the life of the project. **The block was not hardware.** An interrupted install
had left an orphaned `torch` directory with no metadata and no `lib/` — and it
died because **Windows long paths are disabled** while this Python sits ~130
characters deep under a Store path, so torch's deepest headers exceed `MAX_PATH`.
A venv at a short path fixed it, and keeps torch out of the stdlib-only core.

Linear probes on last-token hidden states, Qwen2.5-0.5B-Instruct, 135 distinct
transcripts:

| layer | held-out AUC | ambiguous flagged |
|---|---:|---:|
| 0 (embeddings) | 1.000 | **100%** — lexical, confuses *hard* with *stuck* |
| 4 | 0.863 | 13% |
| 13 | 1.000 | 13% |
| **16–24** | **1.000** | **0%** |
| shuffled-label control | 0.453 | *chance — no leak* |

The layer profile is exactly what the hypothesis predicts: early layers separate
on vocabulary, deep layers separate cleanly *and* correctly call the
healthy-but-hard control healthy.

**And it still should not ship.** What the probe classifies perfectly — a
transcript repeating an identical result — is precisely what `LoopDetector`
already catches with a string comparison:

| | cost per turn |
|---|---:|
| probe forward pass | **558 ms** |
| `LoopDetector` | **0.029 ms** |

**19,000× cheaper for the same detection.** Tier 0 wins outright on the only
failure mode this design could test.

Three methodology bugs were caught by the controls, each of which would have
produced a publishable-looking number: the shuffled-label control rejected the
first run because duplicate prompts spanned the split; layer 0 scoring 1.000
exposed that the classes were separable by vocabulary alone; and standardising
with `sd + 1e-6` amplified float noise on constant features into a perfect AUC
read out of nothing.

### 4.12 A large effect size is not a usable detector

Phase 4 Tier 1 produced the cleanest example in the project of a measurement that
was necessary and nowhere near sufficient.

Against a real model, all three failure modes separated from healthy reasoning
beyond their 95% intervals, and the ambiguous-but-healthy control did *not* drop —
so the signal genuinely tracks failure rather than difficulty (§3.4). By the usual
standards that is a good result: Cohen's d up to 1.68.

Then leave-one-out put it at **ΔF1 +10.8 for removing it.** Recall was identical
(97.6% either way — it caught nothing the behavioural detectors missed) and it
cost **118 extra false positives**, FPR 1.2% → 26.7%.

**Why:** d ≈ 1.4 still leaves heavy per-turn overlap. A controlled comparison
aggregates over a whole generation; a detector must decide on *every turn*, and
across a long run even a three-consecutive-turn requirement fires constantly on
healthy work. **Base rate and run length dominate effect size.**

Two of my own errors were corrected en route. The default threshold was an
absolute 0.6 nats against real gaps of 0.09–0.16, so it could never have fired —
a detector tuned above its own effect size, the same defect `NoProgressDetector`
had. And the first analysis compared two means against a flat cut-off and called
a 0.16 gap "usable" at n=6 with sd 0.10–0.16.

### 4.13 A backspace that disabled a security defence

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
| **4 — Signal ladder** | Tier 0 behavioural ✅ · Tier 1 logprobs ✅ · Tier 2 activation probes ✅ | ✅ Done *(both internal tiers measured, both ship OFF — §3.4, §4.11)* |
| **5 — Productionisation** | injection hardening ✅ · thread-safety ✅ · SQLite checkpoints ✅ · real cost table ✅ · webhook escalation ✅ · PyPI ❌ | 🟡 5 of 6 |

**4 of 5 complete.** Phase 5 is at 5 of 6 — only PyPI packaging remains.

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
| Real-model validation | Supervisor half only, with a 3B model that LOST to the templates; agent obedience still synthetic (§8.1, §8.2) |
| Adapter coverage | **3 of 3** — fixing the two untested ones found 4 bugs (§4.10) |
| CI / portability | **3 OS × 3 Python versions**, plus demos and benchmark floors (§8.4) |
| Secret redaction | **All three egress paths** — with a stated residual gap (§8.5) |
| Prompt-injection hardening | Done — sanitise, fence, trust-boundary clause, 15 tests |
| Cost safety | Done — `AGENTFUSE_OFFLINE`, $0 ever spent on this project |

**Verdict: a strong research instrument, not a deployable product.** The
methodology — hard negatives, leave-one-out ablation, a rate-matched random
control, cluster-adjusted intervals, benchmark-validity checks — is more rigorous
than most guardrail projects publish. The engineering underneath it is not yet
production-grade, and the headline claim is still unvalidated.

---

## 6.5 The ESR merge — what was borrowed, and what was missed

The premise of adopting AE Studio's ESR work was **complementarity**: their signal
is *internal* (the model's own consistency monitoring), ours is *external* (an
independent supervisor watching behaviour). The stated goal was to combine them.
That has **not** happened, and being precise about why is more useful than the
parts that did.

### What was taken

**The experimental methodology, and only that** — leave-one-out ablation plus a
rate-matched random control (`evals/ablation.py`, attributed in
[CITATION.md](CITATION.md)). It has been the single highest-value borrow in the
project. It is what measured the confidence detector at **ΔF1 +10.8 for removal**
and what exposed the loop detector as net-harmful before it was fixed. Without a
random control, a system that simply trips often posts a respectable F1.

### What was not

- **SAE latents.** Tier 2 used raw hidden states with a linear probe, not sparse
  autoencoder features.
- **Zero-ablation of model internals.** We ablate *detectors*; ESR ablates
  *latents*.
- **The phenomenon itself** — models *resisting* steering.

### The miss

`evals/captured/real_7b_supervised.jsonl` **is a direct observation of endogenous
steering resistance, and it was not recognised as one.** The breaker caught the
loop three times and injected a correction three times. The agent ignored all
three and kept calling the same tool.

That reframes §3.3. The conclusion there was *"model-written steering is worse
than the templates."* There is a second explanation, ESR-shaped, that the data
cannot rule out: **perhaps the author of the correction is not the variable at
all, and the agent resists correction regardless of who writes it.** Every
experiment run so far varied *who wrote the steer*. None measured *whether the
agent was receptive to any steer*. Those are different questions and only the
first was asked.

### What the merge should actually be

Internal signals were put into **detection**, and measured losing twice: Tier 1
harmful (§3.4), Tier 2 beaten 19,000× by a string comparison (§4.11). That was
the wrong socket for them.

Detection already works externally at 98.1% F1. The thing an external supervisor
**cannot** see is whether a correction will land — and that is an internal state,
which is exactly what ESR studies.

> **External supervisor → detects the failure.
> Internal signal → predicts whether the steer will be resisted, and therefore
> whether to INJECT or ESCALATE.**

This is motivated by our own measurements rather than by the paper's framing, and
it does **not** require the internal signal to be a good detector — which is
precisely what two experiments established it is not.

### Attempted 2026-08-16, and not measurable — for a reason worth more than the answer

`evals/measure_resistance.py` ran it against a real 7B over 12 tasks. **A
predictor needs a positive class, and this agent supplied one example of
compliance in forty-one chances** (§3.5). There is no variance for an internal
signal to predict. The merge was not refuted; it could not be evaluated.

That is the ESR phenomenon showing up directly in our own system rather than in a
citation. The paper describes models with enough internal consistency to resist
steering; this agent resisted **97.6%** of corrections, from every rung of the
ladder, across every task shape we could construct.

It also changes what the merge would have to be for. Predicting *whether* a steer
lands is uninteresting when the answer is almost always "it does not". The open
question is now **whether any correction lands at all, and what would make one
land** — a question about the intervention mechanism, not about which signal
predicts its success. Injecting a system message into a conversation may simply
be too weak an intervention, and nothing in this project has tested a stronger
one.

---

## 7. What would move it forward, in order

0. **Test the two untested adapters** (§8.3) and **redact secrets** (§8.5).
   These are the gaps between what the README claims and what is demonstrated,
   and they are cheap.
1. **Build the ESR merge** (§6.5) — use an internal signal to predict *steering
   resistance*, not to detect failures. Untested, and the one idea our own data
   argues for rather than borrows.
2. **Settle §3.3 with a larger model.** If a 7B closes the gap it is a model-size
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

## 8. What has NOT been done

Written deliberately, because a report that only lists what was built is a sales
document. Each entry below was verified against the code while writing this, not
recalled.

### 8.1 The central premise is disproven at every size we can test — **SETTLED**

The claim is that a *separate reasoning model* writes better corrections than a
fixed rule. Both a 3B and a 7B have now been run, and **both lost to the
templates** (§3.3). Scaling bought 5–15 points against a 70-point deficit.

The honest product is therefore **the deterministic ladder plus the detectors**.
The reasoning-model layer does not earn its place at any size testable here.

Still genuinely open: frontier reasoning models, for want of credits rather than
want of trying.

### 8.2 Half the recovery loop has never been real

Even in the one real-model test, whether the agent **obeys** a steer came from the
scenario's synthetic `responds_to` field. The supervisor half is now real; the
agent half never has been. `steering_usable = 100%` also remains circular by
construction — the rubric was written alongside the templates it scores — and no
human has ever assessed steering quality.

### 8.3 Two of the three advertised runtimes were untested — **CLOSED**

`openai_sdk` and `langgraph` had never been executed by any test. Writing those
tests found **four real bugs in about twenty minutes**, three in shipped code —
see §4.10. Both adapters now have coverage.

### 8.4 No CI, one machine, one Python — **CLOSED**

CI now runs the suite on **3.9 / 3.11 / 3.13 × Linux / Windows / macOS**, plus a
stdlib-only import check, the benchmark floors from `baseline.json`, and the
three offline demos. The matrix is deliberately the *claim* (`requires-python
>=3.9`) rather than a convenient subset.

### 8.5 Secrets were not redacted anywhere — **CLOSED**

`agentfuse/redact.py` now strips credentials on all three egress paths: the JSONL
trace, the supervisor prompt, and the escalation webhook. Redaction runs *before*
truncation, so a secret cut in half by a length limit cannot escape as a
fragment.

**The residual risk is real and stated:** a credential with no recognisable
format, no naming context, and under 32 characters is indistinguishable from
ordinary text and **will survive**. `escalation_include_agent_text=False` remains
the only complete answer for a sensitive deployment.

### 8.6 The public face was a month stale — **CLOSED**

The dashboard and GIF were rebuilt on **2026-08-14** after a month of drift, and
rebuilding them immediately found a real bug that a green test suite never would
have: the dashboard rendered **"Goal Drift — 0 TRIP · 0 HEAL"**.

`demo_drift.py` had not tripped since local embeddings landed. It hard-coded
`drift_threshold=0.45`, which is unreachable once similarities come from
embeddings (0.6–0.8) — **the identical mistake already found and fixed in the
eval harness, never fixed in the demo.** It exited 0 the whole time, so CI was
green while one of the three headline demos demonstrated nothing. CI now requires
each demo to actually print a trip.

The dashboard also now carries **a real Qwen2.5-7B run the breaker did not
rescue**: caught three times, steered three times, ignored every time. It stays on
the front page because it ends badly — that is consistent with §3.3, and a
dashboard showing only the runs that end well is a demo, not an observability
tool.

Rebuilding it also caught a **false claim**: the page advertised *"Real GPT Run —
Live GPT model driving a real agent; supervised and self-healed."* That trace has
**two records**, a header and one route event — the run that died on `429
insufficient_quota`. No GPT model has ever driven an agent here. Removed.

**It is no longer rebuilt by hand.** CI regenerates and publishes the dashboard on
every push to main that passes the test matrix, the benchmark floors and the demo
gate, committing only when the page actually changed. Making that safe required
fixing determinism twice — embedded wall-clock timestamps, then CRLF/LF — both
invisible while rebuilds were manual, both load-bearing the moment a machine
started committing the output.

### 8.7 Known rough edges, unaddressed

- **Checkpoints lose recent history.** `state()` saves counters, not the event
  list, so after a restore the supervisor's first snapshot has an empty
  `recent_events` and reasons with less context than it would have had.
- **No checkpoint retention.** The SQLite file grows without bound; nothing
  prunes finished runs.
- **`JSONMemory._flush()` rewrites the entire file on every write** — O(n) per
  record. Fine at hundreds, not at hundreds of thousands.
- **`QdrantMemory` is barely exercised** — it had two never-executed bugs when
  first run, and it is still only lightly covered.
- **No webhook authentication** beyond custom headers — no HMAC signing, so a
  receiver cannot verify the escalation came from us.
- **Shared-monitor multi-agent is unsolved.** Locks made it safe; they did not
  make it meaningful. One monitor per agent run is still the only supported model.
- **Async is untested under load.** `observe()` is synchronous and called from
  async hooks. It works; it has never been profiled with concurrent agents.

### 8.8 Phase 4 — complete, and both internal tiers ship OFF

All three tiers are built and measured. **Tier 1 costs 10.8 F1 when enabled
(§3.4); Tier 2 is 19,000× more expensive than the string comparison that already
catches the same thing (§4.11).** Both remain in the tree as measured, opt-in
research tools.

The honest summary of Phase 4: **reading the model's insides did not beat reading
its behaviour.** That is a real answer to the question the phase existed to ask,
and it is worth more than a detector nobody should switch on.

The signal ladder — logprob-based confidence, self-probing, activation probes —
has **zero lines of code**. It was deferred deliberately: it layers a research
direction on top of the steering premise in §8.1, which is unproven.

### 8.9 Genuinely blocked by this environment

Short, and worth separating from the above so the distinction stays honest:

- **No OpenAI credits.** A money constraint, not a hardware one. $0 has ever been
  spent on this project.
- ~~`torch` is broken~~ — **fixed 2026-08-13.** The cause was never CUDA: an
  interrupted install left an orphaned package because Windows long paths are
  disabled under a deep Store path. A venv at a short path resolves it. This was
  the last item on this list that turned out not to be a real constraint.
- **A 70B-class model will not run** on 15.2 GB of RAM.

Everything else in §8 is a choice, not a limit.

---

## 9. Reproducing everything here

```bash
python evals/run_eval.py --generated 40 --json    # full suite + ablation
python evals/run_eval.py --generated 40 --sweep   # threshold sweeps
python evals/validity.py                          # checks on the benchmark itself
python evals/real_model.py --base-url …           # templates vs a real model
pytest evals/ -q                                  # 212-test CI gate
```

No API key required, nothing billed. `evals/baseline.json` records every floor,
every known weakness, and the reasoning behind each threshold.

---

*Methodology for the ablation design is adapted from AE Studio's research on
Endogenous Steering Resistance — methodology only, no code used or derived. Full
attribution in [CITATION.md](CITATION.md).*
