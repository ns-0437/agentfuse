# AgentFuse — Detection Eval

_Generated 2026-08-11 19:45 UTC · 16 scenarios · replay mode (deterministic, no API key)_

## Headline

| Metric | Value | Meaning |
|---|---|---|
| Precision |  63.6% | Can we trust a trip? |
| Recall |  77.8% | Do we catch real failures? |
| F1 |  70.0% | |
| False-positive rate |  57.1% | How often we halt healthy runs |
| Attribution accuracy |  71.4% | Right detector for the failure |
| Confusion | TP=7 FP=4 FN=2 TN=3 | |
| Known-gap misses | 1 | Documented, not regressions |

## Token economics

| Metric | Tokens |
|---|---:|
| Saved by halting early | 74,890 |
| Supervision cost | 25,500 |
| **Net benefit** | **49,390** |
| ROI (saved/spent) | 2.94× |

## Ablation

Leave-one-out per detector, plus a rate-matched random control (methodology after AE Studio's ESR ablations).

| Variant | Recall | Precision | F1 | ΔF1 | Net tokens |
|---|---:|---:|---:|---:|---:|
| full system |  77.8% |  63.6% |  70.0% | — | 49,390 |
| ablate loop |  55.6% |  62.5% |  58.8% | -11.2 | 46,790 |
| ablate drift |  55.6% |  83.3% |  66.7% | -3.3 | 18,200 |
| ablate progress |  77.8% |  63.6% |  70.0% | +0.0 | 49,390 |
| ablate spend |  66.7% |  60.0% |  63.2% | -6.8 | 49,490 |
| random control (p=0.1066) |  44.4% |  57.1% |  50.0% | -20.0 | 760 |

## By family

| Family | Recall | Precision | Counts |
|---|---:|---:|---|
| benign | 0.0% | 0.0% | TP=0 FP=4 FN=0 TN=3 |
| drift | 100.0% | 100.0% | TP=2 FP=0 FN=0 TN=0 |
| loop | 75.0% | 100.0% | TP=3 FP=0 FN=1 TN=0 |
| progress | 0.0% | 0.0% | TP=0 FP=0 FN=1 TN=0 |
| spend | 100.0% | 100.0% | TP=2 FP=0 FN=0 TN=0 |

## Per scenario

| Scenario | Expected | Outcome | Detector | Step |
|---|---|---|---|---:|
| `drift_abrupt_hijack` | trip | TP | drift | 2 |
| `drift_gradual_slide` | trip | TP | drift | 1 |
| `loop_alternating_cycle` | trip | TP | loop | 5 |
| `loop_exact_repeat` | trip | TP | loop | 4 |
| `loop_semantic_variants` | trip | FN (known gap) | — | — |
| `loop_with_interleaved_reasoning` | trip | TP | drift | 4 |
| `spend_burn_rate_spike` | trip | TP | drift | 1 |
| `spend_ceiling_breach` | trip | TP | spend | 9 |
| `stall_busy_no_progress` | trip | FN | — | — |
| `breadth_first_research` | quiet | TN | — | — |
| `expensive_but_healthy_run` | quiet | FP | drift | 6 |
| `legit_subgoal_detour` | quiet | FP | drift | 1 |
| `paraphrased_goal_restatement` | quiet | FP | drift | 1 |
| `polling_until_ready` | quiet | TN | — | — |
| `retry_transient_then_success` | quiet | FP | loop | 3 |
| `short_clean_run` | quiet | TN | — | — |
