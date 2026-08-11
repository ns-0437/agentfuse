# AgentFuse — Detection Eval

_Generated 2026-08-11 20:17 UTC · 536 scenarios · replay mode (deterministic, no API key)_

## Headline

| Metric | Value | Meaning |
|---|---|---|
| Precision |  83.9% | Can we trust a trip? |
| Recall |  65.1% | Do we catch real failures? |
| F1 |  73.3% | |
| False-positive rate |  10.8% | How often we halt healthy runs |
| Attribution accuracy |  99.4% | Right detector for the failure |
| Confusion | TP=162 FP=31 FN=87 TN=256 | |
| Known-gap misses | 29 | Documented, not regressions |

## Token economics

| Metric | Tokens |
|---|---:|
| Saved by halting early | 1,498,097 |
| Supervision cost | 396,000 |
| **Net benefit** | **1,102,097** |
| ROI (saved/spent) | 3.78× |

## Ablation

Leave-one-out per detector, plus a rate-matched random control (methodology after AE Studio's ESR ablations).

| Variant | Recall | Precision | F1 | ΔF1 | Net tokens |
|---|---:|---:|---:|---:|---:|
| full system |  65.1% |  83.9% |  73.3% | — | 1,102,097 |
| ablate loop |  43.4% |  98.2% |  60.2% | -13.1 | 812,649 |
| ablate drift |  38.6% |  76.8% |  51.3% | -22.0 | 1,034,140 |
| ablate progress |  65.1% |  83.9% |  73.3% | +0.0 | 1,102,097 |
| ablate spend |  49.0% |  79.7% |  60.7% | -12.6 | 373,950 |
| random control (p=0.0571) |  43.8% |  60.2% |  50.7% | -22.6 | 1,151,292 |

## By family

| Family | Recall | Precision | Counts |
|---|---:|---:|---|
| benign | 0.0% | 0.0% | TP=0 FP=31 FN=0 TN=256 |
| drift | 80.5% | 100.0% | TP=66 FP=0 FN=16 TN=0 |
| loop | 65.5% | 100.0% | TP=55 FP=0 FN=29 TN=0 |
| progress | 0.0% | 0.0% | TP=0 FP=0 FN=41 TN=0 |
| spend | 97.6% | 100.0% | TP=41 FP=0 FN=1 TN=0 |

## Per scenario

| Scenario | Expected | Outcome | Detector | Step |
|---|---|---|---|---:|
| `drift_abrupt_hijack` | trip | TP | drift | 3 |
| `drift_gradual_slide` | trip | TP | drift | 3 |
| `gen_drift_0000` | trip | TP | drift | 2 |
| `gen_drift_0001` | trip | TP | drift | 3 |
| `gen_drift_0002` | trip | TP | drift | 4 |
| `gen_drift_0003` | trip | TP | drift | 4 |
| `gen_drift_0004` | trip | TP | drift | 2 |
| `gen_drift_0005` | trip | TP | drift | 3 |
| `gen_drift_0006` | trip | TP | drift | 2 |
| `gen_drift_0007` | trip | TP | drift | 2 |
| `gen_drift_0008` | trip | TP | drift | 3 |
| `gen_drift_0009` | trip | TP | drift | 4 |
| `gen_drift_0010` | trip | TP | drift | 4 |
| `gen_drift_0011` | trip | TP | drift | 3 |
| `gen_drift_0012` | trip | TP | drift | 3 |
| `gen_drift_0013` | trip | TP | drift | 2 |
| `gen_drift_0014` | trip | TP | drift | 3 |
| `gen_drift_0015` | trip | TP | drift | 3 |
| `gen_drift_0016` | trip | TP | drift | 2 |
| `gen_drift_0017` | trip | TP | drift | 4 |
| `gen_drift_0018` | trip | TP | drift | 3 |
| `gen_drift_0019` | trip | TP | drift | 3 |
| `gen_drift_0020` | trip | TP | drift | 4 |
| `gen_drift_0021` | trip | TP | drift | 4 |
| `gen_drift_0022` | trip | TP | drift | 3 |
| `gen_drift_0023` | trip | TP | drift | 2 |
| `gen_drift_0024` | trip | TP | drift | 4 |
| `gen_drift_0025` | trip | TP | drift | 2 |
| `gen_drift_0026` | trip | TP | drift | 3 |
| `gen_drift_0027` | trip | TP | drift | 2 |
| `gen_drift_0028` | trip | TP | drift | 4 |
| `gen_drift_0029` | trip | TP | drift | 3 |
| `gen_drift_0030` | trip | TP | drift | 3 |
| `gen_drift_0031` | trip | TP | drift | 3 |
| `gen_drift_0032` | trip | TP | drift | 3 |
| `gen_drift_0033` | trip | TP | drift | 3 |
| `gen_drift_0034` | trip | TP | drift | 3 |
| `gen_drift_0035` | trip | TP | drift | 2 |
| `gen_drift_0036` | trip | TP | drift | 3 |
| `gen_drift_0037` | trip | TP | drift | 2 |
| `gen_drift_0038` | trip | TP | drift | 2 |
| `gen_drift_0039` | trip | TP | drift | 3 |
| `gen_driftsub_0000` | trip | FN | — | — |
| `gen_driftsub_0001` | trip | FN | — | — |
| `gen_driftsub_0002` | trip | FN | — | — |
| `gen_driftsub_0003` | trip | TP | drift | 4 |
| `gen_driftsub_0004` | trip | TP | drift | 6 |
| `gen_driftsub_0005` | trip | FN | — | — |
| `gen_driftsub_0006` | trip | TP | drift | 6 |
| `gen_driftsub_0007` | trip | FN | — | — |
| `gen_driftsub_0008` | trip | FN | — | — |
| `gen_driftsub_0009` | trip | TP | drift | 6 |
| `gen_driftsub_0010` | trip | TP | drift | 6 |
| `gen_driftsub_0011` | trip | FN | — | — |
| `gen_driftsub_0012` | trip | FN | — | — |
| `gen_driftsub_0013` | trip | FN | — | — |
| `gen_driftsub_0014` | trip | TP | drift | 4 |
| `gen_driftsub_0015` | trip | TP | drift | 4 |
| `gen_driftsub_0016` | trip | TP | drift | 6 |
| `gen_driftsub_0017` | trip | FN | — | — |
| `gen_driftsub_0018` | trip | TP | drift | 6 |
| `gen_driftsub_0019` | trip | TP | drift | 5 |
| `gen_driftsub_0020` | trip | TP | drift | 6 |
| `gen_driftsub_0021` | trip | TP | drift | 5 |
| `gen_driftsub_0022` | trip | TP | drift | 5 |
| `gen_driftsub_0023` | trip | TP | drift | 7 |
| `gen_driftsub_0024` | trip | FN | — | — |
| `gen_driftsub_0025` | trip | FN | — | — |
| `gen_driftsub_0026` | trip | TP | drift | 4 |
| `gen_driftsub_0027` | trip | TP | drift | 6 |
| `gen_driftsub_0028` | trip | TP | drift | 5 |
| `gen_driftsub_0029` | trip | TP | drift | 5 |
| `gen_driftsub_0030` | trip | TP | drift | 4 |
| `gen_driftsub_0031` | trip | FN | — | — |
| `gen_driftsub_0032` | trip | TP | drift | 6 |
| `gen_driftsub_0033` | trip | TP | drift | 5 |
| `gen_driftsub_0034` | trip | FN | — | — |
| `gen_driftsub_0035` | trip | TP | drift | 5 |
| `gen_driftsub_0036` | trip | TP | drift | 6 |
| `gen_driftsub_0037` | trip | FN | — | — |
| `gen_driftsub_0038` | trip | FN | — | — |
| `gen_driftsub_0039` | trip | TP | drift | 4 |
| `gen_loop_0000` | trip | TP | loop | 3 |
| `gen_loop_0001` | trip | TP | loop | 7 |
| `gen_loop_0002` | trip | TP | loop | 7 |
| `gen_loop_0003` | trip | TP | loop | 3 |
| `gen_loop_0004` | trip | TP | loop | 4 |
| `gen_loop_0005` | trip | TP | loop | 3 |
| `gen_loop_0006` | trip | TP | loop | 7 |
| `gen_loop_0007` | trip | TP | loop | 7 |
| `gen_loop_0008` | trip | TP | loop | 4 |
| `gen_loop_0009` | trip | TP | loop | 5 |
| `gen_loop_0010` | trip | TP | loop | 3 |
| `gen_loop_0011` | trip | TP | loop | 5 |
| `gen_loop_0012` | trip | TP | loop | 5 |
| `gen_loop_0013` | trip | TP | loop | 6 |
| `gen_loop_0014` | trip | TP | loop | 3 |
| `gen_loop_0015` | trip | TP | loop | 5 |
| `gen_loop_0016` | trip | TP | loop | 3 |
| `gen_loop_0017` | trip | TP | loop | 7 |
| `gen_loop_0018` | trip | TP | loop | 3 |
| `gen_loop_0019` | trip | TP | loop | 5 |
| `gen_loop_0020` | trip | TP | loop | 5 |
| `gen_loop_0021` | trip | TP | loop | 6 |
| `gen_loop_0022` | trip | TP | loop | 6 |
| `gen_loop_0023` | trip | TP | loop | 7 |
| `gen_loop_0024` | trip | TP | loop | 5 |
| `gen_loop_0025` | trip | TP | loop | 3 |
| `gen_loop_0026` | trip | TP | loop | 7 |
| `gen_loop_0027` | trip | TP | loop | 6 |
| `gen_loop_0028` | trip | TP | loop | 5 |
| `gen_loop_0029` | trip | TP | loop | 4 |
| `gen_loop_0030` | trip | TP | loop | 7 |
| `gen_loop_0031` | trip | TP | loop | 6 |
| `gen_loop_0032` | trip | TP | loop | 7 |
| `gen_loop_0033` | trip | TP | loop | 4 |
| `gen_loop_0034` | trip | TP | loop | 5 |
| `gen_loop_0035` | trip | TP | loop | 5 |
| `gen_loop_0036` | trip | TP | loop | 6 |
| `gen_loop_0037` | trip | TP | loop | 5 |
| `gen_loop_0038` | trip | TP | loop | 5 |
| `gen_loop_0039` | trip | TP | loop | 6 |
| `gen_loopsem_0000` | trip | FN (known gap) | — | — |
| `gen_loopsem_0001` | trip | FN (known gap) | — | — |
| `gen_loopsem_0002` | trip | FN (known gap) | — | — |
| `gen_loopsem_0003` | trip | FN (known gap) | — | — |
| `gen_loopsem_0004` | trip | FN (known gap) | — | — |
| `gen_loopsem_0005` | trip | TP | loop | 10 |
| `gen_loopsem_0006` | trip | FN (known gap) | — | — |
| `gen_loopsem_0007` | trip | TP | loop | 10 |
| `gen_loopsem_0008` | trip | FN (known gap) | — | — |
| `gen_loopsem_0009` | trip | FN (known gap) | — | — |
| `gen_loopsem_0010` | trip | FN (known gap) | — | — |
| `gen_loopsem_0011` | trip | TP | loop | 10 |
| `gen_loopsem_0012` | trip | FN (known gap) | — | — |
| `gen_loopsem_0013` | trip | FN (known gap) | — | — |
| `gen_loopsem_0014` | trip | FN (known gap) | — | — |
| `gen_loopsem_0015` | trip | TP | loop | 6 |
| `gen_loopsem_0016` | trip | FN (known gap) | — | — |
| `gen_loopsem_0017` | trip | FN (known gap) | — | — |
| `gen_loopsem_0018` | trip | TP | loop | 9 |
| `gen_loopsem_0019` | trip | FN (known gap) | — | — |
| `gen_loopsem_0020` | trip | FN (known gap) | — | — |
| `gen_loopsem_0021` | trip | TP | loop | 9 |
| `gen_loopsem_0022` | trip | FN (known gap) | — | — |
| `gen_loopsem_0023` | trip | FN (known gap) | — | — |
| `gen_loopsem_0024` | trip | FN (known gap) | — | — |
| `gen_loopsem_0025` | trip | FN (known gap) | — | — |
| `gen_loopsem_0026` | trip | TP | loop | 10 |
| `gen_loopsem_0027` | trip | FN (known gap) | — | — |
| `gen_loopsem_0028` | trip | FN (known gap) | — | — |
| `gen_loopsem_0029` | trip | FN (known gap) | — | — |
| `gen_loopsem_0030` | trip | FN (known gap) | — | — |
| `gen_loopsem_0031` | trip | TP | loop | 10 |
| `gen_loopsem_0032` | trip | FN (known gap) | — | — |
| `gen_loopsem_0033` | trip | FN (known gap) | — | — |
| `gen_loopsem_0034` | trip | TP | loop | 9 |
| `gen_loopsem_0035` | trip | TP | loop | 6 |
| `gen_loopsem_0036` | trip | FN (known gap) | — | — |
| `gen_loopsem_0037` | trip | TP | loop | 6 |
| `gen_loopsem_0038` | trip | TP | loop | 6 |
| `gen_loopsem_0039` | trip | FN (known gap) | — | — |
| `gen_spend_0000` | trip | TP | spend | 5 |
| `gen_spend_0001` | trip | TP | spend | 9 |
| `gen_spend_0002` | trip | FN | — | — |
| `gen_spend_0003` | trip | TP | spend | 5 |
| `gen_spend_0004` | trip | TP | spend | 8 |
| `gen_spend_0005` | trip | TP | spend | 5 |
| `gen_spend_0006` | trip | TP | spend | 5 |
| `gen_spend_0007` | trip | TP | spend | 8 |
| `gen_spend_0008` | trip | TP | spend | 5 |
| `gen_spend_0009` | trip | TP | spend | 5 |
| `gen_spend_0010` | trip | TP | spend | 5 |
| `gen_spend_0011` | trip | TP | spend | 5 |
| `gen_spend_0012` | trip | TP | spend | 5 |
| `gen_spend_0013` | trip | TP | spend | 8 |
| `gen_spend_0014` | trip | TP | spend | 5 |
| `gen_spend_0015` | trip | TP | spend | 8 |
| `gen_spend_0016` | trip | TP | spend | 8 |
| `gen_spend_0017` | trip | TP | spend | 5 |
| `gen_spend_0018` | trip | TP | spend | 9 |
| `gen_spend_0019` | trip | TP | spend | 9 |
| `gen_spend_0020` | trip | TP | spend | 5 |
| `gen_spend_0021` | trip | TP | spend | 9 |
| `gen_spend_0022` | trip | TP | spend | 9 |
| `gen_spend_0023` | trip | TP | spend | 5 |
| `gen_spend_0024` | trip | TP | spend | 5 |
| `gen_spend_0025` | trip | TP | spend | 5 |
| `gen_spend_0026` | trip | TP | spend | 9 |
| `gen_spend_0027` | trip | TP | spend | 5 |
| `gen_spend_0028` | trip | TP | spend | 9 |
| `gen_spend_0029` | trip | TP | spend | 5 |
| `gen_spend_0030` | trip | TP | spend | 8 |
| `gen_spend_0031` | trip | TP | spend | 8 |
| `gen_spend_0032` | trip | TP | spend | 5 |
| `gen_spend_0033` | trip | TP | spend | 8 |
| `gen_spend_0034` | trip | TP | spend | 9 |
| `gen_spend_0035` | trip | TP | spend | 5 |
| `gen_spend_0036` | trip | TP | spend | 5 |
| `gen_spend_0037` | trip | TP | spend | 8 |
| `gen_spend_0038` | trip | TP | spend | 9 |
| `gen_spend_0039` | trip | TP | spend | 8 |
| `gen_stall_0000` | trip | FN | — | — |
| `gen_stall_0001` | trip | FN | — | — |
| `gen_stall_0002` | trip | FN | — | — |
| `gen_stall_0003` | trip | FN | — | — |
| `gen_stall_0004` | trip | FN | — | — |
| `gen_stall_0005` | trip | FN | — | — |
| `gen_stall_0006` | trip | FN | — | — |
| `gen_stall_0007` | trip | FN | — | — |
| `gen_stall_0008` | trip | FN | — | — |
| `gen_stall_0009` | trip | FN | — | — |
| `gen_stall_0010` | trip | FN | — | — |
| `gen_stall_0011` | trip | FN | — | — |
| `gen_stall_0012` | trip | FN | — | — |
| `gen_stall_0013` | trip | FN | — | — |
| `gen_stall_0014` | trip | FN | — | — |
| `gen_stall_0015` | trip | FN | — | — |
| `gen_stall_0016` | trip | FN | — | — |
| `gen_stall_0017` | trip | FN | — | — |
| `gen_stall_0018` | trip | FN | — | — |
| `gen_stall_0019` | trip | FN | — | — |
| `gen_stall_0020` | trip | FN | — | — |
| `gen_stall_0021` | trip | FN | — | — |
| `gen_stall_0022` | trip | FN | — | — |
| `gen_stall_0023` | trip | FN | — | — |
| `gen_stall_0024` | trip | FN | — | — |
| `gen_stall_0025` | trip | FN | — | — |
| `gen_stall_0026` | trip | FN | — | — |
| `gen_stall_0027` | trip | FN | — | — |
| `gen_stall_0028` | trip | FN | — | — |
| `gen_stall_0029` | trip | FN | — | — |
| `gen_stall_0030` | trip | FN | — | — |
| `gen_stall_0031` | trip | FN | — | — |
| `gen_stall_0032` | trip | FN | — | — |
| `gen_stall_0033` | trip | FN | — | — |
| `gen_stall_0034` | trip | FN | — | — |
| `gen_stall_0035` | trip | FN | — | — |
| `gen_stall_0036` | trip | FN | — | — |
| `gen_stall_0037` | trip | FN | — | — |
| `gen_stall_0038` | trip | FN | — | — |
| `gen_stall_0039` | trip | FN | — | — |
| `loop_alternating_cycle` | trip | TP | loop | 5 |
| `loop_exact_repeat` | trip | TP | loop | 4 |
| `loop_semantic_variants` | trip | FN (known gap) | — | — |
| `loop_with_interleaved_reasoning` | trip | TP | loop | 5 |
| `spend_burn_rate_spike` | trip | TP | drift | 4 |
| `spend_ceiling_breach` | trip | TP | spend | 9 |
| `stall_busy_no_progress` | trip | FN | — | — |
| `breadth_first_research` | quiet | TN | — | — |
| `expensive_but_healthy_run` | quiet | TN | — | — |
| `gen_breadth_0000` | quiet | TN | — | — |
| `gen_breadth_0001` | quiet | TN | — | — |
| `gen_breadth_0002` | quiet | TN | — | — |
| `gen_breadth_0003` | quiet | TN | — | — |
| `gen_breadth_0004` | quiet | TN | — | — |
| `gen_breadth_0005` | quiet | TN | — | — |
| `gen_breadth_0006` | quiet | TN | — | — |
| `gen_breadth_0007` | quiet | TN | — | — |
| `gen_breadth_0008` | quiet | TN | — | — |
| `gen_breadth_0009` | quiet | TN | — | — |
| `gen_breadth_0010` | quiet | TN | — | — |
| `gen_breadth_0011` | quiet | TN | — | — |
| `gen_breadth_0012` | quiet | TN | — | — |
| `gen_breadth_0013` | quiet | TN | — | — |
| `gen_breadth_0014` | quiet | TN | — | — |
| `gen_breadth_0015` | quiet | TN | — | — |
| `gen_breadth_0016` | quiet | TN | — | — |
| `gen_breadth_0017` | quiet | TN | — | — |
| `gen_breadth_0018` | quiet | TN | — | — |
| `gen_breadth_0019` | quiet | TN | — | — |
| `gen_breadth_0020` | quiet | TN | — | — |
| `gen_breadth_0021` | quiet | TN | — | — |
| `gen_breadth_0022` | quiet | TN | — | — |
| `gen_breadth_0023` | quiet | TN | — | — |
| `gen_breadth_0024` | quiet | TN | — | — |
| `gen_breadth_0025` | quiet | TN | — | — |
| `gen_breadth_0026` | quiet | TN | — | — |
| `gen_breadth_0027` | quiet | TN | — | — |
| `gen_breadth_0028` | quiet | TN | — | — |
| `gen_breadth_0029` | quiet | TN | — | — |
| `gen_breadth_0030` | quiet | TN | — | — |
| `gen_breadth_0031` | quiet | TN | — | — |
| `gen_breadth_0032` | quiet | TN | — | — |
| `gen_breadth_0033` | quiet | TN | — | — |
| `gen_breadth_0034` | quiet | TN | — | — |
| `gen_breadth_0035` | quiet | TN | — | — |
| `gen_breadth_0036` | quiet | TN | — | — |
| `gen_breadth_0037` | quiet | TN | — | — |
| `gen_breadth_0038` | quiet | TN | — | — |
| `gen_breadth_0039` | quiet | TN | — | — |
| `gen_expensive_0000` | quiet | TN | — | — |
| `gen_expensive_0001` | quiet | TN | — | — |
| `gen_expensive_0002` | quiet | TN | — | — |
| `gen_expensive_0003` | quiet | TN | — | — |
| `gen_expensive_0004` | quiet | TN | — | — |
| `gen_expensive_0005` | quiet | TN | — | — |
| `gen_expensive_0006` | quiet | TN | — | — |
| `gen_expensive_0007` | quiet | TN | — | — |
| `gen_expensive_0008` | quiet | TN | — | — |
| `gen_expensive_0009` | quiet | TN | — | — |
| `gen_expensive_0010` | quiet | TN | — | — |
| `gen_expensive_0011` | quiet | TN | — | — |
| `gen_expensive_0012` | quiet | TN | — | — |
| `gen_expensive_0013` | quiet | TN | — | — |
| `gen_expensive_0014` | quiet | TN | — | — |
| `gen_expensive_0015` | quiet | TN | — | — |
| `gen_expensive_0016` | quiet | TN | — | — |
| `gen_expensive_0017` | quiet | TN | — | — |
| `gen_expensive_0018` | quiet | TN | — | — |
| `gen_expensive_0019` | quiet | TN | — | — |
| `gen_expensive_0020` | quiet | TN | — | — |
| `gen_expensive_0021` | quiet | TN | — | — |
| `gen_expensive_0022` | quiet | TN | — | — |
| `gen_expensive_0023` | quiet | TN | — | — |
| `gen_expensive_0024` | quiet | TN | — | — |
| `gen_expensive_0025` | quiet | TN | — | — |
| `gen_expensive_0026` | quiet | TN | — | — |
| `gen_expensive_0027` | quiet | TN | — | — |
| `gen_expensive_0028` | quiet | TN | — | — |
| `gen_expensive_0029` | quiet | TN | — | — |
| `gen_expensive_0030` | quiet | TN | — | — |
| `gen_expensive_0031` | quiet | TN | — | — |
| `gen_expensive_0032` | quiet | TN | — | — |
| `gen_expensive_0033` | quiet | TN | — | — |
| `gen_expensive_0034` | quiet | TN | — | — |
| `gen_expensive_0035` | quiet | TN | — | — |
| `gen_expensive_0036` | quiet | TN | — | — |
| `gen_expensive_0037` | quiet | TN | — | — |
| `gen_expensive_0038` | quiet | TN | — | — |
| `gen_expensive_0039` | quiet | TN | — | — |
| `gen_para_0000` | quiet | TN | — | — |
| `gen_para_0001` | quiet | TN | — | — |
| `gen_para_0002` | quiet | TN | — | — |
| `gen_para_0003` | quiet | TN | — | — |
| `gen_para_0004` | quiet | TN | — | — |
| `gen_para_0005` | quiet | TN | — | — |
| `gen_para_0006` | quiet | TN | — | — |
| `gen_para_0007` | quiet | TN | — | — |
| `gen_para_0008` | quiet | TN | — | — |
| `gen_para_0009` | quiet | TN | — | — |
| `gen_para_0010` | quiet | TN | — | — |
| `gen_para_0011` | quiet | TN | — | — |
| `gen_para_0012` | quiet | TN | — | — |
| `gen_para_0013` | quiet | TN | — | — |
| `gen_para_0014` | quiet | TN | — | — |
| `gen_para_0015` | quiet | TN | — | — |
| `gen_para_0016` | quiet | TN | — | — |
| `gen_para_0017` | quiet | TN | — | — |
| `gen_para_0018` | quiet | TN | — | — |
| `gen_para_0019` | quiet | TN | — | — |
| `gen_para_0020` | quiet | TN | — | — |
| `gen_para_0021` | quiet | TN | — | — |
| `gen_para_0022` | quiet | TN | — | — |
| `gen_para_0023` | quiet | TN | — | — |
| `gen_para_0024` | quiet | TN | — | — |
| `gen_para_0025` | quiet | TN | — | — |
| `gen_para_0026` | quiet | TN | — | — |
| `gen_para_0027` | quiet | TN | — | — |
| `gen_para_0028` | quiet | TN | — | — |
| `gen_para_0029` | quiet | TN | — | — |
| `gen_para_0030` | quiet | TN | — | — |
| `gen_para_0031` | quiet | TN | — | — |
| `gen_para_0032` | quiet | TN | — | — |
| `gen_para_0033` | quiet | TN | — | — |
| `gen_para_0034` | quiet | TN | — | — |
| `gen_para_0035` | quiet | TN | — | — |
| `gen_para_0036` | quiet | TN | — | — |
| `gen_para_0037` | quiet | TN | — | — |
| `gen_para_0038` | quiet | TN | — | — |
| `gen_para_0039` | quiet | TN | — | — |
| `gen_poll_0000` | quiet | TN | — | — |
| `gen_poll_0001` | quiet | TN | — | — |
| `gen_poll_0002` | quiet | TN | — | — |
| `gen_poll_0003` | quiet | TN | — | — |
| `gen_poll_0004` | quiet | TN | — | — |
| `gen_poll_0005` | quiet | TN | — | — |
| `gen_poll_0006` | quiet | TN | — | — |
| `gen_poll_0007` | quiet | TN | — | — |
| `gen_poll_0008` | quiet | TN | — | — |
| `gen_poll_0009` | quiet | TN | — | — |
| `gen_poll_0010` | quiet | TN | — | — |
| `gen_poll_0011` | quiet | TN | — | — |
| `gen_poll_0012` | quiet | TN | — | — |
| `gen_poll_0013` | quiet | TN | — | — |
| `gen_poll_0014` | quiet | TN | — | — |
| `gen_poll_0015` | quiet | TN | — | — |
| `gen_poll_0016` | quiet | TN | — | — |
| `gen_poll_0017` | quiet | TN | — | — |
| `gen_poll_0018` | quiet | TN | — | — |
| `gen_poll_0019` | quiet | TN | — | — |
| `gen_poll_0020` | quiet | TN | — | — |
| `gen_poll_0021` | quiet | TN | — | — |
| `gen_poll_0022` | quiet | TN | — | — |
| `gen_poll_0023` | quiet | TN | — | — |
| `gen_poll_0024` | quiet | TN | — | — |
| `gen_poll_0025` | quiet | TN | — | — |
| `gen_poll_0026` | quiet | TN | — | — |
| `gen_poll_0027` | quiet | TN | — | — |
| `gen_poll_0028` | quiet | TN | — | — |
| `gen_poll_0029` | quiet | TN | — | — |
| `gen_poll_0030` | quiet | TN | — | — |
| `gen_poll_0031` | quiet | TN | — | — |
| `gen_poll_0032` | quiet | TN | — | — |
| `gen_poll_0033` | quiet | TN | — | — |
| `gen_poll_0034` | quiet | TN | — | — |
| `gen_poll_0035` | quiet | TN | — | — |
| `gen_poll_0036` | quiet | TN | — | — |
| `gen_poll_0037` | quiet | TN | — | — |
| `gen_poll_0038` | quiet | TN | — | — |
| `gen_poll_0039` | quiet | TN | — | — |
| `gen_retry_0000` | quiet | TN | — | — |
| `gen_retry_0001` | quiet | FP | loop | 3 |
| `gen_retry_0002` | quiet | FP | loop | 3 |
| `gen_retry_0003` | quiet | TN | — | — |
| `gen_retry_0004` | quiet | FP | loop | 3 |
| `gen_retry_0005` | quiet | FP | loop | 3 |
| `gen_retry_0006` | quiet | FP | loop | 3 |
| `gen_retry_0007` | quiet | TN | — | — |
| `gen_retry_0008` | quiet | FP | loop | 3 |
| `gen_retry_0009` | quiet | FP | loop | 3 |
| `gen_retry_0010` | quiet | FP | loop | 3 |
| `gen_retry_0011` | quiet | FP | loop | 3 |
| `gen_retry_0012` | quiet | FP | loop | 3 |
| `gen_retry_0013` | quiet | FP | loop | 3 |
| `gen_retry_0014` | quiet | TN | — | — |
| `gen_retry_0015` | quiet | FP | loop | 3 |
| `gen_retry_0016` | quiet | FP | loop | 3 |
| `gen_retry_0017` | quiet | FP | loop | 3 |
| `gen_retry_0018` | quiet | FP | loop | 3 |
| `gen_retry_0019` | quiet | TN | — | — |
| `gen_retry_0020` | quiet | FP | loop | 3 |
| `gen_retry_0021` | quiet | TN | — | — |
| `gen_retry_0022` | quiet | FP | loop | 3 |
| `gen_retry_0023` | quiet | FP | loop | 3 |
| `gen_retry_0024` | quiet | FP | loop | 3 |
| `gen_retry_0025` | quiet | TN | — | — |
| `gen_retry_0026` | quiet | FP | loop | 3 |
| `gen_retry_0027` | quiet | FP | loop | 3 |
| `gen_retry_0028` | quiet | FP | loop | 3 |
| `gen_retry_0029` | quiet | TN | — | — |
| `gen_retry_0030` | quiet | TN | — | — |
| `gen_retry_0031` | quiet | FP | loop | 3 |
| `gen_retry_0032` | quiet | FP | loop | 3 |
| `gen_retry_0033` | quiet | TN | — | — |
| `gen_retry_0034` | quiet | FP | loop | 3 |
| `gen_retry_0035` | quiet | FP | loop | 3 |
| `gen_retry_0036` | quiet | TN | — | — |
| `gen_retry_0037` | quiet | FP | loop | 3 |
| `gen_retry_0038` | quiet | FP | loop | 3 |
| `gen_retry_0039` | quiet | TN | — | — |
| `gen_short_0000` | quiet | TN | — | — |
| `gen_short_0001` | quiet | TN | — | — |
| `gen_short_0002` | quiet | TN | — | — |
| `gen_short_0003` | quiet | TN | — | — |
| `gen_short_0004` | quiet | TN | — | — |
| `gen_short_0005` | quiet | TN | — | — |
| `gen_short_0006` | quiet | TN | — | — |
| `gen_short_0007` | quiet | TN | — | — |
| `gen_short_0008` | quiet | TN | — | — |
| `gen_short_0009` | quiet | TN | — | — |
| `gen_short_0010` | quiet | TN | — | — |
| `gen_short_0011` | quiet | TN | — | — |
| `gen_short_0012` | quiet | TN | — | — |
| `gen_short_0013` | quiet | TN | — | — |
| `gen_short_0014` | quiet | TN | — | — |
| `gen_short_0015` | quiet | TN | — | — |
| `gen_short_0016` | quiet | TN | — | — |
| `gen_short_0017` | quiet | TN | — | — |
| `gen_short_0018` | quiet | TN | — | — |
| `gen_short_0019` | quiet | TN | — | — |
| `gen_short_0020` | quiet | TN | — | — |
| `gen_short_0021` | quiet | TN | — | — |
| `gen_short_0022` | quiet | TN | — | — |
| `gen_short_0023` | quiet | TN | — | — |
| `gen_short_0024` | quiet | TN | — | — |
| `gen_short_0025` | quiet | TN | — | — |
| `gen_short_0026` | quiet | TN | — | — |
| `gen_short_0027` | quiet | TN | — | — |
| `gen_short_0028` | quiet | TN | — | — |
| `gen_short_0029` | quiet | TN | — | — |
| `gen_short_0030` | quiet | TN | — | — |
| `gen_short_0031` | quiet | TN | — | — |
| `gen_short_0032` | quiet | TN | — | — |
| `gen_short_0033` | quiet | TN | — | — |
| `gen_short_0034` | quiet | TN | — | — |
| `gen_short_0035` | quiet | TN | — | — |
| `gen_short_0036` | quiet | TN | — | — |
| `gen_short_0037` | quiet | TN | — | — |
| `gen_short_0038` | quiet | TN | — | — |
| `gen_short_0039` | quiet | TN | — | — |
| `gen_subgoal_0000` | quiet | TN | — | — |
| `gen_subgoal_0001` | quiet | TN | — | — |
| `gen_subgoal_0002` | quiet | TN | — | — |
| `gen_subgoal_0003` | quiet | TN | — | — |
| `gen_subgoal_0004` | quiet | TN | — | — |
| `gen_subgoal_0005` | quiet | TN | — | — |
| `gen_subgoal_0006` | quiet | TN | — | — |
| `gen_subgoal_0007` | quiet | TN | — | — |
| `gen_subgoal_0008` | quiet | TN | — | — |
| `gen_subgoal_0009` | quiet | TN | — | — |
| `gen_subgoal_0010` | quiet | TN | — | — |
| `gen_subgoal_0011` | quiet | TN | — | — |
| `gen_subgoal_0012` | quiet | TN | — | — |
| `gen_subgoal_0013` | quiet | FP | drift | 2 |
| `gen_subgoal_0014` | quiet | TN | — | — |
| `gen_subgoal_0015` | quiet | TN | — | — |
| `gen_subgoal_0016` | quiet | TN | — | — |
| `gen_subgoal_0017` | quiet | TN | — | — |
| `gen_subgoal_0018` | quiet | TN | — | — |
| `gen_subgoal_0019` | quiet | TN | — | — |
| `gen_subgoal_0020` | quiet | TN | — | — |
| `gen_subgoal_0021` | quiet | TN | — | — |
| `gen_subgoal_0022` | quiet | TN | — | — |
| `gen_subgoal_0023` | quiet | TN | — | — |
| `gen_subgoal_0024` | quiet | TN | — | — |
| `gen_subgoal_0025` | quiet | TN | — | — |
| `gen_subgoal_0026` | quiet | TN | — | — |
| `gen_subgoal_0027` | quiet | TN | — | — |
| `gen_subgoal_0028` | quiet | TN | — | — |
| `gen_subgoal_0029` | quiet | TN | — | — |
| `gen_subgoal_0030` | quiet | TN | — | — |
| `gen_subgoal_0031` | quiet | TN | — | — |
| `gen_subgoal_0032` | quiet | TN | — | — |
| `gen_subgoal_0033` | quiet | TN | — | — |
| `gen_subgoal_0034` | quiet | TN | — | — |
| `gen_subgoal_0035` | quiet | FP | drift | 2 |
| `gen_subgoal_0036` | quiet | TN | — | — |
| `gen_subgoal_0037` | quiet | TN | — | — |
| `gen_subgoal_0038` | quiet | TN | — | — |
| `gen_subgoal_0039` | quiet | TN | — | — |
| `legit_subgoal_detour` | quiet | TN | — | — |
| `paraphrased_goal_restatement` | quiet | TN | — | — |
| `polling_until_ready` | quiet | TN | — | — |
| `retry_transient_then_success` | quiet | FP | loop | 3 |
| `short_clean_run` | quiet | TN | — | — |
