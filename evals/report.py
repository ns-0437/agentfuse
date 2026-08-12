"""Rendering for eval results — console, Markdown, and JSON.

The JSON artefact is the one that matters long-term: it is the regression
baseline, so a later change that quietly costs three points of recall shows up
as a diff instead of a surprise.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .metrics import Metrics, family_rates
from .schema import ScenarioResult, CostModel

BAR = "=" * 78
SUB = "-" * 78


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def console_report(full: Metrics, rows, results: list[ScenarioResult],
                   cost: CostModel) -> str:
    out: list[str] = []
    a = out.append

    a("")
    a(BAR)
    a("AGENTFUSE — DETECTION EVAL")
    a(f"scenarios: {full.n}   ·   cost model: {cost.label}")
    a(BAR)

    # -- headline ------------------------------------------------------
    a("")
    a("HEADLINE")
    a(SUB)
    a("  (95% Wilson intervals — a rate without an interval is a rumour)")
    a(f"  precision           {full.precision_ci().render():<28} trust a trip when we see one")
    a(f"  recall              {full.recall_ci().render():<28} real failures caught")
    a(f"  recall  [clustered] {full.recall_ci_clustered().render():<28} <- QUOTE THIS ONE")
    a(f"  FPR     [clustered] {full.fpr_ci_clustered().render():<28} design effect "
      f"{full.design_effect:.1f}x (ICC {full.intra_cluster_correlation:.2f})")
    a(f"  F1                  {_pct(full.f1)}")
    a(f"  false-positive rate {full.fpr_ci().render():<28} healthy runs halted")
    a(f"  attribution         {full.attribution_ci().render():<28} right detector for the failure")
    a(f"  confusion           TP={full.tp}  FP={full.fp}  FN={full.fn}  TN={full.tn}")
    if full.mean_steps_late is not None:
        a(f"  mean steps late     {full.mean_steps_late}          (onset -> trip)")
    a(f"  late detections     {full.detected_late}")
    a(f"  premature trips     {full.detected_premature}          (fired before the failure began)")
    a(f"  known-gap misses    {full.known_gap_misses}          (documented, expected)")

    # -- recovery ------------------------------------------------------
    if full.recovery_eligible:
        a("")
        a("RECOVERY  (does the steering actually fix anything?)")
        a(SUB)
        a(f"  recovery rate       {full.recovery_ci().render():<28} caught failures put back on track")
        a(f"  steering usable     {_pct(full.steering_usable_rate)}      instructions passing the quality rubric")
        a(f"  steering quality    {_pct(full.mean_steering_quality)}      mean rubric score")
        a(f"  escalated to human  {full.recovery_escalated:<5}          correct outcome, not a recovery")

    # -- economics -----------------------------------------------------
    a("")
    a("TOKEN ECONOMICS")
    a(SUB)
    a(f"  tokens saved by halting early      {full.tokens_saved:>9,}")
    a(f"  supervision cost                   {full.supervision_cost:>9,}")
    a(f"  NET benefit                        {full.net_tokens:>9,}")
    a(f"  ROI (saved / spent)                {full.roi:>9.2f}x")
    a(f"  waste if fully unsupervised        {full.wasted_if_unsupervised:>9,}")

    # -- per family ----------------------------------------------------
    a("")
    a("BY FAILURE FAMILY")
    a(SUB)
    a(f"  {'family':<12}{'recall':>9}{'precision':>11}   counts")
    for fam, r in family_rates(full).items():
        c = r["counts"]
        a(f"  {fam:<12}{r['recall']*100:>8.1f}%{r['precision']*100:>10.1f}%   "
          f"TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}")

    # -- ablation ------------------------------------------------------
    a("")
    a("ABLATION  (leave-one-out + rate-matched random control)")
    a(SUB)
    a(f"  {'variant':<30}{'recall':>8}{'prec':>8}{'F1':>8}{'ΔF1':>9}{'net tok':>11}")
    for row in rows:
        m = row.metrics
        d = "" if row.label == "full system" else f"{row.d_f1*100:+8.1f}"
        a(f"  {row.label:<30}{m.recall*100:>7.1f}%{m.precision*100:>7.1f}%"
          f"{m.f1*100:>7.1f}%{d:>9}{m.net_tokens:>11,}")

    # -- per scenario --------------------------------------------------
    a("")
    a("PER SCENARIO")
    a(SUB)
    a(f"  {'':<3}{'scenario':<34}{'expect':<8}{'got':<10}{'detector':<12}{'step'}")
    for r in sorted(results, key=lambda x: (x.should_trip is False, x.scenario_id)):
        mark = {"TP": "OK", "TN": "OK", "FP": "!!", "FN": "!!"}[r.outcome]
        if r.outcome == "FN" and r.known_gap:
            mark = "~~"
        expect = "TRIP" if r.should_trip else "quiet"
        got = r.outcome
        det = r.trip_detector or "-"
        step = r.trip_step_index if r.trip_step_index is not None else "-"
        a(f"  {mark:<3}{r.scenario_id:<34}{expect:<8}{got:<10}{det:<12}{step}")
    a("")
    a("  OK = correct    !! = error    ~~ = known gap (documented, not yet fixed)")

    # -- failures ------------------------------------------------------
    if full.failures:
        a("")
        a("ERRORS TO FIX")
        a(SUB)
        for f in full.failures:
            tag = "KNOWN GAP" if f["known_gap"] else "REGRESSION"
            if f["outcome"] == "FP":
                a(f"  [FP] {f['scenario']:<34} halted a healthy run "
                  f"(fired: {f['actual_detector']})")
            else:
                a(f"  [FN] {f['scenario']:<34} missed a real failure "
                  f"(expected: {f['expected_detector']})  [{tag}]")

    a("")
    a(BAR)
    return "\n".join(out)


def markdown_report(full: Metrics, rows, results: list[ScenarioResult],
                    cost: CostModel) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md: list[str] = []
    a = md.append
    a("# AgentFuse — Detection Eval\n")
    a(f"_Generated {ts} · {full.n} scenarios · replay mode (deterministic, no API key)_\n")

    a("## Headline\n")
    a("| Metric | Value | Meaning |")
    a("|---|---|---|")
    a(f"| Precision | {_pct(full.precision)} | Can we trust a trip? |")
    a(f"| Recall | {_pct(full.recall)} | Do we catch real failures? |")
    a(f"| F1 | {_pct(full.f1)} | |")
    a(f"| False-positive rate | {_pct(full.false_positive_rate)} | How often we halt healthy runs |")
    a(f"| Attribution accuracy | {_pct(full.attribution_accuracy)} | Right detector for the failure |")
    a(f"| Confusion | TP={full.tp} FP={full.fp} FN={full.fn} TN={full.tn} | |")
    a(f"| Known-gap misses | {full.known_gap_misses} | Documented, not regressions |\n")

    a("## Token economics\n")
    a("| Metric | Tokens |")
    a("|---|---:|")
    a(f"| Saved by halting early | {full.tokens_saved:,} |")
    a(f"| Supervision cost | {full.supervision_cost:,} |")
    a(f"| **Net benefit** | **{full.net_tokens:,}** |")
    a(f"| ROI (saved/spent) | {full.roi:.2f}× |\n")

    a("## Ablation\n")
    a("Leave-one-out per detector, plus a rate-matched random control "
      "(methodology after AE Studio's ESR ablations).\n")
    a("| Variant | Recall | Precision | F1 | ΔF1 | Net tokens |")
    a("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row.metrics
        d = "—" if row.label == "full system" else f"{row.d_f1*100:+.1f}"
        a(f"| {row.label} | {_pct(m.recall)} | {_pct(m.precision)} | {_pct(m.f1)} | {d} | {m.net_tokens:,} |")
    a("")

    a("## By family\n")
    a("| Family | Recall | Precision | Counts |")
    a("|---|---:|---:|---|")
    for fam, r in family_rates(full).items():
        c = r["counts"]
        a(f"| {fam} | {r['recall']*100:.1f}% | {r['precision']*100:.1f}% | "
          f"TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']} |")
    a("")

    a("## Per scenario\n")
    a("| Scenario | Expected | Outcome | Detector | Step |")
    a("|---|---|---|---|---:|")
    for r in sorted(results, key=lambda x: (x.should_trip is False, x.scenario_id)):
        note = " (known gap)" if (r.outcome == "FN" and r.known_gap) else ""
        a(f"| `{r.scenario_id}` | {'trip' if r.should_trip else 'quiet'} | "
          f"{r.outcome}{note} | {r.trip_detector or '—'} | "
          f"{r.trip_step_index if r.trip_step_index is not None else '—'} |")
    a("")
    return "\n".join(md)


def write_artifacts(full: Metrics, rows, results: list[ScenarioResult],
                    cost: CostModel, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "replay",
        "cost_model": cost.label,
        "headline": full.to_dict(),
        "families": family_rates(full),
        "ablation": [r.to_dict() for r in rows],
        "scenarios": [
            {
                "id": r.scenario_id, "family": r.family, "outcome": r.outcome,
                "should_trip": r.should_trip, "tripped": r.tripped,
                "detector": r.trip_detector, "expected_detector": r.expected_detector,
                "trip_step_index": r.trip_step_index, "known_gap": r.known_gap,
                "tokens_saved": r.tokens_saved, "supervision_cost": r.supervision_cost,
                "steps_late": r.steps_late,
            } for r in results
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(markdown_report(full, rows, results, cost), encoding="utf-8")
    return payload
