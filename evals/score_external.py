"""Score independently captured, manually labelled AgentFuse traces offline.

Usage: python -m evals.score_external path/to/labels.json [--json]
The manifest is a list of {id, trace, label: {should_trip, note}} objects.
Trace paths are relative to the manifest. No model, API key, or service is used.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

# This command promises an offline replay even on a machine whose shell is
# configured for paid hosted inference. Set the guard before importing runner.
os.environ["AGENTFUSE_OFFLINE"] = "1"

from .runner import run_scenario
from .schema import Label
from .stats import wilson
from .trace_import import scenario_from_trace


def _validate_trace(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing trace: {path}")
    events = summaries = 0
    goals: list[str] = []
    open_calls: dict[str, int] = {}
    seen_summary = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed trace {path}:{number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"trace record must be an object: {path}:{number}")
        kind = record.get("kind")
        if seen_summary:
            raise ValueError(f"records after summary make trace ambiguous: {path}:{number}")
        # OBSERVE mode logs trips but leaves the agent alone. Those records are
        # valid evidence. A recovery/resume changes the trajectory being scored.
        if kind == "recovery" or (kind == "event" and
                record.get("type") in ("recovery", "resume")):
            raise ValueError(f"trace contains AgentFuse intervention: {path}:{number}; "
                             "capture without recovery or interruption")
        if record.get("kind") == "meta":
            goal = record.get("original_goal")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError(f"trace has no original goal: {path}:{number}")
            goals.append(goal)
        if kind == "event" and record.get("type") in ("tool_call", "tool_result"):
            event_type = record["type"]
            call_id = (record.get("meta") or {}).get("call_id")
            step = record.get("step")
            key = f"id:{call_id}" if call_id else f"step:{step}"
            if event_type == "tool_call":
                if open_calls.get(key, 0):
                    raise ValueError(f"overlapping tool calls share an identity: "
                                     f"{path}:{number}")
                open_calls[key] = open_calls.get(key, 0) + 1
            else:
                if open_calls.get(key, 0) == 0:
                    raise ValueError(f"unpaired tool result: {path}:{number}")
                open_calls[key] -= 1
        events += (record.get("kind") == "event"
                   and record.get("type") in ("tool_call", "llm_call"))
        summaries += kind == "summary"
        if kind == "summary" and record.get("status") not in ("complete", "max_turns"):
            raise ValueError(f"trace ended with interrupted status: {path}:{number}")
        seen_summary = kind == "summary"
    if events == 0 or summaries != 1 or len(goals) != 1:
        raise ValueError(f"trace needs one goal, replayable events, and exactly "
                         f"one summary: {path}")
    if any(open_calls.values()):
        raise ValueError(f"trace has tool calls without results: {path}")
    return goals[0]


def load_cases(manifest: Path) -> list[tuple[dict, Path]]:
    """Validate the whole corpus before scoring any case; never skip bad rows."""
    try:
        specs = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {manifest}: {exc}") from exc
    if not isinstance(specs, list) or not specs:
        raise ValueError("manifest must be a non-empty list of labelled cases")
    seen: set[str] = set()
    seen_traces: set[Path] = set()
    cases: list[tuple[dict, Path]] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = spec.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"case {index} needs a unique, non-empty id")
        seen.add(case_id)
        trace_name = spec.get("trace")
        if not isinstance(trace_name, str) or not trace_name.strip():
            raise ValueError(f"case {case_id} needs a trace path")
        label = spec.get("label")
        if (not isinstance(label, dict)
                or type(label.get("should_trip")) is not bool
                or not isinstance(label.get("note"), str)
                or not label["note"].strip()):
            raise ValueError(f"case {case_id} needs a manual boolean label and note")
        try:
            Label(**label)
        except TypeError as exc:
            raise ValueError(f"invalid label for {case_id}: {exc}") from exc
        trace = Path(trace_name)
        if not trace.is_absolute():
            trace = manifest.parent / trace
        trace = trace.resolve()
        if trace in seen_traces:
            raise ValueError(f"case {case_id} reuses a trace already scored")
        seen_traces.add(trace)
        trace_goal = _validate_trace(trace)
        if "goal" in spec and spec["goal"] != trace_goal:
            raise ValueError(f"case {case_id} goal differs from its trace")
        cases.append((spec, trace))
    return cases


def _rate(successes: int, total: int) -> dict[str, Any] | None:
    return wilson(successes, total).to_dict() if total else None


def score_manifest(manifest: Path) -> dict:
    rows = []
    for spec, trace in load_cases(manifest):
        scenario = scenario_from_trace(
            trace, label=Label(**spec["label"]),
            scenario_id=spec["id"], family="external")
        if not scenario.goal.strip() or not scenario.steps:
            raise ValueError(f"case {spec['id']} needs a goal and replayable events")
        result = run_scenario(scenario)
        rows.append({
            "id": spec["id"], "truth": scenario.label.should_trip,
            "tripped": result.tripped, "detector": result.trip_detector,
            "events": len(scenario.steps),
        })

    tp = sum(row["truth"] and row["tripped"] for row in rows)
    fp = sum(not row["truth"] and row["tripped"] for row in rows)
    fn = sum(row["truth"] and not row["tripped"] for row in rows)
    tn = sum(not row["truth"] and not row["tripped"] for row in rows)
    healthy = [row["events"] for row in rows if not row["truth"]]
    failing = [row["events"] for row in rows if row["truth"]]
    return {
        "cases": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "false_positive_rate": _rate(fp, fp + tn),
        "median_events_healthy": statistics.median(healthy) if healthy else None,
        "median_events_failing": statistics.median(failing) if failing else None,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest of manually labelled traces")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        report = score_manifest(args.manifest)
    except ValueError as exc:
        print(f"external evaluation failed: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"External traces: {report['cases']}  TP={report['tp']} "
              f"FP={report['fp']} FN={report['fn']} TN={report['tn']}")
        for name in ("precision", "recall", "false_positive_rate"):
            rate = report[name]
            if rate is None:
                print(f"{name}: n/a (no eligible cases)")
            else:
                print(f"{name}: {rate['point']:.1%} "
                      f"95% CI [{rate['ci_low']:.1%}, {rate['ci_high']:.1%}] "
                      f"n={rate['n']}")
        print("This scores detection on the supplied traces, not recovery success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
