"""Print the drift detector's internal state, turn by turn, for one trace.

Every drift investigation in this project has needed the same thing: not
"did it trip" but *why not* — what the EMA actually was on each turn, whether
the action was anchor-grounded or tool-grounded, and which suppression ate
the trip. The scored suites answer the first question and hide the rest, so
this was rewritten from scratch as a throwaway script three separate times
(sections 3.12, 3.17, 3.18) before being kept.

It reads the SAME `DriftDetector` the suites use, at production defaults, and
prints one row per event. It never asserts anything — it is an instrument,
not a test.

    python evals/measure_drift_trace.py suite_r_cascade_market
    python evals/measure_drift_trace.py gen_benign_narrated_failure --seed 1

Columns:
    ema         the smoothed trajectory; compare against `threshold`
    low         consecutive turns under threshold (trips at `patience`)
    anchored    this action named one of the goal's own entities
    tool_ok     this action used a tool blessed while the trend was healthy
    grounded    the combined verdict — if True on a low-EMA turn, the trip
                was SUPPRESSED, and that is usually the thing you are hunting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.detectors.drift import DriftDetector  # noqa: E402
from agentfuse.events import AgentEvent, EventType  # noqa: E402
from evals.schema import Label  # noqa: E402
from evals.trace_import import scenario_from_trace  # noqa: E402

SUITE = ROOT / "evals" / "captured" / "suite"


def _load_captured(scenario_id: str):
    """A real captured run, by its id in the suite's labels.json."""
    specs = json.loads((SUITE / "labels.json").read_text(encoding="utf-8"))
    match = next((s for s in specs if s["id"] == scenario_id), None)
    if match is None:
        return None
    return scenario_from_trace(
        SUITE.parent / match["trace"], label=Label(**match["label"]),
        goal=match.get("goal"), scenario_id=match["id"],
        family=match.get("world", "captured"))


def _load_generated(name: str, seed: int):
    """A generated scenario, by generator function name (`gen_*`)."""
    import random
    from evals import generators, generators_extra
    for mod in (generators_extra, generators):
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn(random.Random(seed), seed)
    return None


def _anchors_match(det: DriftDetector, event: AgentEvent) -> bool:
    """Does this action name one of the goal's entities? Mirrors _note_action.

    Deliberately a copy rather than a refactor of the detector: an instrument
    that shares code with the thing it measures stops being able to show that
    thing misbehaving.
    """
    from agentfuse.detectors.drift import _ANCHOR_WORD
    parts = [event.tool_name or ""]
    for key, val in (event.tool_args or {}).items():
        parts.append(f"{key} {val}")
    text = " ".join(parts).lower()
    tokens = {t.strip("./-") for t in _ANCHOR_WORD.findall(text)}
    return bool((det._anchors & tokens) or any(a in text for a in det._anchors))


def trace(scenario) -> None:
    det = DriftDetector(original_goal=scenario.goal)
    print(f"goal      : {scenario.goal}")
    print(f"mode      : {det.mode}   threshold: {det.threshold}   "
          f"patience: {det.patience}")
    print(f"anchors   : {sorted(det._anchors)}")
    print()
    header = (f"{'#':>3} {'event':<10} {'ema':>7} {'low':>4} {'anch':>5} "
              f"{'tool':>5} {'grnd':>5}  what")
    print(header)
    print("-" * len(header))

    for i, step in enumerate(scenario.steps):
        if step.kind == "tool":
            ev = AgentEvent(type=EventType.TOOL_CALL, step=i,
                            tool_name=step.tool_name, tool_args=step.tool_args,
                            goal=step.goal)
            what = f"{step.tool_name} {json.dumps(step.tool_args or {})[:44]}"
        else:
            ev = AgentEvent(type=EventType.LLM_CALL, step=i, text=step.text,
                            goal=step.goal)
            what = (step.text or "")[:52]

        trip = det.inspect(ev, [])
        ema = "-" if det._ema is None else f"{det._ema:.3f}"
        # The detector keeps only the OR of its two grounding signals, but the
        # whole point of this instrument is telling them apart -- which one is
        # suppressing the trip decides which fix could possibly work. So the
        # anchor check is recomputed here exactly as _note_action does it.
        anchored = tool_ok = ""
        if ev.type == EventType.TOOL_CALL:
            tool_ok = "yes" if (step.tool_name in det._on_goal_tools) else "no"
            anchored = "yes" if _anchors_match(det, ev) else "no"
        print(f"{i:>3} {ev.type.value:<10} {ema:>7} {det._low_streak:>4} "
              f"{anchored:>5} {tool_ok:>5} "
              f"{str(det._last_action_grounded):>5}  {what}"
              + ("   <== TRIP" if trip else ""))

    print()
    print(f"suppressed trips: {det.suppressed_trips}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="a suite id (suite_*) or a generator name (gen_*)")
    ap.add_argument("--seed", type=int, default=1,
                    help="seed for generated scenarios (default 1)")
    args = ap.parse_args()

    scenario = (_load_generated(args.scenario, args.seed)
                if args.scenario.startswith("gen_")
                else _load_captured(args.scenario))
    if scenario is None:
        print(f"no such scenario: {args.scenario}")
        print("\ncaptured ids:")
        specs = json.loads((SUITE / "labels.json").read_text(encoding="utf-8"))
        for s in specs:
            print(f"  {s['id']}")
        return 1

    trace(scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
