"""Turn real captured runs into eval scenarios.

The generators fix sampling error, not authoring bias: they encode *my* model of
what agent failure looks like. The only cure for that is trajectories nobody
designed to be caught — real runs, captured in the field, labelled after the fact.

AgentFuse already writes a JSONL trace for every supervised run, so the pipeline
is short:

    1. Run agents with a tracer:  ``MonitorConfig(jsonl_path="runs/my_run.jsonl")``
    2. Import:                    ``python evals/trace_import.py runs/my_run.jsonl``
    3. Label it (did it genuinely fail? which mode? from which step?)
    4. Commit the labelled case into ``evals/captured/``

Labelling is a human judgement and is deliberately *not* automated: asking the
system under test to label its own benchmark would reintroduce exactly the
circularity this file exists to remove.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # works both as `python evals/trace_import.py` and as a package import
    from .schema import Scenario, Label, StepSpec
except ImportError:  # pragma: no cover - direct script execution
    from evals.schema import Scenario, Label, StepSpec  # noqa: E402

CAPTURED_DIR = Path(__file__).resolve().parent / "captured"


def scenario_from_trace(path: Path, label: Label, goal: Optional[str] = None,
                        scenario_id: Optional[str] = None,
                        family: str = "captured") -> Scenario:
    """Rebuild a Scenario from a JSONL trace produced by ``Tracer``."""
    steps: list[StepSpec] = []
    meta: dict = {}
    pending: dict[int, StepSpec] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        if rec.get("kind") == "meta":
            meta = rec
            continue
        if rec.get("kind") != "event":
            continue

        t, step_no = rec.get("type"), rec.get("step", 0)
        if t == "tool_call":
            s = StepSpec(kind="tool", tool_name=rec.get("tool_name"),
                         tool_args=rec.get("tool_args") or {},
                         tokens_in=rec.get("tokens_in", 0),
                         tokens_out=rec.get("tokens_out", 0),
                         goal=rec.get("goal"), node=rec.get("node", "agent"))
            pending[step_no] = s
            steps.append(s)
        elif t == "tool_result":
            s = pending.get(step_no)
            if s is not None:
                s.result = rec.get("text")
        elif t == "llm_call":
            steps.append(StepSpec(kind="think", text=rec.get("text"),
                                  goal=rec.get("goal"),
                                  tokens_in=rec.get("tokens_in", 0),
                                  tokens_out=rec.get("tokens_out", 0),
                                  node=rec.get("node", "agent")))
        elif t == "state_update":
            if steps:
                steps[-1].progress = True

    return Scenario(
        id=scenario_id or f"captured_{path.stem}",
        title=f"Captured run: {path.stem}",
        family=family,
        goal=goal or meta.get("original_goal", ""),
        steps=steps,
        description=f"Imported from {path.name} ({len(steps)} steps).",
        label=label,
    )


def load_captured() -> list[Scenario]:
    """Load every labelled capture in ``evals/captured/``.

    Each capture is a JSON file: ``{"trace": "...jsonl", "goal": "...",
    "label": {"should_trip": true, "detector": "loop", "onset_index": 3}}``
    """
    if not CAPTURED_DIR.exists():
        return []
    out: list[Scenario] = []
    for spec_path in sorted(CAPTURED_DIR.glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        trace = Path(spec["trace"])
        if not trace.is_absolute():
            trace = CAPTURED_DIR / trace
        if not trace.exists():
            continue
        out.append(scenario_from_trace(
            trace,
            label=Label(**spec["label"]),
            goal=spec.get("goal"),
            scenario_id=spec.get("id") or spec_path.stem,
        ))
    return out


def _main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python evals/trace_import.py <trace.jsonl> [--id NAME]")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no such trace: {path}")
        return 1

    sc = scenario_from_trace(path, label=Label(should_trip=False))
    print(f"\nImported {path.name}")
    print(f"  goal   : {sc.goal[:90]}")
    print(f"  steps  : {len(sc.steps)}   tokens: {sc.total_tokens:,}")
    for i, s in enumerate(sc.steps[:12]):
        what = s.tool_name or (s.text or "")[:52]
        print(f"    [{i:>2}] {s.kind:<6} {what}")
    if len(sc.steps) > 12:
        print(f"    ... {len(sc.steps) - 12} more")

    stem = path.stem
    print("\nTo add this as a labelled eval case, write "
          f"evals/captured/{stem}.json:\n")
    print(json.dumps({
        "id": f"captured_{stem}",
        "trace": f"../../runs/{path.name}",
        "goal": sc.goal,
        "label": {"should_trip": True, "detector": "loop",
                  "onset_index": 0, "detect_by_index": 5,
                  "note": "why you judged this a failure (or not)"},
    }, indent=2))
    print("\nLabel it yourself — automating this would make the benchmark circular.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
