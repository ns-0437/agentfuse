"""Read a local AgentFuse trace without loading optional SDKs or raw payloads."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def inspect_trace(path: str) -> dict:
    """Summarize one JSONL trace; never include tool arguments or event text.

    A summary followed by more events is stale (for example, a resumed run
    crashed), so only a summary at the end proves a final status.
    """
    trace = Path(path)
    detectors: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    records = events = trips = recoveries = steps = 0
    final_summary: dict | None = None
    try:
        with trace.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("kind"), str):
                    raise ValueError(f"invalid trace record on line {line_number}")
                records += 1
                kind = record["kind"]
                if kind == "summary":
                    final_summary = record
                    continue
                # Even a new metadata record means an appended run began.
                final_summary = None
                if kind == "event":
                    events += 1
                    step = record.get("step")
                    if isinstance(step, int) and not isinstance(step, bool):
                        steps = max(steps, step)
                elif kind == "trip":
                    trips += 1
                    detector = record.get("detector")
                    if isinstance(detector, str):
                        detectors[detector] += 1
                elif kind == "recovery":
                    recoveries += 1
                    strategy = record.get("strategy")
                    if isinstance(strategy, str):
                        strategies[strategy] += 1
    except OSError as exc:
        raise ValueError(f"cannot read trace: {exc.strerror or type(exc).__name__}") from exc

    if records == 0:
        raise ValueError("trace is empty")
    return {
        "trace_path": str(trace.resolve()),
        "status": final_summary.get("status") if final_summary else "incomplete",
        "has_final_summary": final_summary is not None,
        "records": records,
        "events": events,
        "steps": final_summary.get("steps", steps) if final_summary else steps,
        "trips": trips,
        "recoveries": recoveries,
        "detectors": dict(sorted(detectors.items())),
        "strategies": dict(sorted(strategies.items())),
        "total_tokens": final_summary.get("total_tokens") if final_summary else None,
        "total_cost_usd": final_summary.get("total_cost_usd") if final_summary else None,
        "cost_is_complete": final_summary.get("cost_is_complete") if final_summary else None,
        "unpriced_tokens": final_summary.get("unpriced_tokens") if final_summary else None,
    }
