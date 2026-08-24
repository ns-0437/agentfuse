"""Observability layer.

Everything the monitor sees and decides is emitted here: a live, human-readable
trace to the console plus a machine-readable JSONL log you can pipe into any
observability backend. This is the "graph routes, state changes, token spend"
telemetry the brief asks for.

Uses ``rich`` for a colored live view when available, and degrades to plain
``print`` when it isn't — so the demo looks great but never *requires* a dep.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from .redact import redact_obj
from .events import AgentEvent, EventType
from .detectors.base import Trip
from .recovery import SteeringPath


def _enable_unicode() -> bool:
    """Best-effort: put stdout in UTF-8 so emoji render; report if it worked.

    Windows consoles default to cp1252 and raise on emoji. We try to reconfigure
    to UTF-8; if that isn't possible we signal callers to use ASCII markers.
    """
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in enc:
        return True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


_UNICODE = _enable_unicode()

try:  # optional pretty output
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    _RICH = True
    # force_terminal keeps colors; safe_box avoids glyphs the console can't draw
    _console = Console(safe_box=True)
except Exception:  # pragma: no cover
    _RICH = False
    _console = None


_ICONS_UNICODE = {
    EventType.TOOL_CALL: "🔧",
    EventType.TOOL_RESULT: "📤",
    EventType.LLM_CALL: "🧠",
    EventType.STATE_UPDATE: "📝",
    EventType.ROUTE: "➡️ ",
    EventType.TRIP: "⚡",
    EventType.RECOVERY: "🧭",
    EventType.RESUME: "▶️ ",
    EventType.ABORT: "🛑",
    EventType.COMPLETE: "✅",
}

_ICONS_ASCII = {
    EventType.TOOL_CALL: "[tool]",
    EventType.TOOL_RESULT: "[<-  ]",
    EventType.LLM_CALL: "[llm ]",
    EventType.STATE_UPDATE: "[stat]",
    EventType.ROUTE: "[rout]",
    EventType.TRIP: "[TRIP]",
    EventType.RECOVERY: "[heal]",
    EventType.RESUME: "[ >> ]",
    EventType.ABORT: "[STOP]",
    EventType.COMPLETE: "[ OK ]",
}

_ICONS = _ICONS_UNICODE if _UNICODE else _ICONS_ASCII
_TRIP_MARK = "⚡" if _UNICODE else "!!"
_HEAL_MARK = "🧭" if _UNICODE else ">>"


class Tracer:
    def __init__(self, jsonl_path: Optional[str] = None, echo: bool = True,
                 append: bool = False):
        self.echo = echo
        self.jsonl_path = jsonl_path
        self._fh = None
        if jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
            # `append=True` is how a checkpoint-resumed monitor keeps its trace
            # consistent with its own restored state. Without it, every fresh
            # `Tracer` truncates the file on construction -- correct for the
            # common case (a new run, or a demo intentionally starting clean),
            # wrong for a restart: the monitor's tokens/ladder/escalation
            # status all correctly carry across a restart (checkpoint.py), but
            # the one human-readable record of how they got there would be
            # silently erased the instant the resumed process constructs its
            # own Tracer -- before `Monitor.restore()` even runs.
            self._fh = open(jsonl_path, "a" if append else "w", encoding="utf-8")
        self.trips = 0
        self.recoveries = 0

    def _write(self, record: dict) -> None:
        if self._fh:
            # Credentials are stripped on the way to disk, not on the way out of
            # it. The trace is durable and often shipped to a log backend, so a
            # secret written here is a secret written permanently and in a place
            # nobody re-reads. `redact_obj` walks the whole record because tool
            # ARGUMENTS carry secrets at least as often as tool results.
            self._fh.write(json.dumps(redact_obj(record), default=str) + "\n")
            self._fh.flush()

    def meta(self, record: dict) -> None:
        """Write a run-level metadata record (objective, config) as the first line."""
        self._write({"kind": "meta", **record})

    def event(self, event: AgentEvent) -> None:
        self._write({"kind": "event", **event.to_dict()})
        if not self.echo:
            return
        icon = _ICONS.get(event.type, "•")
        detail = ""
        if event.tool_name:
            detail = f"{event.tool_name}({json.dumps(event.tool_args or {}, default=str)[:80]})"
        elif event.text:
            detail = str(event.text)[:90]
        line = f"  {icon} step {event.step:>3} [{event.node}] {event.type.value:<12} {detail}"
        if _RICH:
            _console.print(line, style="dim")
        else:
            print(line)

    def trip(self, event: AgentEvent, trip: Trip) -> None:
        self.trips += 1
        _echo = self.echo
        self._write({"kind": "trip", "step": event.step, "detector": trip.detector,
                     "severity": trip.severity.value, "reason": trip.reason,
                     "evidence": trip.evidence})
        if not _echo:
            return
        title = f"{_TRIP_MARK} CIRCUIT BREAKER TRIPPED - {trip.detector.upper()} ({trip.severity.value})"
        body = f"{trip.reason}"
        if _RICH:
            _console.print(Panel(body, title=title, border_style="yellow", expand=False))
        else:
            print(f"\n{'='*70}\n{title}\n{trip.reason}\n{'='*70}")

    def recovery(self, path: SteeringPath) -> None:
        self.recoveries += 1
        _echo = self.echo
        self._write({"kind": "recovery", "action": path.action.value,
                     "instruction": path.instruction, "rationale": path.rationale,
                     "confidence": path.confidence, "backend": path.backend})
        if not _echo:
            return
        title = f"{_HEAL_MARK} STEERING RECOVERY - action={path.action.value} (conf {path.confidence:.2f}, via {path.backend})"
        body = f"[rationale] {path.rationale}\n\n[injected instruction]\n{path.instruction}"
        if _RICH:
            _console.print(Panel(body, title=title, border_style="cyan", expand=False))
        else:
            print(f"\n{'-'*70}\n{title}\n{body}\n{'-'*70}")

    def summary(self, totals: dict) -> None:
        self._write({"kind": "summary", **totals})
        if not self.echo:
            return
        if _RICH:
            t = Table(title="AgentFuse run summary", show_header=False, border_style="green")
            for k, v in totals.items():
                t.add_row(str(k), str(v))
            _console.print(t)
        else:
            print("\n=== AgentFuse run summary ===")
            for k, v in totals.items():
                print(f"  {k}: {v}")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
