"""Infinite / repetitive tool-loop detector.

The classic long-horizon failure: an agent calls the same tool with the same
arguments over and over, each time hoping for a different answer. We fingerprint
every tool call as ``(tool_name, hash(args))`` and count repeats inside a sliding
window. Crucially, a repeat only *counts against* the agent if the working state
has NOT advanced in between — genuine progress resets suspicion.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..events import AgentEvent, EventType
from .base import Detector, Trip, Severity


class LoopDetector(Detector):
    name = "loop"

    def __init__(self, threshold: int = 3, window: int = 12):
        self.threshold = threshold      # repeats of the same signature before tripping
        self.window = window            # how many recent tool calls we remember
        self._signatures: deque[str] = deque(maxlen=window)
        self._last_progress_step = 0

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        # Any real state advance clears the slate — the agent is making headway.
        if event.type == EventType.STATE_UPDATE and event.state is not None:
            self._last_progress_step = event.step
            self._signatures.clear()
            return None

        if event.type != EventType.TOOL_CALL:
            return None

        sig = event.tool_signature
        if sig is None:
            return None

        self._signatures.append(sig)
        repeats = sum(1 for s in self._signatures if s == sig)

        if repeats >= self.threshold:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"Tool '{event.tool_name}' called with identical arguments "
                    f"{repeats}x within the last {len(self._signatures)} calls, "
                    f"with no state progress since step {self._last_progress_step}."
                ),
                evidence={
                    "tool": event.tool_name,
                    "args": event.tool_args,
                    "repeats": repeats,
                    "signature": sig,
                    "steps_without_progress": event.step - self._last_progress_step,
                },
            )
        return None

    def reset(self) -> None:
        self._signatures.clear()
