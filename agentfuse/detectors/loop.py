"""Infinite / repetitive tool-loop detector.

The classic long-horizon failure: an agent calls the same tool with the same
arguments over and over, each time hoping for a different answer.

The naive version of this — count identical ``(tool, args)`` calls — halts
legitimate retries. Retrying a flaky endpoint twice and succeeding on the third
attempt produces three identical calls, and a pure call-counter fires on that
third call *before its result arrives*, killing a run that was about to work.
The eval suite caught exactly that (``retry_transient_then_success``).

So the definition is tightened: a loop is the same call producing **the same
result**, repeatedly, with no state progress. Retries that eventually return
something different are not loops. Two signals:

  1. **Primary** — repeats of the ``(tool, args, result)`` triple. High
     confidence, and evaluated on the tool *result*, so the outcome is known
     before we halt anything.
  2. **Fallback** — the same ``(tool, args)`` signature repeating far more often
     (``2x`` threshold) with no progress, whatever the results say. This still
     catches loops whose output varies cosmetically — timestamps, request ids,
     percentages that never reach completion — which the primary signal alone
     would miss.

Any genuine state advance clears both counters: an agent making headway is not
looping, regardless of what it repeats.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..events import AgentEvent, EventType, stable_hash
from .base import Detector, Trip, Severity


class LoopDetector(Detector):
    name = "loop"

    def __init__(self, threshold: int = 3, window: int = 12,
                 blind_multiplier: int = 2):
        self.threshold = threshold          # identical (call, result) repeats before tripping
        self.window = window                # how many recent calls we remember
        self.blind_multiplier = blind_multiplier  # slack for the result-varying fallback
        self._pairs: deque[str] = deque(maxlen=window)      # (call + result) fingerprints
        self._signatures: deque[str] = deque(maxlen=window)  # call-only fingerprints
        self._pending: Optional[str] = None                  # call awaiting its result
        self._pending_meta: dict = {}
        self._last_progress_step = 0

    # ------------------------------------------------------------------
    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        # Any real state advance clears the slate — the agent is making headway.
        if event.type == EventType.STATE_UPDATE and event.state is not None:
            self._last_progress_step = event.step
            self._pairs.clear()
            self._signatures.clear()
            self._pending = None
            return None

        if event.type == EventType.TOOL_CALL:
            return self._on_call(event)
        if event.type == EventType.TOOL_RESULT:
            return self._on_result(event)
        return None

    # ------------------------------------------------------------------
    def _on_call(self, event: AgentEvent) -> Optional[Trip]:
        sig = loop_signature(event)
        if sig is None:
            return None

        self._pending = sig
        self._pending_meta = {"tool": event.tool_name, "args": event.tool_args,
                              "step": event.step}
        self._signatures.append(sig)

        # Fallback: the same call repeated well past the threshold with no
        # progress is a loop even if its output keeps changing slightly.
        blind_limit = self.threshold * self.blind_multiplier
        repeats = sum(1 for s in self._signatures if s == sig)
        if repeats >= blind_limit:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"Tool '{event.tool_name}' called with identical arguments "
                    f"{repeats}x with no state progress since step "
                    f"{self._last_progress_step}, despite varying results."
                ),
                evidence={
                    "tool": event.tool_name,
                    "args": event.tool_args,
                    "repeats": repeats,
                    "signature": sig,
                    "signal": "call-only fallback",
                    "steps_without_progress": event.step - self._last_progress_step,
                },
            )
        return None

    def _on_result(self, event: AgentEvent) -> Optional[Trip]:
        """Pair the result with the call that produced it, then count repeats."""
        if self._pending is None:
            return None

        pair = f"{self._pending}->{stable_hash(_normalise(event.text))}"
        self._pending = None
        self._pairs.append(pair)

        repeats = sum(1 for p in self._pairs if p == pair)
        if repeats >= self.threshold:
            meta = self._pending_meta
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"Tool '{meta.get('tool')}' returned an identical result for "
                    f"identical arguments {repeats}x within the last "
                    f"{len(self._pairs)} calls, with no state progress since step "
                    f"{self._last_progress_step}."
                ),
                evidence={
                    "tool": meta.get("tool"),
                    "args": meta.get("args"),
                    "repeats": repeats,
                    "result": (event.text or "")[:120],
                    "signal": "call+result",
                    "steps_without_progress": event.step - self._last_progress_step,
                },
            )
        return None

    def reset(self) -> None:
        self._pairs.clear()
        self._signatures.clear()
        self._pending = None


def _normalise(text: Optional[str]) -> str:
    """Collapse incidental formatting so equal results compare equal."""
    return " ".join((text or "").split()).lower()


_PUNCT = str.maketrans("", "", "\"'`,.;:!?()[]{}<>")


def _normalise_args(value):
    """Canonicalise argument values so cosmetic differences collapse.

    ``{"q": "ACME unpaid"}``, ``{"q": "acme  unpaid"}`` and ``{"q": "acme unpaid?"}``
    are the same request expressed three ways. Hashing them raw produces three
    different signatures and the loop is never seen.

    Scope, stated plainly: this normalises **surface** form only — case,
    surrounding and repeated whitespace, and trailing punctuation. Genuinely
    reworded arguments ("acme unpaid" vs "unpaid invoices for acme") still
    produce distinct signatures and remain a known gap; closing that needs
    embeddings, not string handling.
    """
    if isinstance(value, str):
        return " ".join(value.translate(_PUNCT).split()).lower()
    if isinstance(value, dict):
        return {k: _normalise_args(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise_args(v) for v in value]
    return value


def loop_signature(event: AgentEvent) -> Optional[str]:
    """Normalised ``(tool, args)`` fingerprint used for loop detection."""
    if event.tool_name is None:
        return None
    return f"{event.tool_name}:{stable_hash(_normalise_args(event.tool_args or {}))}"
