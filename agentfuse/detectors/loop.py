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

Two measured corrections, added after leave-one-out ablation showed this detector
was **net-harmful** — removing it improved system F1 by 1.8 points, because it
produced every remaining false positive while the progress detector caught the
same loops as a backstop. Both causes turned out to be specific and fixable:

  1. **An identical ERROR is not an identical answer.** ``HTTP 503`` three times
     is the signature of a healthy retry against a flaky endpoint, and the
     detector was halting those runs on the third attempt — one step before the
     success that was about to arrive. An error result is an explicit invitation
     to retry, so error-shaped results get ``retry_multiplier`` times the
     patience. An agent still retrying a permanently-broken endpoint six times
     over is caught; one that succeeds on the fourth attempt is not.

  2. **The rest was the benchmark, not the detector.** The other 14 false
     positives came from ``gen_long_sparse_benign``, which drew its "varied
     work" from a pool of 4 tools and 4 argument dicts and so emitted the *same
     call with the same result* up to 11 times running, with no state change,
     labelled healthy. That is a loop with a benign label. The generator was
     fixed; no detector change was warranted or made.

     A first-cycle grace period was tried here first, mirroring the bounded one
     in ``NoProgressDetector``. Once the generator was corrected it measured
     **exactly zero effect** — identical recall, F1, FPR and attribution with it
     on and off — while costing slower detection on runs that loop from their
     very first call. It was removed rather than shipped. Recorded because
     "harmless and inert" is still a cost, and because the fix that looked
     obvious was aimed at the wrong component.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

from ..events import AgentEvent, EventType, stable_hash
from .base import Detector, Trip, Severity

#: Result text that reads as a transient failure rather than an answer. Kept
#: deliberately narrow: matching too eagerly would hand extra patience to loops
#: whose output merely *mentions* an error, and the cost of a miss here is a
#: later trip, while the cost of over-matching is a missed loop entirely.
_ERROR_SHAPED = re.compile(
    r"\b(error|exception|traceback|failed|failure|timed?[ -]?out|timeout|"
    r"unavailable|refused|denied|unauthori[sz]ed|forbidden|throttled|"
    r"rate[ -]?limit(ed)?|too many requests|try again|retry|"
    r"http[ /]?[45]\d\d|\b[45]\d\d\s+(service|internal|bad|gateway|not))\b",
    re.IGNORECASE)


def looks_like_error(text: Optional[str]) -> bool:
    """Is this result a transient failure the agent is entitled to retry?"""
    return bool(_ERROR_SHAPED.search(text or ""))


def _lane(event: AgentEvent) -> str:
    """Which in-flight call this event belongs to.

    An explicit call id from the adapter is authoritative. Failing that,
    ``(node, tool)`` separates parallel calls to *different* tools and different
    agents, which covers the cases that actually occur. Two concurrent calls to
    the SAME tool on the same node remain genuinely ambiguous without an id from
    the runtime, and the later call wins — stated rather than hidden, because it
    is a limit of the event stream, not something a detector can resolve.
    """
    meta = event.meta or {}
    call_id = meta.get("call_id") or meta.get("tool_call_id")
    if call_id:
        return str(call_id)
    return f"{event.node or ''}:{event.tool_name or ''}"


class LoopDetector(Detector):
    name = "loop"

    def __init__(self, threshold: int = 3, window: int = 12,
                 blind_multiplier: int = 2, retry_multiplier: int = 2,
                 calibrator=None):
        # Optional per-run calibration. Widens the threshold for workloads that
        # legitimately repeat calls (polling, retrying) — never tightens it.
        self.calibrator = calibrator
        self.threshold = threshold          # identical (call, result) repeats before tripping
        self.window = window                # how many recent calls we remember
        self.blind_multiplier = blind_multiplier  # slack for the result-varying fallback
        self.retry_multiplier = retry_multiplier  # slack when the result is an error
        self._pairs: deque[str] = deque(maxlen=window)      # (call + result) fingerprints
        self._signatures: deque[str] = deque(maxlen=window)  # call-only fingerprints
        # Calls awaiting their results, keyed by lane rather than held in a
        # single slot. A single slot assumes calls and results strictly
        # alternate, which is false the moment an agent issues parallel tool
        # calls: call(a), call(b), result(a) paired A's OUTCOME with B's
        # SIGNATURE. Measured consequence was a trip that named a tool the agent
        # never called — and a steer built from that evidence tells the agent to
        # stop using something it never used.
        self._pending: dict[str, str] = {}
        self._pending_meta: dict[str, dict] = {}
        self._last_progress_step = 0

    # ------------------------------------------------------------------
    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        # Any real state advance clears the slate — the agent is making headway.
        #
        # Keyed on the *presence of state*, not on the event type. Adapters
        # signal progress differently: the AgentKit hooks attach state to the
        # TOOL_RESULT they already emit, while other callers send a standalone
        # STATE_UPDATE. Matching only on the event type meant that in production
        # this detector never reset on genuine progress — a real bug the
        # benchmark could not see, because the benchmark emitted the event the
        # detector happened to want.
        if event.state is not None:
            self._last_progress_step = event.step
            self._pairs.clear()
            self._signatures.clear()
            self._pending.clear()
            self._pending_meta.clear()
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

        lane = _lane(event)
        self._pending[lane] = sig
        self._pending_meta[lane] = {"tool": event.tool_name, "args": event.tool_args,
                                    "step": event.step}
        self._signatures.append(sig)
        # No verdict here. The fallback is counted on the call and *judged* on
        # the result, so this detector never halts an action whose outcome it
        # has not seen. See the module docstring.
        return None

    def _on_result(self, event: AgentEvent) -> Optional[Trip]:
        """Pair the result with the call that produced it, then count repeats."""
        lane = _lane(event)
        sig = self._pending.pop(lane, None)
        if sig is None:
            return None
        meta = self._pending_meta.pop(lane, {})

        pair = f"{sig}->{stable_hash(_normalise(event.text))}"
        self._pairs.append(pair)

        # Fallback, judged here rather than on the call: the same call repeated
        # well past the threshold with no progress is a loop even if its output
        # keeps changing slightly. Checked first — it is the stronger evidence.
        # list() so a concurrent append cannot raise "deque mutated during
        # iteration" — measured, not hypothetical: it crashed the supervisor.
        blind_limit = self._effective_threshold() * self.blind_multiplier
        sig_repeats = sum(1 for s in list(self._signatures) if s == sig)
        if sig_repeats >= blind_limit:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"Tool '{meta.get('tool')}' called with identical arguments "
                    f"{sig_repeats}x with no state progress since step "
                    f"{self._last_progress_step}, despite varying results."
                ),
                evidence={
                    "tool": meta.get("tool"),
                    "args": meta.get("args"),
                    "repeats": sig_repeats,
                    "signature": sig,
                    "signal": "call-only fallback",
                    "steps_without_progress": event.step - self._last_progress_step,
                },
            )

        is_error = looks_like_error(event.text)
        effective = self._effective_threshold(is_error=is_error)
        repeats = sum(1 for p in list(self._pairs) if p == pair)
        if repeats >= effective:
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
                    "error_shaped_result": is_error,
                    "steps_without_progress": event.step - self._last_progress_step,
                },
            )
        return None

    # ------------------------------------------------------------------
    def _effective_threshold(self, is_error: bool = False) -> float:
        """The bar this run has to clear, after calibration, retries and grace."""
        base = (self.calibrator.loop_threshold(self.threshold)
                if self.calibrator else self.threshold)
        if is_error:
            # An identical error is the signature of a legitimate retry, not of
            # an agent re-asking a question it already has the answer to.
            # Measured: without this, FP 6 -> 22 and FPR 1.25% -> 4.58%, with
            # recall and attribution unchanged. It buys precision, nothing else.
            base *= self.retry_multiplier
        return base

    def reset(self) -> None:
        self._pairs.clear()
        self._signatures.clear()
        self._pending.clear()
        self._pending_meta.clear()


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
