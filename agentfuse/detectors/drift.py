"""Goal-drift detector.

Over hundreds of steps an agent's *interpreted* objective can quietly slide away
from the objective it was given. Each step looks locally fine; the drift is only
visible against the original goal.

Why this needed rebuilding
--------------------------
The first version compared the goal against a *single* recent message using a
lexical score. Measured on the eval suite, that signal barely separated the
classes it had to distinguish:

    abrupt off-topic      0.124   should trip
    gradual drift         0.276   should trip
    legitimate paraphrase 0.332   should NOT trip
    on-task               0.323   should NOT trip

A usable threshold has to sit inside a ~0.05 window, which is not an operating
point — it is a coin toss with extra steps. Two changes address it:

1. **Semantic comparison when available.** Embeddings put a paraphrase close to
   the goal (that is the entire point of an embedding) where a bag-of-words score
   cannot. The goal vector is computed once and cached; probe vectors are cached
   by text, so repeated phrasings cost nothing.

2. **Trajectory-aware scoring.** Judging one message at a time is noisy: a single
   aside reads as drift, and a gradual slide hides because no individual turn
   looks bad. The detector now tracks an exponential moving average of recent
   similarity, so it responds to the *direction of travel* rather than to
   whichever sentence arrived last.

The lexical path remains as a fallback so the detector — and the whole library —
still works with no API key. It is genuinely weaker, and says so: in lexical mode
the threshold is held lower to protect the false-positive rate, trading recall
for trust.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from typing import Callable, Optional

from ..embedding import get_embedder
from ..events import AgentEvent, EventType
from .base import Detector, Trip, Severity

_WORD = re.compile(r"[a-z0-9]+")

# Embeddings separate the classes well enough to sit near the middle; the
# lexical fallback does not, so it runs tighter to keep false positives down.
# Swept against the local bge-base model on the generated suite:
# 0.60 -> drift recall 83.3%, 0.65 -> 96.7% at FPR 6.1% (best F1 92.0%),
# 0.69 -> 100% drift recall but FPR jumps to 11.5%. 0.65 is the knee.
DEFAULT_THRESHOLD_EMBEDDING = 0.65
DEFAULT_THRESHOLD_LEXICAL = 0.20

# How much weight the trend puts on the newest turn. Smoothing is only worth
# paying for when the underlying signal is good enough to be worth trusting:
#
#   embeddings — separate the classes well, so a single off-topic aside really is
#     noise and suppressing it is valuable (see test_single_aside_does_not_trip).
#   lexical    — measured across the suite, smoothing bought NO reduction in
#     false positives (flat at 5.7% for every alpha from 1.0 down to 0.5) while
#     costing 11 points of drift recall. The fallback is already conservative
#     enough at threshold 0.20 that delaying it only loses detections.
DEFAULT_EMA_EMBEDDING = 0.5
DEFAULT_EMA_LEXICAL = 0.85


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _lexical_similarity(a: str, b: str) -> float:
    """Dependency-free similarity: token overlap blended with sequence ratio."""
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.5 * jaccard + 0.5 * ratio


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class DriftDetector(Detector):
    name = "drift"

    def __init__(self, original_goal: str, threshold: Optional[float] = None,
                 patience: int = 2, ema_alpha: Optional[float] = None,
                 embedder: Optional[Callable[[str], list[float]]] = None,
                 cache_size: int = 512):
        self.original_goal = original_goal
        self.patience = patience          # consecutive low-similarity turns to trip
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size

        # An injected embedder makes this testable without network access, which
        # is how the trajectory logic is verified in CI.
        self._backend_mode = "lexical"
        self._embedder = embedder or self._make_embedder()
        self.mode = ("embedding" if embedder else self._backend_mode)             if self._embedder else "lexical"

        if threshold is None:
            threshold = (DEFAULT_THRESHOLD_EMBEDDING if self._embedder
                         else DEFAULT_THRESHOLD_LEXICAL)
        self.threshold = threshold
        if ema_alpha is None:
            ema_alpha = (DEFAULT_EMA_EMBEDDING if self._embedder
                         else DEFAULT_EMA_LEXICAL)
        self.ema_alpha = ema_alpha        # weight on the newest observation

        self._goal_vec: Optional[list[float]] = None
        self._ema: Optional[float] = None
        self._low_streak = 0
        self._last_similarity: Optional[float] = None

    # ------------------------------------------------------------------
    def _make_embedder(self) -> Optional[Callable[[str], list[float]]]:
        """Resolve an embedder: local ONNX first, hosted second, else lexical.

        Local is preferred because it is free, needs no key, keeps the agent's
        reasoning off a third party's servers, and costs ~4ms on CPU rather than
        a network round trip on the supervision hot path.
        """
        embedder, mode = get_embedder()
        self._backend_mode = mode
        return embedder

    def _vector(self, text: str) -> Optional[list[float]]:
        """Embed with a small cache — the goal never changes and phrasings repeat."""
        if self._embedder is None:
            return None
        if text in self._cache:
            return self._cache[text]
        try:
            vec = self._embedder(text)
        except Exception:
            # Never let a network blip take down the supervised run: fall back
            # to lexical scoring for the rest of this run.
            self._embedder = None
            self.mode = "lexical (degraded)"
            self.threshold = min(self.threshold, DEFAULT_THRESHOLD_LEXICAL)
            return None
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[text] = vec
        return vec

    def _similarity(self, probe: str) -> float:
        if self._embedder is not None:
            if self._goal_vec is None:
                self._goal_vec = self._vector(self.original_goal)
            probe_vec = self._vector(probe)
            if self._goal_vec is not None and probe_vec is not None:
                return _cosine(self._goal_vec, probe_vec)
        return _lexical_similarity(self.original_goal, probe)

    # ------------------------------------------------------------------
    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        probe = event.goal or (event.text if event.type == EventType.LLM_CALL else None)
        if not probe:
            return None

        sim = self._similarity(probe)
        self._last_similarity = sim

        # Track the direction of travel, not just the latest sentence. A single
        # aside should not trip; a sustained slide should.
        self._ema = sim if self._ema is None else (
            self.ema_alpha * sim + (1 - self.ema_alpha) * self._ema)

        if self._ema < self.threshold:
            self._low_streak += 1
        else:
            self._low_streak = 0

        if self._low_streak >= self.patience:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"Agent's working goal has drifted from the original objective "
                    f"(trend {self._ema:.2f} < {self.threshold:.2f} for "
                    f"{self._low_streak} consecutive turns; latest turn {sim:.2f})."
                ),
                evidence={
                    "similarity": round(sim, 3),
                    "trend": round(self._ema, 3),
                    "threshold": self.threshold,
                    "mode": self.mode,
                    "original_goal": self.original_goal[:200],
                    "current_goal": probe[:200],
                },
            )
        return None

    def reset(self) -> None:
        self._low_streak = 0
        self._ema = None
