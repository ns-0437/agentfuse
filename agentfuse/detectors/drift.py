"""Goal-drift detector.

Over hundreds of steps an agent's *interpreted* objective can quietly slide away
from the objective it was given. Each step looks locally fine; the drift is only
visible against the original system prompt. We measure the semantic distance
between the original goal and the agent's recent reasoning/goal text.

Two backends:
  * If ``OPENAI_API_KEY`` is set and the OpenAI SDK is installed, we use real
    embeddings (cosine distance).
  * Otherwise we fall back to a dependency-free lexical similarity (difflib +
    token-Jaccard) so the detector — and the whole demo — still runs offline.

The lexical fallback is deliberately conservative; it exists so the mechanism is
demonstrable without a key, not to claim embedding-grade accuracy.
"""

from __future__ import annotations

import difflib
import os
import re
from typing import Optional

from ..env import load_env
from ..events import AgentEvent, EventType
from .base import Detector, Trip, Severity

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _lexical_similarity(a: str, b: str) -> float:
    """0..1 similarity with zero dependencies."""
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.5 * jaccard + 0.5 * ratio


class DriftDetector(Detector):
    name = "drift"

    def __init__(self, original_goal: str, threshold: float = 0.20, patience: int = 2):
        self.original_goal = original_goal
        self.threshold = threshold      # trip when similarity drops BELOW this
        self.patience = patience        # consecutive low-similarity turns required
        self._low_streak = 0
        self._embedder = self._make_embedder()

    def _make_embedder(self):
        load_env()  # pick up a key from .env if the shell has none
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI()

            def embed(text: str):
                r = client.embeddings.create(
                    model=os.getenv("AGENTFUSE_EMBED_MODEL", "text-embedding-3-small"),
                    input=text[:8000],
                )
                return r.data[0].embedding

            return embed
        except Exception:
            return None

    def _similarity(self, a: str, b: str) -> float:
        if self._embedder is None:
            return _lexical_similarity(a, b)
        try:
            va, vb = self._embedder(a), self._embedder(b)
            dot = sum(x * y for x, y in zip(va, vb))
            na = sum(x * x for x in va) ** 0.5
            nb = sum(y * y for y in vb) ** 0.5
            return dot / (na * nb) if na and nb else 0.0
        except Exception:
            return _lexical_similarity(a, b)

    def inspect(self, event: AgentEvent, history: list[AgentEvent]) -> Optional[Trip]:
        probe = event.goal or (event.text if event.type == EventType.LLM_CALL else None)
        if not probe:
            return None

        sim = self._similarity(self.original_goal, probe)
        if sim < self.threshold:
            self._low_streak += 1
        else:
            self._low_streak = 0

        if self._low_streak >= self.patience:
            return Trip(
                detector=self.name,
                severity=Severity.TRIP,
                reason=(
                    f"Agent's working goal has drifted from the original objective "
                    f"(similarity {sim:.2f} < {self.threshold:.2f} for "
                    f"{self._low_streak} consecutive turns)."
                ),
                evidence={
                    "similarity": round(sim, 3),
                    "backend": "embeddings" if self._embedder else "lexical",
                    "original_goal": self.original_goal[:200],
                    "current_goal": probe[:200],
                },
            )
        return None

    def reset(self) -> None:
        self._low_streak = 0
