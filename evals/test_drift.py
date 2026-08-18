"""Drift detector tests, including the embedding path — without an API key.

The embedding path is the one that matters (the lexical fallback demonstrably
cannot separate a paraphrase from gradual drift), but it normally needs a live
OpenAI key, which means it would go untested in CI and untested on any machine
without billing configured.

The detector therefore accepts an injected embedder. These tests supply a
deterministic synthetic one built from topic vectors: text about the goal's
topic embeds near the goal, text about another topic embeds away from it —
exactly the property a real embedding model provides, reproduced in a way that
runs offline and never costs anything.

That validates the *logic* — caching, trajectory smoothing, threshold selection,
graceful degradation — while leaving the quality of real embeddings to be
confirmed once a key is available.

    pytest evals/test_drift.py -v
"""

from __future__ import annotations

import math
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentfuse.detectors.drift import (  # noqa: E402
    DriftDetector, DEFAULT_THRESHOLD_EMBEDDING, DEFAULT_THRESHOLD_LEXICAL,
    _lexical_similarity,
)
from agentfuse.events import AgentEvent, EventType  # noqa: E402

GOAL = "Summarize the Q3 revenue figures from the finance report into three bullet points."

# Topic axes: finance vs marketing. A real embedding model places a paraphrase
# near its source and an unrelated topic far away; this reproduces that.
_TOPICS = {
    "revenue": (1.0, 0.0), "earnings": (0.97, 0.05), "figures": (0.93, 0.0),
    "quarter": (0.9, 0.05), "financial": (0.95, 0.0), "report": (0.85, 0.1),
    "summarize": (0.8, 0.1), "condense": (0.78, 0.12), "bullet": (0.8, 0.05),
    "marketing": (0.1, 1.0), "campaign": (0.05, 0.97), "influencer": (0.0, 0.95),
    "social": (0.05, 0.9), "advertising": (0.05, 0.93), "calendar": (0.1, 0.85),
}


def synthetic_embedder(text: str) -> list[float]:
    """Deterministic stand-in with real embedding-like geometry."""
    x = y = 0.0
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        if word in _TOPICS:
            dx, dy = _TOPICS[word]
            x += dx
            y += dy
    if x == y == 0.0:
        x = y = 0.5  # neutral text sits between the axes
    norm = math.sqrt(x * x + y * y)
    return [x / norm, y / norm]


def _probe(det: DriftDetector, text: str, step: int = 1):
    return det.inspect(AgentEvent(type=EventType.LLM_CALL, step=step, text=text, goal=text), [])


def _make(**kw) -> DriftDetector:
    return DriftDetector(original_goal=GOAL, embedder=synthetic_embedder, **kw)


# ------------------------------------------------------------ mode selection
def test_embedding_mode_selected_when_embedder_supplied():
    d = _make()
    assert d.mode == "embedding"
    assert d.threshold == DEFAULT_THRESHOLD_EMBEDDING


def test_lexical_mode_uses_its_own_lower_threshold(monkeypatch):
    """One constant cannot serve both scales — that was the original bug.

    Lexical must now be requested explicitly: with fastembed installed, an
    unconfigured detector resolves to the local model, which is the desired
    default.
    """
    monkeypatch.setenv("AGENTFUSE_EMBED_BACKEND", "none")
    d = DriftDetector(original_goal=GOAL)
    assert d.mode == "lexical"
    assert d.threshold == DEFAULT_THRESHOLD_LEXICAL


def test_explicit_threshold_always_wins():
    assert _make(threshold=0.9).threshold == 0.9


# ------------------------------------------------- the case lexical gets wrong
def test_paraphrase_does_not_trip_under_embeddings():
    """The failure that motivated this rewrite: a restatement is not drift."""
    d = _make()
    for i, text in enumerate([
        "Condense the quarter earnings figures from the financial report.",
        "Summarize quarterly revenue figures into bullet points.",
        "Produce a short summary of the quarter financial revenue figures.",
    ]):
        assert _probe(d, text, i + 1) is None, f"paraphrase wrongly tripped: {text!r}"


def test_lexical_mode_cannot_separate_paraphrase_from_drift():
    """Documents *why* embeddings were needed, rather than asserting it."""
    paraphrase = "Condense the third-quarter earnings numbers into a short digest."
    drift = "Q3 revenue looks tied to the new product line's marketing push."
    gap = abs(_lexical_similarity(GOAL, paraphrase) - _lexical_similarity(GOAL, drift))
    assert gap < 0.15, (
        "if lexical similarity ever separates these cleanly, revisit the fallback")


def test_sustained_drift_trips_under_embeddings():
    d = _make()
    trip = None
    for i, text in enumerate([
        "Reviewing competitor advertising spend across social channels.",
        "Analysing influencer campaign performance and social reach.",
        "Drafting a marketing content calendar for social campaigns.",
    ]):
        trip = trip or _probe(d, text, i + 1)
    assert trip is not None, "sustained off-topic drift was not detected"
    assert trip.detector == "drift"
    assert trip.evidence["mode"] == "embedding"


# ------------------------------------------------------- trajectory behaviour
def test_single_aside_does_not_trip():
    """One tangent between on-task turns is not drift; the EMA should absorb it."""
    d = _make()
    assert _probe(d, "Summarize the quarter revenue figures.", 1) is None
    assert _probe(d, "Briefly noting the marketing campaign context.", 2) is None
    assert _probe(d, "Back to the quarterly earnings report figures.", 3) is None


def test_trend_is_reported_alongside_the_latest_turn():
    d = _make(patience=1)
    trip = None
    for i in range(3):
        trip = trip or _probe(d, "Marketing campaign social advertising calendar.", i + 1)
    assert trip is not None
    assert "trend" in trip.evidence and "similarity" in trip.evidence


def test_reset_clears_the_trend():
    d = _make(patience=1)
    for i in range(3):
        _probe(d, "Marketing campaign social advertising.", i + 1)
    d.reset()
    assert d._ema is None and d._low_streak == 0


# ------------------------------------------------------------------ caching
def test_vectors_are_cached_so_repeated_text_is_free():
    calls = {"n": 0}

    def counting(text: str) -> list[float]:
        calls["n"] += 1
        return synthetic_embedder(text)

    d = DriftDetector(original_goal=GOAL, embedder=counting)
    for i in range(5):
        _probe(d, "Summarize the quarter revenue figures.", i + 1)
    # One embed for the goal, one for the repeated probe — not one per turn.
    assert calls["n"] == 2, f"expected 2 embed calls, got {calls['n']}"


def test_embedder_failure_degrades_instead_of_crashing():
    """A network blip must never take down the run it is supervising."""
    def broken(text: str) -> list[float]:
        raise RuntimeError("network down")

    d = DriftDetector(original_goal=GOAL, embedder=broken)
    assert _probe(d, "Some reasoning about revenue figures.", 1) is None
    assert d.mode == "lexical (degraded)"
    assert d.threshold <= DEFAULT_THRESHOLD_LEXICAL


# --------------------------------------- provider-outage resilience
# A supervisor that dies when the provider does is worse than no supervisor:
# it takes down the run it was meant to protect. Verified after a live 429
# (insufficient_quota) surfaced during real-model validation.
def test_recovery_engine_survives_provider_failure():
    from agentfuse.recovery import RecoveryEngine
    from agentfuse.events import ExecutionSnapshot

    class Failing:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("Error code: 429 - insufficient_quota")

    engine = RecoveryEngine(backend="real")
    engine._client = Failing()
    snapshot = ExecutionSnapshot(
        step=3, original_goal="Rotate the production database credential.",
        current_goal=None, total_tokens=100, total_cost_usd=0.0,
        route_history=["agent"], recent_events=[], trip_reason="looping",
        trip_detector="loop", trip_evidence={"tool": "search_files"})

    path = engine.recover(snapshot)
    assert path.instruction, "a provider outage must still yield usable steering"
    assert path.backend == "mock", "should have fallen back to the offline steerer"


def test_offline_switch_blocks_billing_but_not_local_models(monkeypatch):
    """AGENTFUSE_OFFLINE means "do not spend money", not "do not think".

    A local ONNX model bills nothing and touches no network, so offline mode
    must not disable it — conflating the two would force the weakest signal on
    every CI run for no benefit. What offline must guarantee is that the HOSTED
    backends stay unused even when a key is present.
    """
    from agentfuse.embedding import openai_embedder
    from agentfuse.recovery import RecoveryEngine

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-looks-real-enough")
    monkeypatch.setenv("AGENTFUSE_OFFLINE", "1")

    assert RecoveryEngine().backend == "mock", "offline must not call a hosted model"
    assert openai_embedder() is None, "offline must not call hosted embeddings"


def test_local_embeddings_are_preferred_over_hosted(monkeypatch):
    """Free, offline and ~4ms beats billed and network-bound on a hot path."""
    from agentfuse.embedding import get_embedder, local_embedder

    if local_embedder() is None:
        pytest.skip("fastembed not installed")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-looks-real-enough")
    monkeypatch.delenv("AGENTFUSE_OFFLINE", raising=False)
    _, mode = get_embedder()
    assert mode == "embedding:local"


# ------------------------------------------------- action grounding (section 3.9)
def _act(det: DriftDetector, tool: str, args: dict, step: int = 1):
    return det.inspect(AgentEvent(type=EventType.TOOL_CALL, step=step,
                                  tool_name=tool, tool_args=args), [])


_DRIFTING_PROSE = [
    "Reviewing competitor advertising spend across social channels.",
    "Analysing influencer campaign performance and social reach.",
    "Drafting a marketing content calendar for social campaigns.",
]


def test_prose_drift_with_on_goal_actions_is_suppressed():
    """The real false positive, reduced to its mechanism.

    An agent narrating repeated tool failures reads as divergent while doing
    exactly what it was asked. Halting it destroys the one result a human needed
    -- in the captured case, the finding that the credential does not exist.
    """
    d = _make()
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        _act(d, "read_report", {"section": "revenue figures"}, i + 1)
        trip = trip or _probe(d, text, i + 1)
    assert trip is None, "drift halted an agent whose actions never left the goal"
    assert d.suppressed_trips > 0, "suppression happened but was not counted"


def test_prose_drift_with_off_goal_actions_still_trips():
    """The whole point of drift. Grounding must not become a blanket amnesty."""
    d = _make()
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        _act(d, "book_meeting", {"room": "Zurich", "time": "14:00"}, i + 1)
        trip = trip or _probe(d, text, i + 1)
    assert trip is not None, "genuine drift was suppressed by the grounding check"
    assert trip.detector == "drift"


def test_pure_reasoning_drift_still_trips_without_any_actions():
    """An agent that has taken no action has no corroborating signal to offer,
    so prose remains the only evidence and must still be acted on."""
    d = _make()
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        trip = trip or _probe(d, text, i + 1)
    assert trip is not None, "drift stopped working for agents that only reason"


def test_grounding_follows_the_latest_action_not_the_first():
    """An agent that starts on-goal and wanders off must lose its amnesty."""
    d = _make()
    _act(d, "read_report", {"section": "revenue figures"}, 1)
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        _act(d, "book_meeting", {"room": "Zurich"}, i + 2)
        trip = trip or _probe(d, text, i + 2)
    assert trip is not None, "grounding from an early action masked later drift"


def test_grounding_can_be_switched_off():
    d = _make(action_grounding=False)
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        _act(d, "read_report", {"section": "revenue figures"}, i + 1)
        trip = trip or _probe(d, text, i + 1)
    assert trip is not None, "action_grounding=False did not restore prose-only drift"


def test_generic_goal_words_are_not_anchors():
    """Without this, `write_file path ./logs/out.txt` matches the "file" in
    almost any goal and every drifted run looks grounded."""
    from agentfuse.detectors.drift import _goal_anchors
    anchors = _goal_anchors("Find the connection file under ./config and read "
                            "the current secret, then store a new value.")
    assert "config" in anchors and "connection" in anchors and "secret" in anchors
    for generic in ("find", "file", "read", "current", "store", "value", "new"):
        assert generic not in anchors, f"{generic!r} is too generic to anchor on"


def test_anchors_keep_path_like_identifiers():
    from agentfuse.detectors.drift import _goal_anchors
    anchors = _goal_anchors("Rotate prod/db/primary described in ./config/db.conn")
    assert any("config" in a for a in anchors)
    assert any("prod" in a for a in anchors)


# ------------------------------------------ tool continuity (the second signal)
#
# `fetch_page {"url": "a.com"}` shares no token with GOAL, so the anchor signal
# cannot ground it. That is the whole point: it stands in for every goal whose
# tools take opaque arguments, where the lexical check is measurably useless.

def _opaque_run(tools):
    d = _make()
    _probe(d, "Summarize the quarter revenue figures from the report.", 1)
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        _act(d, tools[i], {"url": "a.com"}, i * 2 + 2)
        trip = trip or _probe(d, text, i * 2 + 3)
    return d, trip


def test_opaque_arguments_still_ground_via_tool_continuity():
    """The measured blind spot in anchor matching.

    A goal like "research the top three competitors" produces anchors that a real
    call such as `web_search {"url": "a.com"}` can never match, leaving the same
    false positive open for every goal whose tools take opaque arguments.
    Continuing to use the tool the agent was already using while healthy is the
    evidence that closes it.
    """
    d, trip = _opaque_run(["fetch_page"] * 3)
    assert "fetch_page" in d._on_goal_tools, "on-goal tool was never learned"
    assert trip is None, "drift halted an agent still using its on-goal tool"
    assert d.suppressed_trips > 0


def test_switching_to_new_tools_still_trips_under_opaque_arguments():
    d, trip = _opaque_run(["fetch_page", "book_meeting", "send_email"])
    assert trip is not None, "tool continuity suppressed a genuine tool switch"


def test_tools_are_not_learned_while_the_trend_is_already_low():
    """Otherwise a drifting agent's new tools grant themselves amnesty."""
    d = _make()
    for i, text in enumerate(_DRIFTING_PROSE):
        _probe(d, text, i + 1)
    _act(d, "book_meeting", {"room": "Zurich"}, 9)
    assert "book_meeting" not in d._on_goal_tools


def test_stale_grounding_does_not_grant_permanent_amnesty():
    """Regression: drift_abrupt_hijack stopped tripping.

    The agent made ONE on-goal call during its healthy opening turn and then
    drifted purely in reasoning, taking no further action. That single stale call
    was suppressing every turn after it. Corroboration has to mean the agent is
    still acting, so it expires.
    """
    d = _make()
    _probe(d, "Summarize the quarter revenue figures from the report.", 1)
    _act(d, "fetch_page", {"url": "a.com"}, 2)
    trip = None
    for i, text in enumerate(_DRIFTING_PROSE):
        trip = trip or _probe(d, text, i + 3)
    assert trip is not None, "one stale on-goal call suppressed all later drift"
