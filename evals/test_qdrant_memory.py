"""Exercise QdrantMemory — the backend that had never once been executed.

Roughly eighty lines of it shipped unrun. Running it for the first time found two
bugs, both of the kind that only dead code hides:

  1. ``failed_strategies`` was missing entirely. It was added to JSONMemory when
     the recency-window bug was fixed and never mirrored here. Nothing would have
     failed loudly: the engine swallows memory faults by design, so the ladder
     would simply never climb and the same correction would be reissued forever.
  2. ``recall_similar`` — the entire reason this backend exists — called
     ``QdrantClient.search()``, which current clients no longer have.

Both are now covered. A deterministic synthetic embedder is used so this runs
offline and free; the geometry (related text nearby, unrelated text far) is the
property a real embedding model provides.

    pytest evals/test_qdrant_memory.py -v
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

pytest.importorskip("qdrant_client", reason="pip install agentfuse[memory]")

from agentfuse.memory import QdrantMemory, RecoveryRecord  # noqa: E402
from agentfuse.recovery import RecoveryEngine  # noqa: E402
from agentfuse.events import ExecutionSnapshot  # noqa: E402
from agentfuse.strategies import LADDER, ESCALATE  # noqa: E402

_TOPICS = {"loop": (1, 0), "repeat": (.95, .05), "stuck": (.9, .1), "search": (.85, .1),
           "grep": (.85, .1), "drift": (0, 1), "wander": (.05, .95),
           "objective": (.1, .9), "topic": (.05, .9)}


def embed(text: str) -> list[float]:
    x = y = 0.0
    for w in str(text).lower().replace(":", " ").split():
        if w in _TOPICS:
            x += _TOPICS[w][0]
            y += _TOPICS[w][1]
    if x == y == 0.0:
        x = y = 0.5
    n = math.sqrt(x * x + y * y)
    return [x / n, y / n]


@pytest.fixture
def mem():
    return QdrantMemory(embedder=embed)


def _rec(sig="sig-a", strategy="re-anchor", tool="search_files",
         goal="stop the search loop repeat", detector="loop"):
    return RecoveryRecord(signature=sig, detector=detector, goal=goal,
                          strategy=strategy, instruction=f"instruction for {strategy}",
                          tool=tool)


def test_backend_constructs_without_a_server():
    """Embedded mode: a library, not a service."""
    m = QdrantMemory(embedder=embed)
    assert m._dim == 2


def test_failed_strategies_exists_and_works(mem):
    """Bug 1: this method was missing, and its absence was silent."""
    rid = mem.remember(_rec())
    mem.mark_outcome(rid, worked=False)
    assert mem.failed_strategies("sig-a") == {"re-anchor"}


def test_unverified_records_are_not_treated_as_failed(mem):
    mem.remember(_rec())
    assert mem.failed_strategies("sig-a") == set()


def test_failed_instructions_round_trip(mem):
    rid = mem.remember(_rec())
    mem.mark_outcome(rid, worked=False)
    assert mem.failed_instructions("sig-a") == ["instruction for re-anchor"]


def test_recall_similar_works_on_current_clients(mem):
    """Bug 2: this called a client method that no longer exists."""
    mem.remember(_rec())
    mem.remember(_rec(sig="sig-b", strategy="decompose", tool="think",
                      goal="agent wander off objective topic", detector="drift"))
    hits = mem.recall_similar("loop", "grep_files", "stuck repeat search loop", limit=2)
    assert hits, "semantic recall returned nothing"
    assert hits[0].detector == "loop", "nearest neighbour was not the loop record"


def test_recall_similar_crosses_tool_boundaries(mem):
    """The whole point: exact signature matching cannot do this."""
    mem.remember(_rec(tool="search_files"))
    hits = mem.recall_similar("loop", "grep_files", "stuck repeat loop", limit=1)
    assert hits[0].tool == "search_files"


def test_ladder_climbs_with_this_backend():
    """End to end: the bug above would have frozen the ladder on rung one."""
    engine = RecoveryEngine(backend="mock", memory=QdrantMemory(embedder=embed))
    snap = ExecutionSnapshot(
        step=3, original_goal="Rotate the production database credential.",
        current_goal=None, total_tokens=10, total_cost_usd=0.0,
        route_history=["agent"], recent_events=[], trip_reason="looping",
        trip_detector="loop", trip_evidence={"tool": "search_files"})

    seen = []
    for _ in range(len(LADDER)):
        path = engine.recover(snap)
        seen.append(path.strategy)
        engine.verify(path, worked=False)

    assert seen[:4] == LADDER[:4], f"ladder did not climb: {seen}"
    assert seen[-1] == ESCALATE
    assert len(set(seen)) == len(seen), f"a rung was retried: {seen}"


def test_backends_expose_the_same_interface():
    """A backend missing a method fails silently, so check the shape explicitly."""
    from agentfuse.memory import JSONMemory
    required = ("remember", "recall", "mark_outcome",
                "failed_instructions", "failed_strategies", "strategies_tried")
    for backend in (JSONMemory(), QdrantMemory(embedder=embed)):
        missing = [m for m in required if not callable(getattr(backend, m, None))]
        assert not missing, f"{type(backend).__name__} missing {missing}"
