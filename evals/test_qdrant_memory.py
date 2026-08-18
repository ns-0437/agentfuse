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


# ------------------------------------------------- the persistent path (path=)
# Every test above uses `:memory:`, which is why both bugs below survived: they
# are invisible unless the store is reopened. `path=` is the mode anyone running
# agents across restarts would actually use.

def test_point_ids_are_stable_across_processes():
    """This was `abs(hash(record_id)) % 2**63`.

    Python randomises string hashing per process, so the same record got a
    different point id after every restart. In `:memory:` nothing notices; with
    `path=`, mark_outcome after a restart upserts a SECOND point instead of
    updating the first, leaving the stale worked=None copy behind — so the
    ladder can read a rung it has already disproved as untried.
    """
    import subprocess
    import sys as _sys
    code = ("import sys; sys.path.insert(0, r'%s');"
            "from agentfuse.memory import QdrantMemory;"
            "print(QdrantMemory._point_id('rec-abc-123'))" % str(ROOT))
    ids = {subprocess.run([_sys.executable, "-c", code], capture_output=True,
                          text=True).stdout.strip() for _ in range(3)}
    assert len(ids) == 1, f"point id is not stable across processes: {ids}"


def test_failed_strategies_survive_a_restart(tmp_path):
    """The lesson has to outlive the process that learned it.

    Measured before the fix: {'re-anchor'} before a restart, set() after — while
    the record was still in the collection with worked=False. A memory that
    silently forgets which corrections already failed is worse than no memory,
    because the engine swallows memory faults by design and the ladder simply
    stops climbing.
    """
    path = str(tmp_path / "qd")
    mem = QdrantMemory(embedder=embed, path=path)
    rid = mem.remember(_rec(sig="sig-a", strategy="re-anchor"))
    mem.mark_outcome(rid, False)
    assert mem.failed_strategies("sig-a") == {"re-anchor"}
    mem._client.close()

    reopened = QdrantMemory(embedder=embed, path=path)
    assert reopened.failed_strategies("sig-a") == {"re-anchor"}
    assert len(reopened.recall("sig-a")) == 1
    reopened._client.close()


def test_marking_an_outcome_after_a_restart_updates_rather_than_duplicates(tmp_path):
    """The consequence of an unstable point id, stated as behaviour."""
    path = str(tmp_path / "qd")
    mem = QdrantMemory(embedder=embed, path=path)
    rid = mem.remember(_rec(sig="sig-a", strategy="re-anchor"))
    mem._client.close()

    reopened = QdrantMemory(embedder=embed, path=path)
    reopened.mark_outcome(rid, False)
    hits = reopened.recall_similar("loop", "search_files",
                                   "stop the search loop repeat", limit=20)
    assert len(hits) == 1, f"restart duplicated the record: {len(hits)} points"
    assert hits[0].worked is False
    reopened._client.close()


def test_reopening_an_empty_store_does_not_crash(tmp_path):
    """Rehydration runs on every construction, including the first."""
    mem = QdrantMemory(embedder=embed, path=str(tmp_path / "qd"))
    assert mem.recall("nothing") == []
    assert mem.failed_strategies("nothing") == set()
    mem._client.close()


def test_rehydration_paginates_beyond_one_scroll_page(tmp_path):
    """A single scroll(limit=N) returns the first N and drops the rest.

    A PARTIAL memory is strictly worse than an empty one: the ladder would
    believe it knows what has already been tried while missing exactly the
    records that would have stopped it repeating a failed correction.
    """
    path = str(tmp_path / "qd")
    mem = QdrantMemory(embedder=embed, path=path)
    ids = [mem.remember(_rec(sig=f"sig-{i}", strategy=f"st-{i}"))
           for i in range(1200)]          # > the 1000-record page size
    for rid in ids:
        mem.mark_outcome(rid, False)
    mem._client.close()

    reopened = QdrantMemory(embedder=embed, path=path)
    assert len(reopened._by_id) == 1200
    assert reopened.failed_strategies("sig-1199") == {"st-1199"}
    reopened._client.close()


def test_legacy_duplicate_points_do_not_resurrect_an_untried_verdict(tmp_path):
    """A store written before point ids were stable holds legacy ids.

    After an upgrade, mark_outcome upserts at the corrected id and the stale
    point survives. Returning both shows the ladder one rung as simultaneously
    tried and untried — and the stale copy is the one carrying worked=None,
    which is the wrong one to believe.
    """
    from qdrant_client.models import PointStruct
    path = str(tmp_path / "qd")
    mem = QdrantMemory(embedder=embed, path=path)
    rec = _rec(sig="sig-a", strategy="re-anchor")
    mem.remember(rec)
    mem.mark_outcome(rec.record_id, False)

    # Simulate the legacy point: same payload, worked=None, an unrelated id.
    stale = rec.to_dict()
    stale["worked"] = None
    mem._client.upsert(collection_name=mem._collection, points=[PointStruct(
        id=424242, vector=embed(mem._text_for(rec)), payload=stale)])

    hits = mem.recall_similar("loop", "search_files",
                              "stop the search loop repeat", limit=20)
    assert len(hits) == 1, f"duplicate record returned twice: {len(hits)}"
    assert hits[0].worked is False, "the stale untried copy won"
    mem._client.close()
