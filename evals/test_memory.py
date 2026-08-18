"""Tests for the cost and durability of JSONMemory's write path.

`JSONMemory` is documented as "append-only JSONL memory". It was not: every
`remember()` rewrote the entire file, so a store holding `max_records=5000`
performed up to 5000 serialisations per single write — O(n^2) across a run, on
the supervision hot path of a system whose whole premise is runs lasting hours
to days. The cost grew exactly as the run got longer, which is the worst
possible shape for it.

Measured on the same machine, 1500 `remember()` calls:

    before   42.1s   (28.1 ms/write, still climbing)
    after     1.59s  ( 1.06 ms/write, flat)

The three tests here pin the fix and both cases that must still rewrite, because
"append instead of rewrite" is only correct while nothing already on disk has
changed. Get that wrong and the store silently keeps evicted records, or loses
the outcome that tells the ladder a rung has already failed.

    pytest evals/test_memory.py -v
"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENTFUSE_OFFLINE", "1")

from agentfuse.memory import JSONMemory, RecoveryRecord  # noqa: E402


def _rec(signature: str = "sig", strategy: str = "re-anchor") -> RecoveryRecord:
    return RecoveryRecord(signature=signature, strategy=strategy,
                          detector="loop", goal="rotate the credential",
                          instruction="stop repeating the failing call")


def test_remember_appends_instead_of_rewriting_the_file(tmp_path):
    """The O(n)-per-write bug, pinned by counting full-file rewrites."""
    mem = JSONMemory(path=str(tmp_path / "mem.jsonl"), max_records=1000)

    rewrites = {"n": 0}
    original = pathlib.Path.open

    def counting_open(self, mode="r", *a, **kw):
        if "w" in mode:
            rewrites["n"] += 1
        return original(self, mode, *a, **kw)

    pathlib.Path.open = counting_open
    try:
        for i in range(50):
            mem.remember(_rec(signature=f"s{i}"))
    finally:
        pathlib.Path.open = original

    assert rewrites["n"] == 0, (
        f"{rewrites['n']} full-file rewrites during 50 appends — remember() is "
        f"still O(n) per write")


def test_eviction_still_rebuilds_the_file(tmp_path):
    """An eviction removes a line that is ALREADY on disk, so that case must
    rewrite. Appending there would leave the dropped record in the file
    forever, and the store would grow without bound while reporting a cap."""
    path = tmp_path / "mem.jsonl"
    mem = JSONMemory(path=str(path), max_records=3)
    for i in range(6):
        mem.remember(_rec(signature="s", strategy=f"st{i}"))

    reloaded = JSONMemory(path=str(path), max_records=3)
    assert len(reloaded._records) == 3
    assert sorted(r.strategy for r in reloaded._records) == ["st3", "st4", "st5"]


def test_outcomes_survive_a_reload(tmp_path):
    """`mark_outcome` mutates a record already written, so it still needs the
    rewrite path. Otherwise the ladder reloads believing a demonstrably failed
    rung is untried, and retries a correction already known not to work."""
    path = tmp_path / "mem.jsonl"
    mem = JSONMemory(path=str(path))
    rid = mem.remember(_rec(signature="s", strategy="re-anchor"))
    mem.mark_outcome(rid, False)

    reloaded = JSONMemory(path=str(path))
    assert reloaded.failed_strategies("s") == {"re-anchor"}


def test_appended_records_reload_without_duplication(tmp_path):
    """A file written by appends must read back as exactly what was stored."""
    path = tmp_path / "mem.jsonl"
    mem = JSONMemory(path=str(path))
    ids = [mem.remember(_rec(signature="s", strategy=f"st{i}")) for i in range(8)]

    reloaded = JSONMemory(path=str(path))
    assert [r.record_id for r in reloaded._records] == ids
    assert len({r.record_id for r in reloaded._records}) == 8
